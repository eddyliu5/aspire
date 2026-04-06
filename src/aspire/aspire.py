import os, json, logging, math, random
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from huggingface_hub import hf_hub_download
from sklearn.exceptions import NotFittedError
from .model import ASPIREEnhanced, Example, train_examples
from .data_loader import (
    build_examples_from_feature_specs,
    build_training_examples_from_feature_specs,
    build_prediction_dataframe,
    df_to_examples,
    training_metadata_from_feature_specs,
    validate_feature_specs,
)
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureProjection(nn.Module):
    """2-layer MLP projecting raw numeric features into model_dim space."""

    def __init__(self, in_dim: int, model_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParallelMLP(nn.Module):
    """Direct raw-feature → logits branch, ensembled with the ASPIRE cosine head."""

    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss that down-weights easy examples."""
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _get_lr_scale(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup then cosine decay, returns multiplicative scale in (0, 1]."""
    if epoch < warmup_epochs:
        return float(epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _class_balanced_support_indices(
    train_labels: List[int],
    n_train: int,
    num_support: int,
    seed: int,
    n_classes: int,
) -> List[int]:
    """Return indices of `num_support` examples with equal class representation."""
    rng = random.Random(seed)
    per_class_indices: Dict[int, List[int]] = {c: [] for c in range(n_classes)}
    for idx, lbl in enumerate(train_labels):
        if lbl in per_class_indices:
            per_class_indices[lbl].append(idx)

    per_class = max(1, num_support // n_classes)
    support: List[int] = []
    for c in range(n_classes):
        pool = per_class_indices[c]
        if not pool:
            continue
        chosen = rng.choices(pool, k=per_class)
        support.extend(chosen)
    return support[:num_support]

class AspireModel(torch.nn.Module):
    """User-facing wrapper for model."""

    def __init__(
        self,
        config: dict,
        feature_specs: Optional[Sequence[Mapping[str, Any]]] = None,
        dataset_context: str = "",
        target_indices: Optional[Sequence[int]] = None,
        target_column: str = "__target__",
    ):
        super().__init__()
        self.config = config
        self.model = ASPIREEnhanced(**config)
        self.feature_specs_ = validate_feature_specs(feature_specs) if feature_specs is not None else []
        self.dataset_context = str(dataset_context or "")
        if target_indices is not None:
            self.target_indices_ = [int(idx) for idx in target_indices]
        elif self.feature_specs_:
            self.target_indices_ = [len(self.feature_specs_) - 1]
        else:
            self.target_indices_ = []
        self._has_loaded_weights = False
        self.is_fitted_ = False
        self.fit_mode_: Optional[str] = None
        self.target_column_: str = target_column
        self.feature_desc_: List[dict] = []
        self.dataset_description_: Optional[str] = None
        self.task_description_: Optional[str] = None
        self.target_description_: Optional[str] = None
        self.feature_names_in_: List[str] = []

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: str = "best_model.pt",
        config_name: str = "config.json",
        device: str = "cpu",
        feature_specs: Optional[Sequence[Mapping[str, Any]]] = None,
        dataset_context: str = "",
        target_indices: Optional[Sequence[int]] = None,
        target_column: str = "__target__",
    ):
        """
        Load model + config from Hugging Face hub or local path.

        Preferred setup for checkpoint usage is `feature_specs` plus optional
        `dataset_context` and `target_indices` (defaults to the last column).
        """
        if os.path.isdir(repo_id):
            cfg_path = os.path.join(repo_id, config_name)
            weights_path = os.path.join(repo_id, filename)
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"Missing config file: {cfg_path}")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing weights file: {weights_path}")
        else:
            cfg_path = hf_hub_download(repo_id=repo_id, filename=config_name)
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        checkpoint_payload = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint_payload, dict):
            for key in (
                "model_dim",
                "num_heads",
                "num_inds",
                "mask_prob",
                "max_targets",
                "intra_layers",
                "inter_layers",
                "shared_bert",
                "use_intra_set2set",
                "use_dataset_description",
                "use_echoices",
            ):
                if key in checkpoint_payload:
                    cfg[key] = checkpoint_payload[key]

        model = cls(
            cfg,
            feature_specs=feature_specs,
            dataset_context=dataset_context,
            target_indices=target_indices,
            target_column=target_column,
        ).to(device)
        state_dict = checkpoint_payload

        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        incompatible = model.model.load_state_dict(state_dict, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing:
            logger.warning("Missing keys when loading checkpoint: %d", len(missing))
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %d", len(unexpected))
        model._has_loaded_weights = True
        model.eval()
        return model

    def save_pretrained(self, save_dir: str = "./checkpoints", filename: str = "best_model.pt"):
        """Save current model weights and config."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, filename))
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
    
    @torch.no_grad()
    def predict(self, data: Any, target: Optional[List[str]] = None, batch_size: int = 32):
        """Run prediction on an Example or a pandas DataFrame."""
        self.model.eval()

        if isinstance(data, Example):
            return self.model.predict(data)

        if isinstance(data, pd.DataFrame) and self.fit_mode_ == "head_v2":
            return self._predict_head_v2(data)

        if isinstance(data, pd.DataFrame):
            target = target or []

            if "feature_desc" in data.attrs and "dataset_desc" in data.attrs:
                prediction_df = data
                examples = df_to_examples(prediction_df, target)
            elif self.feature_specs_:
                examples = build_examples_from_feature_specs(
                    X=data,
                    feature_specs=self.feature_specs_,
                    dataset_context=self.dataset_context,
                    target_indices=self.target_indices_ if self.target_indices_ else None,
                )
            elif self.is_fitted_ and self.feature_desc_ and self.dataset_description_ is not None:
                prediction_df = build_prediction_dataframe(
                    X=data,
                    feature_desc=self.feature_desc_,
                    dataset_desc=self.dataset_description_,
                    target_column=self.target_column_,
                )
                examples = df_to_examples(prediction_df)
            elif target:
                examples = df_to_examples(data, target)
            else:
                raise NotFittedError(
                    "predict on a raw DataFrame requires either feature_specs (set at init), "
                    "a fitted model, or a DataFrame with metadata attrs."
                )

            preds = []
            step = max(1, batch_size)
            for idx in range(0, len(examples), step):
                for example in examples[idx:idx + step]:
                    prediction = self.model.predict(example)
                    if isinstance(prediction, list) and len(prediction) == 1:
                        preds.append(prediction[0])
                    else:
                        preds.append(prediction)
            return preds

        raise TypeError("predict expects an Example or pandas.DataFrame")

    def fit(
        self,
        X: Any,
        y: Sequence[Any],
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        random_state: Optional[int] = None,
        show_progress: bool = True,
        finetune_mode: str = "default",
        num_support: int = 5,
        test_fraction: float = 0.15,
        focal_gamma: float = 2.0,
        warmup_epochs: int = 5,
        label_descriptions: Optional[Dict[str, str]] = None,
    ) -> "AspireModel":
        if not self.feature_specs_:
            raise ValueError(
                "feature_specs is required. Pass it when initializing AspireModel."
            )

        examples = build_training_examples_from_feature_specs(
            X=X,
            y=y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_indices=self.target_indices_ if self.target_indices_ else None,
        )
        normalized_metadata = training_metadata_from_feature_specs(
            X=X,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_indices=self.target_indices_ if self.target_indices_ else None,
        )
        if not examples:
            raise ValueError("No valid training examples could be built from X/y.")
        self.feature_desc_ = [vars(feature) for feature in examples[0].features]

        if finetune_mode == "head_v2":
            if not self._has_loaded_weights:
                raise ValueError("finetune_mode='head_v2' requires a pre-loaded checkpoint.")
            self.fit_mode_ = "head_v2"
            target_name_set = {self.feature_desc_[idx]["name"] for idx in self.target_indices_ if 0 <= idx < len(self.feature_desc_)}
            self.feature_names_in_ = [f["name"] for f in self.feature_desc_ if f["name"] not in target_name_set]
            self.dataset_description_ = normalized_metadata["dataset_description"]
            self.task_description_ = normalized_metadata["task_description"]
            self.target_description_ = normalized_metadata["target_description"]
            self.is_fitted_ = True
            return self._fit_head_v2(
                X=X, y=y,
                num_epochs=num_epochs, batch_size=batch_size,
                num_support=num_support, random_state=random_state,
                show_progress=show_progress, test_fraction=test_fraction,
                focal_gamma=focal_gamma, warmup_epochs=warmup_epochs,
                label_descriptions=label_descriptions,
            )

        self.fit_mode_ = "finetune" if self._has_loaded_weights else "scratch"

        if self.fit_mode_ == "finetune":
            if learning_rate == 1e-3:
                resolved_lr_head = 1e-4
                resolved_lr_backbone = 1e-5
            else:
                resolved_lr_head = learning_rate
                resolved_lr_backbone = learning_rate
            freeze_bert = True
            resolved_num_support = num_support
            val_fraction = 0.2
            patience = 5
        else:
            resolved_lr_head = learning_rate
            resolved_lr_backbone = learning_rate
            freeze_bert = False
            resolved_num_support = 0
            val_fraction = 0.0
            patience = 0

        train_examples(
            model=self.model,
            examples=examples,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
            lr_head=resolved_lr_head,
            lr_backbone=resolved_lr_backbone,
            freeze_bert=freeze_bert,
            num_support=resolved_num_support,
            val_fraction=val_fraction,
            patience=patience,
            show_progress=show_progress,
        )

        target_name_set = {self.feature_desc_[idx]["name"] for idx in self.target_indices_ if 0 <= idx < len(self.feature_desc_)}
        self.feature_names_in_ = [f["name"] for f in self.feature_desc_ if f["name"] not in target_name_set]
        self.dataset_description_ = normalized_metadata["dataset_description"]
        self.task_description_ = normalized_metadata["task_description"]
        self.target_description_ = normalized_metadata["target_description"]
        self.is_fitted_ = True
        return self

    def _fit_head_v2(
        self,
        X: pd.DataFrame,
        y: Sequence[Any],
        num_epochs: int = 30,
        batch_size: int = 32,
        num_support: int = 5,
        random_state: Optional[int] = None,
        show_progress: bool = True,
        test_fraction: float = 0.15,
        focal_gamma: float = 2.0,
        warmup_epochs: int = 5,
        label_descriptions: Optional[Dict[str, str]] = None,
    ) -> "AspireModel":

        device = next(self.model.parameters()).device
        seed = random_state if random_state is not None else 42

        target_idx = self.target_indices_[0]
        target_spec = self.feature_specs_[target_idx]
        classes: List[str] = list(target_spec.get("choices") or sorted(set(str(v) for v in y)))
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        labels: List[int] = [class_to_idx.get(str(v), 0) for v in y]

        n_total = len(labels)
        n_val = max(1, int(n_total * test_fraction))
        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_arr = X if isinstance(X, np.ndarray) else X.values
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr.astype(np.float32))
        self._hv2_scaler = scaler
        self._hv2_feat_dim = X_scaled.shape[1]
        self._hv2_classes = classes
        self._hv2_class_to_idx = class_to_idx

        for name, param in self.model.named_parameters():
            freeze = any(name.startswith(p) for p in (
                "shared_text.", "semantic_grounding.", "atom_processing."
            ))
            param.requires_grad_(not freeze)

        feat_proj = FeatureProjection(self._hv2_feat_dim, self.model.model_dim).to(device)
        parallel_mlp = ParallelMLP(self._hv2_feat_dim, n_classes).to(device)

        label_counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
        label_counts = np.where(label_counts == 0, 1.0, label_counts)
        class_weights = torch.tensor(1.0 / label_counts, dtype=torch.float32, device=device)
        class_weights = class_weights / class_weights.sum() * n_classes

        if label_descriptions:
            label_texts = [label_descriptions.get(c, c) for c in classes]
        else:
            label_texts = classes
        with torch.no_grad():
            raw_vecs = self.model._get_label_vecs(label_texts, device)
            label_embs = self.model.cls_head.category_proj(raw_vecs)
        label_embs = label_embs.detach()

        examples = build_training_examples_from_feature_specs(
            X=X, y=y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_indices=self.target_indices_ if self.target_indices_ else None,
        )
        feats = examples[0].features

        with torch.no_grad():
            phi_cache = torch.stack([self.model.semantic_grounding(f, device) for f in feats], dim=0)

        self.model.eval()
        with torch.no_grad():
            atom_cache: List[torch.Tensor] = []
            for ex in examples:
                atoms = torch.stack([
                    self.model.atom_processing(f, v, phi_cache[i], device)
                    for i, (f, v) in enumerate(zip(ex.features, ex.values))
                ], dim=0)
                atom_cache.append(atoms.cpu())

        with torch.no_grad():
            ctx_tokens = self.model._encode_context(self.dataset_context, device)
            tgt_tokens = self.model._encode_target_description(feats, [target_idx], device)

        base_lrs = [1e-4, 1e-3, 1e-3]
        trainable_params = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad], "lr": base_lrs[0]},
            {"params": list(feat_proj.parameters()), "lr": base_lrs[1]},
            {"params": list(parallel_mlp.parameters()), "lr": base_lrs[2]},
        ]
        optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-2)

        best_val_f1 = -1.0
        best_state: Optional[Dict] = None

        from tqdm.auto import tqdm

        progress = tqdm(total=num_epochs, desc="head_v2", unit="epoch") if show_progress else None

        for epoch in range(num_epochs):
            scale = _get_lr_scale(epoch, warmup_epochs, num_epochs)
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * scale

            self.model.train()
            feat_proj.train()
            parallel_mlp.train()

            rng2 = random.Random(seed + epoch)
            rng2.shuffle(train_idx)
            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, len(train_idx), batch_size):
                batch_indices = train_idx[b_start: b_start + batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=device)

                sup_indices = _class_balanced_support_indices(
                    train_labels, len(train_idx), num_support, seed + epoch + b_start, n_classes
                )
                sup_atoms_list = [atom_cache[i].to(device) for i in sup_indices]

                for i in batch_indices:
                    atoms = atom_cache[i].to(device)
                    atoms_intra = self.model._apply_intra_set2set(atoms)
                    q_idx = [j for j in range(len(feats)) if j != target_idx]
                    query_atoms = atoms_intra[q_idx]
                    target_atoms = atoms_intra[[target_idx]]

                    h = self.model.inter_aggregator(
                        query_atoms=query_atoms,
                        target_atoms=target_atoms,
                        support_atoms=sup_atoms_list,
                        context_data_tokens=ctx_tokens,
                        context_target_tokens=tgt_tokens,
                    )[0]

                    aspire_logits = self.model.cls_head.logits(h.unsqueeze(0), label_embs).squeeze(0)
                    x_i = torch.tensor(X_scaled[i], dtype=torch.float32, device=device).unsqueeze(0)
                    mlp_logits = parallel_mlp(x_i).squeeze(0)
                    combined = aspire_logits + mlp_logits

                    lbl = torch.tensor(labels[i], dtype=torch.long, device=device)
                    loss = _focal_loss(combined.unsqueeze(0), lbl.unsqueeze(0), class_weights, focal_gamma)
                    batch_loss = batch_loss + loss

                if len(batch_indices) > 0:
                    (batch_loss / len(batch_indices)).backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(feat_proj.parameters()) + list(parallel_mlp.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    epoch_loss += batch_loss.item() / len(batch_indices)
                    n_batches += 1

            self.model.eval()
            feat_proj.eval()
            parallel_mlp.eval()
            val_preds = []
            with torch.no_grad():
                sup_indices_val = _class_balanced_support_indices(
                    train_labels, len(train_idx), num_support, seed + 9999, n_classes
                )
                sup_atoms_val = [atom_cache[i].to(device) for i in sup_indices_val]

                for i in val_idx:
                    atoms = atom_cache[i].to(device)
                    atoms_intra = self.model._apply_intra_set2set(atoms)
                    q_idx = [j for j in range(len(feats)) if j != target_idx]
                    h = self.model.inter_aggregator(
                        query_atoms=atoms_intra[q_idx],
                        target_atoms=atoms_intra[[target_idx]],
                        support_atoms=sup_atoms_val,
                        context_data_tokens=ctx_tokens,
                        context_target_tokens=tgt_tokens,
                    )[0]
                    aspire_logits = self.model.cls_head.logits(h.unsqueeze(0), label_embs).squeeze(0)
                    x_v = torch.tensor(X_scaled[i], dtype=torch.float32, device=device).unsqueeze(0)
                    mlp_logits = parallel_mlp(x_v).squeeze(0)
                    combined = aspire_logits + mlp_logits
                    val_preds.append(int(combined.argmax().item()))

            val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {
                    "model": deepcopy(self.model.state_dict()),
                    "feat_proj": deepcopy(feat_proj.state_dict()),
                    "parallel_mlp": deepcopy(parallel_mlp.state_dict()),
                }

            if progress is not None:
                progress.set_postfix({"loss": f"{epoch_loss/max(1,n_batches):.4f}", "val_f1": f"{val_f1:.4f}"})
                progress.update(1)

        if progress is not None:
            progress.close()

        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            feat_proj.load_state_dict(best_state["feat_proj"])
            parallel_mlp.load_state_dict(best_state["parallel_mlp"])

        self._hv2_feat_proj = feat_proj
        self._hv2_parallel_mlp = parallel_mlp
        self._hv2_label_embs = label_embs
        self._hv2_ctx_tokens = ctx_tokens
        self._hv2_tgt_tokens = tgt_tokens
        self._hv2_feats = feats
        self._hv2_target_idx = target_idx
        self._hv2_train_labels = train_labels
        self._hv2_train_idx = train_idx
        self._hv2_atom_cache = atom_cache
        self._hv2_num_support = num_support
        self._hv2_n_classes = n_classes
        self._hv2_seed = seed

        self.model.eval()
        return self

    def _predict_head_v2(self, X: pd.DataFrame) -> List[Any]:
        """Predict using the trained head_v2 modules."""
        device = next(self.model.parameters()).device
        X_arr = X if isinstance(X, np.ndarray) else X.values
        X_scaled = self._hv2_scaler.transform(X_arr.astype(np.float32))

        feats = self._hv2_feats
        target_idx = self._hv2_target_idx
        label_embs = self._hv2_label_embs
        ctx_tokens = self._hv2_ctx_tokens
        tgt_tokens = self._hv2_tgt_tokens
        num_support = self._hv2_num_support
        n_classes = self._hv2_n_classes
        classes = self._hv2_classes
        train_labels = self._hv2_train_labels
        train_idx = self._hv2_train_idx
        atom_cache = self._hv2_atom_cache
        seed = self._hv2_seed

        feat_proj = self._hv2_feat_proj
        parallel_mlp = self._hv2_parallel_mlp

        self.model.eval()
        feat_proj.eval()
        parallel_mlp.eval()

        sup_indices = _class_balanced_support_indices(
            train_labels, len(train_idx), num_support, seed + 99999, n_classes
        )
        sup_atoms_list = [atom_cache[i].to(device) for i in sup_indices]

        preds = []
        with torch.no_grad():
            phi_cache = torch.stack(
                [self.model.semantic_grounding(f, device) for f in feats], dim=0
            )

            for row_idx in range(len(X_arr)):
                row = X_arr[row_idx]
                atoms = torch.stack([
                    self.model.atom_processing(feats[j], row[j], phi_cache[j], device)
                    for j in range(len(feats))
                ], dim=0)
                atoms_intra = self.model._apply_intra_set2set(atoms)
                q_idx = [j for j in range(len(feats)) if j != target_idx]
                h = self.model.inter_aggregator(
                    query_atoms=atoms_intra[q_idx],
                    target_atoms=atoms_intra[[target_idx]],
                    support_atoms=sup_atoms_list,
                    context_data_tokens=ctx_tokens,
                    context_target_tokens=tgt_tokens,
                )[0]
                aspire_logits = self.model.cls_head.logits(h.unsqueeze(0), label_embs).squeeze(0)
                x_t = torch.tensor(X_scaled[row_idx], dtype=torch.float32, device=device).unsqueeze(0)
                mlp_logits = parallel_mlp(x_t).squeeze(0)
                combined = aspire_logits + mlp_logits
                preds.append(classes[int(combined.argmax().item())])

        return preds
