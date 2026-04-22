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
from .model import ASPIREEnhanced, ASPIRELite, Example, train_examples
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


class FeatureInjector(nn.Module):

    def __init__(self, raw_dim: int, model_dim: int):
        super().__init__()
        hidden = min(512, max(128, raw_dim // 4))
        self.net = nn.Sequential(
            nn.Linear(raw_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearHead(nn.Module):

    def __init__(self, model_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(model_dim, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _get_lr_scale(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
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

def _balanced_support_indices(
    train_labels: List[int],
    n_train: int,
    n_test: int,
    num_support: int,
    seed: int,
    n_classes: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    random.seed(seed)
    pools = {c: [j for j in range(n_train) if train_labels[j] == c] for c in range(n_classes)}

    def sample(exclude=None):
        per_class = max(1, num_support // n_classes)
        remainder = num_support - per_class * n_classes
        s: List[int] = []
        for c in range(n_classes):
            pool = [j for j in pools[c] if j != exclude]
            k = per_class + (1 if c < remainder else 0)
            if pool:
                s.extend(random.choices(pool, k=min(k, len(pool))))
        random.shuffle(s)
        return s[:num_support]

    train_sup = [sample(exclude=i) for i in range(n_train)]
    test_sup  = [sample() for _ in range(n_test)]
    return train_sup, test_sup


@torch.no_grad()
def _warmup_bert(model, features, dataset_context: str, target_idx: int, class_names: List[str], device):
    for f in features:
        model.shared_text.encode_text(f.description or f.name, is_context=False, device=device)
        if f.choices:
            for c in f.choices:
                model.shared_text.encode_text(str(c), is_context=False, device=device)
    model.shared_text.encode_text("[UNK]", is_context=False, device=device)
    for cn in class_names:
        model.shared_text.encode_text(str(cn), is_context=False, device=device)
    model.shared_text.encode_text_sequence(dataset_context, is_context=True, device=device)
    target_desc = features[target_idx].description or features[target_idx].name
    model.shared_text.encode_text_sequence(target_desc, is_context=False, device=device)


@torch.no_grad()
def _compute_phi(model, features, device) -> torch.Tensor:
    return torch.stack([model.semantic_grounding(f, device) for f in features])


@torch.no_grad()
def _precompute_cat_cache(model, features, device) -> Dict:
    cat_cache: Dict = {}
    for i, f in enumerate(features):
        if f.dtype == "categorical" and f.choices:
            cat_cache[i] = {}
            for choice in f.choices:
                emb = model.shared_text.encode_text(str(choice), is_context=False, device=device)
                cat_cache[i][str(choice)] = model.atom_processing.cat_proj(emb).cpu()
    return cat_cache


@torch.no_grad()
def _vectorized_atoms(
    model, examples, features, phi: torch.Tensor, cat_cache: Dict,
    target_mask: bool = False, batch_size: int = 1024,
) -> torch.Tensor:
    device = next(model.parameters()).device
    ap = model.atom_processing
    M = len(features)
    N = len(examples)
    D = phi.size(-1)
    scale = math.sqrt(model.model_dim)
    target_set = set(examples[0].target_indices) if target_mask else set()
    nu_all = torch.zeros(N, M, D)

    for fi, f in enumerate(features):
        if fi in target_set or f.dtype != "continuous":
            continue
        vals = []
        for ex in examples:
            v = ex.values[fi]
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            if math.isnan(v) or math.isinf(v):
                vals.append(0.5)
                continue
            if f.value_range:
                vmin, vmax = f.value_range
                norm = 0.5 if vmax <= vmin else float(np.clip((v - vmin) / max(1e-8, vmax - vmin), 0, 1))
            else:
                norm = (math.tanh(v / 100.0) + 1.0) / 2.0
            vals.append(norm)
        vals_t = torch.tensor(vals, dtype=torch.float32, device=device)
        ff = torch.cos(2.0 * math.pi * ap.fourier_freqs.unsqueeze(0) * vals_t.unsqueeze(1))
        nu_all[:, fi, :] = ap.fourier_proj(ff).cpu()

    for fi, f in enumerate(features):
        if fi in target_set or f.dtype != "categorical":
            continue
        for n, ex in enumerate(examples):
            val_str = str(ex.values[fi])
            if fi in cat_cache and val_str in cat_cache[fi]:
                nu_all[n, fi, :] = cat_cache[fi][val_str]
            else:
                emb = model.shared_text.encode_text(val_str, is_context=False, device=device)
                nu_all[n, fi, :] = ap.cat_proj(emb).cpu()

    for fi in target_set:
        nu_all[:, fi, :] = ap.missing_embed.detach().cpu()

    phi_exp = phi.unsqueeze(0).expand(N, -1, -1).cpu()
    all_atoms = []
    for start in range(0, N * M, batch_size):
        end = min(start + batch_size, N * M)
        p_b = torch.nan_to_num(phi_exp.reshape(-1, D)[start:end].to(device), nan=0.0)
        n_b = torch.nan_to_num(nu_all.reshape(-1, D)[start:end].to(device), nan=0.0)
        phi_n = F.normalize(p_b, dim=-1, eps=1e-8) * scale
        nu_n  = F.normalize(n_b, dim=-1, eps=1e-8) * scale
        all_atoms.append(ap.atom_mlp(torch.cat([phi_n, nu_n], dim=-1)).cpu())
    return torch.cat(all_atoms).reshape(N, M, D)


@torch.no_grad()
def _cache_h_vectors(
    model, q_atoms: torch.Tensor, s_atoms: torch.Tensor,
    examples, support_indices: List[List[int]],
    ctx_data_tokens: torch.Tensor, ctx_target_tokens: torch.Tensor,
    dataset_context: str, device,
) -> torch.Tensor:
    is_lite = isinstance(model, ASPIRELite)
    N, M, D = q_atoms.shape
    non_tgt = [k for k in range(M) if k not in set(examples[0].target_indices)]

    if is_lite:
        ctx_raw = model.shared_text.encode_text(dataset_context, is_context=True, device=device)
        ctx_emb = model.ctx_proj(ctx_raw).unsqueeze(0)

        ctx_expanded = ctx_emb.unsqueeze(0).expand(N, -1, -1)
        q_with_ctx = torch.cat([ctx_expanded, q_atoms.to(device)], dim=1)
        q_interacted = model.interaction.forward_batch(q_with_ctx)
        q_feat_embs = q_interacted[:, 1:, :]

        all_sup_ids = sorted({si for sup in support_indices for si in sup})
        s_with_ctx = torch.cat([
            ctx_emb.unsqueeze(0).expand(len(all_sup_ids), -1, -1),
            s_atoms[all_sup_ids].to(device),
        ], dim=1)
        s_interacted = model.interaction.forward_batch(s_with_ctx)
        s_feat_map = {sid: s_interacted[k, 1:, :] for k, sid in enumerate(all_sup_ids)}
    else:
        q_feat_embs = model._apply_intra_set2set(q_atoms.to(device))
        all_sup_ids = sorted({si for sup in support_indices for si in sup})
        s_feat_all  = model._apply_intra_set2set(s_atoms[all_sup_ids].to(device))
        s_feat_map  = {sid: s_feat_all[k] for k, sid in enumerate(all_sup_ids)}

    h_list = []
    for i, ex in enumerate(examples):
        q_feat   = q_feat_embs[i]
        sup_feats = [s_feat_map[si] for si in support_indices[i]]
        h = model.inter_aggregator(
            query_atoms=q_feat[non_tgt],
            target_atoms=q_feat[ex.target_indices],
            support_atoms=sup_feats,
            context_data_tokens=ctx_data_tokens,
            context_target_tokens=ctx_target_tokens,
        )[0]
        h_list.append(h.cpu())
    return torch.stack(h_list)


class AspireModel(torch.nn.Module):

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
        _enhanced_keys = {"model_dim", "num_heads", "num_inds", "mask_prob", "max_targets",
                          "intra_layers", "inter_layers", "shared_bert",
                          "use_intra_set2set", "use_dataset_description", "use_echoices"}
        self.model = ASPIREEnhanced(**{k: v for k, v in config.items() if k in _enhanced_keys})
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
        model_type: str = "auto",
    ):
        if repo_id.endswith((".pt", ".pth")) and os.path.isfile(repo_id):
            weights_path = repo_id
            cfg_path = os.path.join(os.path.dirname(repo_id), config_name)
        elif os.path.isdir(repo_id):
            cfg_path = os.path.join(repo_id, config_name)
            weights_path = os.path.join(repo_id, filename)
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing weights file: {weights_path}")
        else:
            cfg_path = hf_hub_download(repo_id=repo_id, filename=config_name)
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

        cfg: Dict[str, Any] = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)

        checkpoint_payload = torch.load(weights_path, map_location=device, weights_only=False)

        aspire_lite_keys = ("interaction_layers",)
        enhanced_keys = ("intra_layers", "use_intra_set2set", "use_dataset_description", "use_echoices")

        if isinstance(checkpoint_payload, dict):
            ckpt_keys = set(checkpoint_payload.keys())
            if model_type == "auto":
                use_lite = any(k in ckpt_keys for k in aspire_lite_keys) and \
                           not any(k in ckpt_keys for k in enhanced_keys)
            else:
                use_lite = model_type == "aspire_lite"

            for key in ("model_dim", "num_heads", "num_inds", "mask_prob", "max_targets",
                        "inter_layers", "shared_bert", "interaction_layers",
                        "intra_layers", "use_intra_set2set", "use_dataset_description", "use_echoices"):
                if key in checkpoint_payload:
                    cfg[key] = checkpoint_payload[key]
        else:
            use_lite = model_type == "aspire_lite"

        if use_lite:
            inner_model = ASPIRELite(
                model_dim=cfg.get("model_dim", 768),
                num_heads=cfg.get("num_heads", 8),
                num_inds=cfg.get("num_inds", 32),
                interaction_layers=cfg.get("interaction_layers", 2),
                inter_layers=cfg.get("inter_layers", 2),
                mask_prob=cfg.get("mask_prob", 0.4),
                max_targets=cfg.get("max_targets", 3),
                shared_bert=cfg.get("shared_bert", "bert-base-uncased"),
            )
        else:
            inner_model = ASPIREEnhanced(**{k: v for k, v in cfg.items()
                                            if k in ("model_dim", "num_heads", "num_inds",
                                                      "mask_prob", "max_targets", "intra_layers",
                                                      "inter_layers", "shared_bert",
                                                      "use_intra_set2set", "use_dataset_description",
                                                      "use_echoices")})

        wrapper = cls(
            cfg,
            feature_specs=feature_specs,
            dataset_context=dataset_context,
            target_indices=target_indices,
            target_column=target_column,
        )
        wrapper.model = inner_model.to(device)

        state_dict = checkpoint_payload
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        incompatible = wrapper.model.load_state_dict(state_dict, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing:
            logger.warning("Missing keys when loading checkpoint: %d", len(missing))
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %d", len(unexpected))
        wrapper._has_loaded_weights = True
        wrapper.eval()
        return wrapper

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

        if isinstance(data, pd.DataFrame) and self.fit_mode_ == "v2":
            return self._predict_v2(data)

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

        if finetune_mode == "v2":
            if not self._has_loaded_weights:
                raise ValueError("finetune_mode='v2' requires a pre-loaded checkpoint.")
            self.fit_mode_ = "v2"
            target_name_set = {self.feature_desc_[idx]["name"] for idx in self.target_indices_ if 0 <= idx < len(self.feature_desc_)}
            self.feature_names_in_ = [f["name"] for f in self.feature_desc_ if f["name"] not in target_name_set]
            self.dataset_description_ = normalized_metadata["dataset_description"]
            self.task_description_ = normalized_metadata["task_description"]
            self.target_description_ = normalized_metadata["target_description"]
            self.is_fitted_ = True
            return self._fit_v2(
                X=X, y=y,
                num_epochs=num_epochs, batch_size=batch_size,
                num_support=num_support, random_state=random_state,
                show_progress=show_progress, test_fraction=test_fraction,
            )

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

    def _fit_v2(
        self,
        X: pd.DataFrame,
        y: Sequence[Any],
        num_epochs: int = 100,
        batch_size: int = 32,
        num_support: int = 15,
        random_state: Optional[int] = None,
        show_progress: bool = True,
        test_fraction: float = 0.15,
    ) -> "AspireModel":
        """Full-backbone-freeze finetuning: cls_head + FeatureInjector + class_bias + LinearHead."""
        device = next(self.model.parameters()).device
        seed = random_state if random_state is not None else 42

        target_idx = self.target_indices_[0]
        target_spec = self.feature_specs_[target_idx]
        classes: List[str] = list(target_spec.get("choices") or sorted(set(str(v) for v in y)))
        n_classes = len(classes)
        categories = classes + ["[UNK]"]
        class_names = classes
        class_to_idx = {c: i for i, c in enumerate(categories)}
        labels: List[int] = [class_to_idx.get(str(v), class_to_idx["[UNK]"]) for v in y]

        n_total = len(labels)
        n_val = max(1, int(n_total * test_fraction))
        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        val_idx_list = indices[:n_val]
        train_idx_list = indices[n_val:]

        train_labels = [labels[i] for i in train_idx_list]
        val_labels   = [labels[i] for i in val_idx_list]

        X_arr = X if isinstance(X, np.ndarray) else X.values

        low_card_cols = [
            f["name"] for f in self.feature_specs_
            if f["name"] != self.feature_specs_[target_idx]["name"]
            and (f["dtype"] == "continuous"
                 or (f["dtype"] == "categorical" and len(f.get("choices") or []) <= 100))
        ]
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        low_card_df = X_df[[c for c in low_card_cols if c in X_df.columns]]
        raw_df = pd.get_dummies(low_card_df).astype(float)
        scaler = StandardScaler()
        raw_np = np.nan_to_num(scaler.fit_transform(raw_df.values.astype(np.float32)))
        raw_dim = raw_np.shape[1]
        self._tv2_scaler = scaler
        self._tv2_raw_cols = list(raw_df.columns)

        from .data_loader import build_training_examples_from_feature_specs
        examples = build_training_examples_from_feature_specs(
            X=X, y=y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_indices=self.target_indices_ if self.target_indices_ else None,
        )
        feats = examples[0].features

        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.cls_head.parameters():
            p.requires_grad_(True)

        model_dim = self.model.model_dim
        class_bias = nn.Parameter(torch.zeros(n_classes, model_dim, device=device))

        _warmup_bert(self.model, feats, self.dataset_context, target_idx, class_names, device)
        phi       = _compute_phi(self.model, feats, device)
        cat_cache = _precompute_cat_cache(self.model, feats, device)

        train_exs = [examples[i] for i in train_idx_list]
        val_exs   = [examples[i] for i in val_idx_list]

        train_q = _vectorized_atoms(self.model, train_exs, feats, phi, cat_cache, target_mask=True)
        val_q   = _vectorized_atoms(self.model, val_exs,   feats, phi, cat_cache, target_mask=True)
        train_s = _vectorized_atoms(self.model, train_exs, feats, phi, cat_cache, target_mask=False)

        ctx_data_tokens = self.model.shared_text.encode_text_sequence(
            self.dataset_context, is_context=True, device=device)
        target_desc = feats[target_idx].description or feats[target_idx].name
        ctx_tgt_tokens = self.model.shared_text.encode_text_sequence(
            target_desc, is_context=False, device=device)

        train_sup, val_sup = _balanced_support_indices(
            train_labels, len(train_exs), len(val_exs), num_support, seed, n_classes)
        val_sup_full = [sample for sample in val_sup]

        train_h = _cache_h_vectors(
            self.model, train_q, train_s, train_exs, train_sup,
            ctx_data_tokens, ctx_tgt_tokens, self.dataset_context, device)
        val_h = _cache_h_vectors(
            self.model, val_q, train_s, val_exs, val_sup_full,
            ctx_data_tokens, ctx_tgt_tokens, self.dataset_context, device)

        raw_tr_t = torch.tensor(raw_np[train_idx_list], dtype=torch.float32, device=device)
        raw_val_t = torch.tensor(raw_np[val_idx_list],  dtype=torch.float32, device=device)

        label_counts = np.bincount(train_labels, minlength=len(categories)).astype(np.float32)
        total = sum(label_counts[:n_classes])
        cw = [total / (n_classes * label_counts[c]) if label_counts[c] > 0 else 0.0 for c in range(n_classes)]
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device)

        feat_inj = FeatureInjector(raw_dim, model_dim).to(device)
        lin_head = LinearHead(model_dim, n_classes).to(device)

        opt = torch.optim.AdamW([
            {"params": self.model.cls_head.parameters(), "lr": 1e-3},
            {"params": feat_inj.parameters(),            "lr": 5e-4},
            {"params": [class_bias],                     "lr": 1e-3},
        ], weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

        opt_lin = torch.optim.AdamW([
            {"params": lin_head.parameters(),  "lr": 1e-3},
            {"params": feat_inj.parameters(),  "lr": 5e-4},
        ], weight_decay=1e-4)
        sch_lin = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lin, T_max=num_epochs)

        best_f1, best_state = -1.0, None
        best_f1_lin, best_state_lin = -1.0, None

        from tqdm.auto import tqdm
        progress = tqdm(total=num_epochs, desc="v2", unit="epoch") if show_progress else None

        for epoch in range(num_epochs):
            self.model.cls_head.train()
            feat_inj.train()
            lin_head.train()

            perm = torch.randperm(train_h.shape[0])
            ep_loss = ep_loss_lin = 0.0
            n_batches = 0

            for start in range(0, len(train_labels), batch_size):
                idx = [perm[j].item() for j in range(start, min(start + batch_size, len(train_labels)))
                       if train_labels[perm[j].item()] < n_classes]
                if not idx:
                    continue
                targets_t = torch.tensor([train_labels[i] for i in idx], device=device, dtype=torch.long)

                with torch.no_grad():
                    lv = torch.stack([
                        self.model.shared_text.encode_text(str(c), is_context=False, device=device)
                        for c in class_names
                    ])
                    label_embs = self.model.cls_head.category_proj(lv) + class_bias

                h = train_h[idx].to(device) + feat_inj(raw_tr_t[idx])
                logits = self.model.cls_head.logits(h, label_embs, context_tokens=ctx_tgt_tokens)
                loss = F.cross_entropy(logits, targets_t, weight=class_weights)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.cls_head.parameters()) + list(feat_inj.parameters()) + [class_bias], 1.0)
                    opt.step()
                    ep_loss += loss.item()

                h_lin = train_h[idx].to(device) + feat_inj(raw_tr_t[idx]).detach()
                loss_lin = F.cross_entropy(lin_head(h_lin.detach()), targets_t, weight=class_weights)
                if not (torch.isnan(loss_lin) or torch.isinf(loss_lin)):
                    opt_lin.zero_grad(); loss_lin.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(lin_head.parameters()) + list(feat_inj.parameters()), 1.0)
                    opt_lin.step()
                    ep_loss_lin += loss_lin.item()

                n_batches += 1

            sch.step(); sch_lin.step()

            self.model.cls_head.eval(); feat_inj.eval(); lin_head.eval()
            with torch.no_grad():
                lv = torch.stack([
                    self.model.shared_text.encode_text(str(c), is_context=False, device=device)
                    for c in class_names
                ])
                label_embs_val = self.model.cls_head.category_proj(lv) + class_bias
                h_val = val_h.to(device) + feat_inj(raw_val_t)
                val_logits = self.model.cls_head.logits(h_val, label_embs_val, context_tokens=ctx_tgt_tokens)
                val_preds  = val_logits.argmax(dim=-1).cpu().tolist()
                pred_labels = [categories[min(p, n_classes - 1)] for p in val_preds]
                true_labels = [categories[min(t, n_classes - 1)] for t in val_labels]
                val_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

                val_lin_preds = lin_head(val_h.to(device) + feat_inj(raw_val_t)).argmax(dim=-1).cpu().tolist()
                pred_lin = [categories[min(p, n_classes - 1)] for p in val_lin_preds]
                val_f1_lin = f1_score(true_labels, pred_lin, average="macro", zero_division=0)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {
                    "cls_head":   deepcopy(self.model.cls_head.state_dict()),
                    "feat_inj":   deepcopy(feat_inj.state_dict()),
                    "class_bias": class_bias.detach().clone(),
                }
            if val_f1_lin > best_f1_lin:
                best_f1_lin = val_f1_lin
                best_state_lin = {
                    "lin_head":  deepcopy(lin_head.state_dict()),
                    "feat_inj":  deepcopy(feat_inj.state_dict()),
                }

            if progress is not None:
                progress.set_postfix({
                    "loss": f"{ep_loss/max(1,n_batches):.4f}",
                    "val_f1": f"{val_f1:.4f}",
                })
                progress.update(1)

        if progress is not None:
            progress.close()

        if best_state:
            self.model.cls_head.load_state_dict(best_state["cls_head"])
            feat_inj.load_state_dict(best_state["feat_inj"])
            with torch.no_grad():
                class_bias.copy_(best_state["class_bias"])
        if best_state_lin:
            lin_head.load_state_dict(best_state_lin["lin_head"])

        self._tv2_feat_inj    = feat_inj
        self._tv2_lin_head    = lin_head
        self._tv2_class_bias  = class_bias
        self._tv2_classes     = classes
        self._tv2_categories  = categories
        self._tv2_feats       = feats
        self._tv2_target_idx  = target_idx
        self._tv2_train_exs   = train_exs
        self._tv2_train_s     = train_s
        self._tv2_train_labels = train_labels
        self._tv2_num_support  = num_support
        self._tv2_n_classes    = n_classes
        self._tv2_seed         = seed
        self._tv2_ctx_data     = ctx_data_tokens
        self._tv2_ctx_tgt      = ctx_tgt_tokens

        self.model.eval()
        return self

    def _predict_v2(self, X: pd.DataFrame) -> List[Any]:
        device = next(self.model.parameters()).device

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        raw_df = pd.get_dummies(X_df).astype(float)
        raw_df = raw_df.reindex(columns=self._tv2_raw_cols, fill_value=0.0)
        raw_np = np.nan_to_num(self._tv2_scaler.transform(raw_df.values.astype(np.float32)))
        raw_t  = torch.tensor(raw_np, dtype=torch.float32, device=device)

        feats       = self._tv2_feats
        target_idx  = self._tv2_target_idx
        categories  = self._tv2_categories
        class_names = self._tv2_classes
        n_classes   = self._tv2_n_classes
        train_labels = self._tv2_train_labels
        num_support  = self._tv2_num_support
        seed         = self._tv2_seed

        phi       = _compute_phi(self.model, feats, device)
        cat_cache = _precompute_cat_cache(self.model, feats, device)

        from .data_loader import build_training_examples_from_feature_specs
        dummy_y = [categories[0]] * len(X_df)
        test_exs = build_training_examples_from_feature_specs(
            X=X_df, y=dummy_y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_indices=self.target_indices_ if self.target_indices_ else None,
        )

        test_q = _vectorized_atoms(self.model, test_exs, feats, phi, cat_cache, target_mask=True)
        _, test_sup = _balanced_support_indices(
            train_labels, len(self._tv2_train_exs), len(test_exs), num_support, seed + 99999, n_classes)

        test_h = _cache_h_vectors(
            self.model, test_q, self._tv2_train_s, test_exs, test_sup,
            self._tv2_ctx_data, self._tv2_ctx_tgt, self.dataset_context, device)

        self.model.cls_head.eval()
        self._tv2_feat_inj.eval()
        with torch.no_grad():
            lv = torch.stack([
                self.model.shared_text.encode_text(str(c), is_context=False, device=device)
                for c in class_names
            ])
            label_embs = self.model.cls_head.category_proj(lv) + self._tv2_class_bias
            h = test_h.to(device) + self._tv2_feat_inj(raw_t)
            logits = self.model.cls_head.logits(h, label_embs, context_tokens=self._tv2_ctx_tgt)
            preds_idx = logits.argmax(dim=-1).cpu().tolist()

        return [categories[min(p, n_classes - 1)] for p in preds_idx]

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
