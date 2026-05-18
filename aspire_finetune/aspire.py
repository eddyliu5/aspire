import logging
import random
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler

from .data_loader import build_bundle_from_feature_specs, infer_feature_specs
from .model import (
    Backbone, DatasetBundle, D_MODEL, LinearHead,
    set_seeds, prepare_batch, compute_loss, _decode_mog, _decode_bins,
)

logger = logging.getLogger(__name__)


class _RidgeProbe:
    def __init__(self):
        self._ridge = Ridge(alpha=1.0)
        self._y_min = 0.0
        self._y_range = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeProbe":
        self._y_min = float(np.min(y))
        self._y_range = float(np.max(y) - np.min(y)) or 1.0
        y_norm = (y - self._y_min) / self._y_range
        self._ridge.fit(X, y_norm)
        in_sample = self._ridge.predict(X)
        nrmse = float(np.sqrt(np.mean((in_sample - y_norm) ** 2)))
        logger.info("Ridge probe  in-sample NRMSE=%.4f  (optimistic lower bound)", nrmse)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._ridge.predict(X) * self._y_range + self._y_min


def _load_checkpoint(
    ckpt_path: str,
    device: str,
    interaction_layers: int = 6,
    n_mog: int = 10,
) -> Tuple[Backbone, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    reg_head = "mog" if any(k.startswith("head.pi.") for k in state) else "discretized"
    model = Backbone(
        d_model=D_MODEL, num_interaction_layers=interaction_layers,
        unfreeze_layers=0, reg_head=reg_head, n_mog=n_mog,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_desc_missing = [k for k in missing if "desc_proj" not in k]
    if non_desc_missing:
        logger.warning("Unexpected missing keys: %s", non_desc_missing[:5])
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:5])
    logger.info("Loaded %s  epoch=%s  val_loss=%.4f  reg_head=%s",
                ckpt_path, ckpt.get("epoch", "?"),
                ckpt.get("val_loss", float("nan")), reg_head)
    return model.to(device), reg_head


def _freeze_backbone(model: Backbone, unfreeze_top_layers: int = 0):
    for p in model.parameters():
        p.requires_grad = False
    for mod in [model.head, model.desc_proj]:
        for p in mod.parameters():
            p.requires_grad = True
    for emb in [model.support_type_emb, model.query_type_emb, model.label_type_emb]:
        emb.requires_grad = True
    if unfreeze_top_layers > 0:
        layers = model.tabular_encoder.encoder.layers
        for layer in layers[len(layers) - unfreeze_top_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %d / %d (%.2f%%)", n_train, n_total, 100 * n_train / n_total)


@torch.no_grad()
def _cache_h_vectors(
    model: Backbone,
    bundle: DatasetBundle,
    rows: List,
    target_col: str,
    support_rows: Optional[List],
    include_desc: bool,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    amp_dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embs, all_labels = [], []
    for i in range(0, len(rows), batch_size):
        br = rows[i: i + batch_size]
        batch = prepare_batch(bundle, br, target_col,
                              support_rows=support_rows, mask_prob=0.0,
                              include_desc=include_desc)
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            _, enc = model(
                batch["x_txt"], batch["x_num"], batch["d_output"],
                target_type=batch["target_type"],
                support_x_txt=batch.get("support_x_txt"),
                support_x_num=batch.get("support_x_num"),
                desc_txt=batch.get("desc_txt"),
                return_encoded=True,
            )
        d_out = batch["d_output"]
        emb = enc[:, :d_out, :].float().mean(dim=1)
        all_embs.append(emb.cpu().numpy())
        if batch["target_type"] == "num":
            all_labels.extend(batch["y_cont"].tolist())
        else:
            all_labels.extend(batch["y"].tolist())
    return np.concatenate(all_embs, axis=0), np.array(all_labels)


class ASPIRE:

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        feature_specs: Optional[Sequence[Mapping[str, Any]]] = None,
        dataset_context: str = "",
        target_column: str = "__target__",
        interaction_layers: int = 6,
        n_mog: int = 10,
        unfreeze_top_layers: int = 0,
    ):
        self.checkpoint = checkpoint
        self.device = device if torch.cuda.is_available() else "cpu"
        self.feature_specs_ = list(feature_specs) if feature_specs else []
        self.dataset_context = dataset_context
        self.target_column_ = target_column
        self.interaction_layers = interaction_layers
        self.n_mog = n_mog
        self.unfreeze_top_layers = unfreeze_top_layers

        self.is_fitted_ = False
        self.fit_mode_: Optional[str] = None
        self._classes: List[str] = []
        self._bundle: Optional[DatasetBundle] = None
        self._model: Optional[Backbone] = None
        self._reg_head: str = "mog"
        self._train_rows: List = []
        self._support_rows: List = []
        self._v2_head: Optional[LinearHead] = None
        self._xgb_clf = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        device: str = "cuda",
        feature_specs: Optional[Sequence[Mapping[str, Any]]] = None,
        dataset_context: str = "",
        target_column: str = "__target__",
        interaction_layers: int = 6,
        n_mog: int = 10,
        unfreeze_top_layers: int = 0,
    ) -> "ASPIRE":
        return cls(
            checkpoint=checkpoint,
            device=device,
            feature_specs=feature_specs,
            dataset_context=dataset_context,
            target_column=target_column,
            interaction_layers=interaction_layers,
            n_mog=n_mog,
            unfreeze_top_layers=unfreeze_top_layers,
        )

    def fit(
        self,
        X: Any,
        y: Sequence[Any],
        finetune_mode: str = "v2",
        num_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_support: int = 0,
        test_fraction: float = 0.15,
        val_fraction: Optional[float] = None,
        random_state: int = 42,
        include_desc: bool = True,
        show_progress: bool = True,
    ) -> "ASPIRE":
        set_seeds(random_state)

        # accept old mode names as aliases
        _mode_aliases = {"linear_probe": "v2", "head_finetune": "head_v2"}
        finetune_mode = _mode_aliases.get(finetune_mode, finetune_mode)

        # val_fraction is an alias for test_fraction
        if val_fraction is not None:
            test_fraction = val_fraction

        if not self.feature_specs_:
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            tmp = pd.concat([X_df.reset_index(drop=True),
                             pd.Series(y, name=self.target_column_).reset_index(drop=True)], axis=1)
            self.feature_specs_ = infer_feature_specs(tmp, self.target_column_)

        bundle = build_bundle_from_feature_specs(
            X=X, y=y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_col=self.target_column_,
        )
        self._bundle = bundle

        target_type = bundle.col_types.get(self.target_column_, "cat")
        if target_type == "cat":
            self._classes = sorted(bundle.df[self.target_column_].dropna().astype(str).unique().tolist())
        else:
            self._classes = []

        df = bundle.df.dropna(subset=[self.target_column_]).reset_index(drop=True)
        all_idx = list(range(len(df)))
        random.shuffle(all_idx)
        n_val = max(1, int(len(all_idx) * test_fraction))
        val_idx = all_idx[:n_val]
        train_idx = all_idx[n_val:]

        all_rows = [df.iloc[i] for i in train_idx]
        if num_support > 0 and len(all_rows) > num_support:
            self._support_rows = all_rows[:num_support]
            self._train_rows = all_rows[num_support:]
        else:
            self._support_rows = []
            self._train_rows = all_rows

        val_rows = [df.iloc[i] for i in val_idx]

        model, reg_head = _load_checkpoint(
            self.checkpoint, self.device,
            interaction_layers=self.interaction_layers, n_mog=self.n_mog,
        )
        self._reg_head = reg_head

        device_obj = torch.device(self.device)
        use_amp = device_obj.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        use_scaler = use_amp and amp_dtype == torch.float16

        if finetune_mode == "v2":
            self._model, self._v2_head = self._fit_v2(
                model=model, bundle=bundle, device_obj=device_obj,
                use_amp=use_amp, amp_dtype=amp_dtype,
                num_epochs=num_epochs, lr=learning_rate,
                batch_size=batch_size, include_desc=include_desc,
                val_rows=val_rows, show_progress=show_progress,
            )
        elif finetune_mode == "v2_xgb":
            self._model, self._xgb_clf = self._fit_v2_xgb(
                model=model, bundle=bundle, device_obj=device_obj,
                use_amp=use_amp, amp_dtype=amp_dtype,
                batch_size=batch_size, include_desc=include_desc,
            )
            self._v2_head = None
        elif finetune_mode == "v2_ft":
            self._model, self._v2_head = self._fit_v2_ft(
                model=model, bundle=bundle, device_obj=device_obj,
                use_amp=use_amp, amp_dtype=amp_dtype, use_scaler=use_scaler,
                num_epochs=num_epochs, lr=learning_rate,
                batch_size=batch_size, include_desc=include_desc,
                val_rows=val_rows, show_progress=show_progress,
                unfreeze_top_layers=self.unfreeze_top_layers,
            )
        elif finetune_mode == "head_v2":
            _freeze_backbone(model, unfreeze_top_layers=self.unfreeze_top_layers)
            self._model = self._fit_head_v2(
                model=model, bundle=bundle, device_obj=device_obj,
                use_amp=use_amp, amp_dtype=amp_dtype, use_scaler=use_scaler,
                num_epochs=num_epochs, lr=learning_rate,
                batch_size=batch_size, include_desc=include_desc,
                val_rows=val_rows, show_progress=show_progress,
            )
            self._v2_head = None
        else:
            raise ValueError(f"Unknown finetune_mode: '{finetune_mode}'. Use 'v2', 'v2_ft', 'v2_xgb', or 'head_v2'.")

        self.fit_mode_ = finetune_mode
        self.is_fitted_ = True
        return self

    def _fit_v2(
        self, model, bundle, device_obj, use_amp, amp_dtype,
        num_epochs, lr, batch_size, include_desc, val_rows, show_progress,
    ) -> Tuple[Backbone, LinearHead]:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        target_type = bundle.col_types.get(self.target_column_, "cat")
        is_reg = target_type == "num"

        logger.info("Extracting embeddings: %d train rows", len(self._train_rows))

        train_embs, train_labels = _cache_h_vectors(
            model, bundle, self._train_rows, self.target_column_,
            self._support_rows or None, include_desc,
            device_obj, batch_size, use_amp, amp_dtype,
        )
        val_embs, val_labels = _cache_h_vectors(
            model, bundle, val_rows, self.target_column_,
            self._support_rows or None, include_desc,
            device_obj, batch_size, use_amp, amp_dtype,
        )

        if is_reg:
            logger.info("Embedding shape: %s  [regression, Ridge probe]", train_embs.shape)
            logger.info("  label range=[%.4f, %.4f]  emb norm mean=%.4f",
                        float(np.min(train_labels)), float(np.max(train_labels)),
                        float(np.linalg.norm(train_embs, axis=1).mean()))
            all_embs = np.concatenate([train_embs, val_embs], axis=0)
            all_labels = np.concatenate([train_labels, val_labels], axis=0)
            probe = _RidgeProbe()
            probe.fit(all_embs, all_labels)
            return model, probe
        else:
            all_embs = np.concatenate([train_embs, val_embs], axis=0)
            all_labels = np.concatenate([train_labels, val_labels], axis=0)
            n_classes = len(self._classes)
            logger.info("Embedding shape: %s  n_classes=%d  [LogisticRegression probe]",
                        train_embs.shape, n_classes)
            probe = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000,
                solver="lbfgs", multi_class="auto",
            )
            probe.fit(all_embs, all_labels)
            val_preds = probe.predict(val_embs)
            val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            logger.info("LR probe  val F1-macro=%.4f", val_f1)
            return model, probe

    def _fit_head_v2(
        self, model, bundle, device_obj, use_amp, amp_dtype, use_scaler,
        num_epochs, lr, batch_size, include_desc, val_rows, show_progress,
    ) -> Backbone:
        scaler = GradScaler(enabled=use_scaler)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        log_every = max(1, num_epochs // 5)
        best_loss, best_state = float("inf"), None

        for epoch in range(num_epochs):
            model.train()
            random.shuffle(self._train_rows)
            ep_loss, n_b = 0.0, 0

            for i in range(0, len(self._train_rows), batch_size):
                br = self._train_rows[i: i + batch_size]
                if not br:
                    continue
                batch = prepare_batch(bundle, br, self.target_column_,
                                      support_rows=None, mask_prob=0.0,
                                      include_desc=include_desc)
                with autocast(device_type=device_obj.type, enabled=use_amp, dtype=amp_dtype):
                    preds = model(
                        batch["x_txt"], batch["x_num"], batch["d_output"],
                        target_type=batch["target_type"],
                        support_x_txt=batch.get("support_x_txt"),
                        support_x_num=batch.get("support_x_num"),
                        desc_txt=batch.get("desc_txt"),
                    )
                    loss = compute_loss(preds, batch["y"], batch["target_type"],
                                       device_obj, reg_head=self._reg_head,
                                       y_cont=batch.get("y_cont"))
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(); continue
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                ep_loss += loss.item(); n_b += 1
            scheduler.step()

            model.eval()
            v_loss, v_b = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(val_rows), batch_size):
                    br = val_rows[i: i + batch_size]
                    if not br: continue
                    batch = prepare_batch(bundle, br, self.target_column_,
                                         support_rows=self._support_rows or None,
                                         mask_prob=0.0, include_desc=include_desc)
                    with autocast(device_type=device_obj.type, enabled=use_amp, dtype=amp_dtype):
                        preds = model(
                            batch["x_txt"], batch["x_num"], batch["d_output"],
                            target_type=batch["target_type"],
                            support_x_txt=batch.get("support_x_txt"),
                            support_x_num=batch.get("support_x_num"),
                            desc_txt=batch.get("desc_txt"),
                        )
                        loss = compute_loss(preds, batch["y"], batch["target_type"],
                                           device_obj, reg_head=self._reg_head,
                                           y_cont=batch.get("y_cont"))
                    v_loss += loss.item(); v_b += 1

            avg_val = v_loss / max(1, v_b)
            if avg_val < best_loss:
                best_loss = avg_val
                best_state = deepcopy(model.state_dict())

            if show_progress and (epoch + 1) % log_every == 0:
                logger.info("  [head_v2] epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                            epoch + 1, num_epochs, ep_loss / max(1, n_b), avg_val)

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        logger.info("Best head_v2 val loss: %.4f", best_loss)
        return model

    def _fit_v2_ft(
        self, model, bundle, device_obj, use_amp, amp_dtype, use_scaler,
        num_epochs, lr, batch_size, include_desc, val_rows, show_progress,
        unfreeze_top_layers: int = 2,
    ) -> Tuple[Backbone, LinearHead]:
        for p in model.parameters():
            p.requires_grad = False

        layers = model.tabular_encoder.encoder.layers
        n_unfreeze = min(unfreeze_top_layers, len(layers))
        for layer in layers[len(layers) - n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True

        target_type = bundle.col_types.get(self.target_column_, "cat")
        is_reg = target_type == "num"
        n_classes = len(self._classes) if not is_reg else 1
        head = LinearHead(D_MODEL, n_classes).to(device_obj)

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info("v2_ft: unfreezing top %d layers  trainable=%d / %d (%.2f%%)",
                    n_unfreeze, n_train + sum(p.numel() for p in head.parameters()),
                    n_total, 100 * (n_train + sum(p.numel() for p in head.parameters())) / n_total)

        scaler = GradScaler(enabled=use_scaler)
        all_params = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_metric, best_model_state, best_head_state = (-1.0 if not is_reg else float("inf")), None, None
        log_every = max(1, num_epochs // 5)

        for epoch in range(num_epochs):
            model.train(); head.train()
            random.shuffle(self._train_rows)
            ep_loss, n_b = 0.0, 0

            for i in range(0, len(self._train_rows), batch_size):
                br = self._train_rows[i: i + batch_size]
                if not br:
                    continue
                batch = prepare_batch(bundle, br, self.target_column_,
                                      support_rows=self._support_rows or None,
                                      mask_prob=0.0, include_desc=include_desc)
                with autocast(device_type=device_obj.type, enabled=use_amp, dtype=amp_dtype):
                    _, enc = model(
                        batch["x_txt"], batch["x_num"], batch["d_output"],
                        target_type=batch["target_type"],
                        support_x_txt=batch.get("support_x_txt"),
                        support_x_num=batch.get("support_x_num"),
                        desc_txt=batch.get("desc_txt"),
                        return_encoded=True,
                    )
                    d_out = batch["d_output"]
                    emb = enc[:, :d_out, :].float().mean(dim=1)
                    out = head(emb)
                    if is_reg:
                        loss = F.mse_loss(out.squeeze(1),
                                          torch.tensor(batch["y_cont"], dtype=torch.float32, device=device_obj))
                    else:
                        loss = F.cross_entropy(out,
                                               torch.tensor(batch["y"], dtype=torch.long, device=device_obj))

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(); continue
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                ep_loss += loss.item(); n_b += 1
            scheduler.step()

            # Validation
            model.eval(); head.eval()
            val_preds_all, val_labels_all = [], []
            with torch.no_grad():
                for i in range(0, len(val_rows), batch_size):
                    br = val_rows[i: i + batch_size]
                    if not br: continue
                    batch = prepare_batch(bundle, br, self.target_column_,
                                          support_rows=self._support_rows or None,
                                          mask_prob=0.0, include_desc=include_desc)
                    with autocast(device_type=device_obj.type, enabled=use_amp, dtype=amp_dtype):
                        _, enc = model(
                            batch["x_txt"], batch["x_num"], batch["d_output"],
                            target_type=batch["target_type"],
                            support_x_txt=batch.get("support_x_txt"),
                            support_x_num=batch.get("support_x_num"),
                            desc_txt=batch.get("desc_txt"),
                            return_encoded=True,
                        )
                    d_out = batch["d_output"]
                    emb = enc[:, :d_out, :].float().mean(dim=1)
                    out = head(emb)
                    if is_reg:
                        val_preds_all.append(out.squeeze(1).cpu())
                        val_labels_all.append(torch.tensor(batch["y_cont"], dtype=torch.float32))
                    else:
                        val_preds_all.append(out.argmax(dim=1).cpu().numpy())
                        val_labels_all.extend(batch["y"].tolist())

            if is_reg:
                vp = torch.cat(val_preds_all); vl = torch.cat(val_labels_all)
                metric = F.mse_loss(vp, vl).item()
                improved = metric < best_metric
            else:
                vp = np.concatenate(val_preds_all)
                metric = f1_score(val_labels_all, vp, average="macro", zero_division=0)
                improved = metric > best_metric

            if improved:
                best_metric = metric
                best_model_state = deepcopy(model.state_dict())
                best_head_state = deepcopy(head.state_dict())

            if show_progress and (epoch + 1) % log_every == 0:
                logger.info("  [v2_ft] epoch %d/%d  loss=%.4f  val_%s=%.4f",
                            epoch + 1, num_epochs, ep_loss / max(1, n_b),
                            "mse" if is_reg else "f1", metric)

        if best_model_state:
            model.load_state_dict(best_model_state)
        if best_head_state:
            head.load_state_dict(best_head_state)
        model.eval(); head.eval()
        logger.info("Best v2_ft val %s: %.4f", "MSE" if is_reg else "F1", best_metric)
        return model, head

    def _fit_v2_xgb(
        self, model, bundle, device_obj, use_amp, amp_dtype,
        batch_size, include_desc,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is required for v2_xgb mode: pip install xgboost")

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        logger.info("Extracting embeddings: %d train rows", len(self._train_rows))
        train_embs, train_labels = _cache_h_vectors(
            model, bundle, self._train_rows, self.target_column_,
            self._support_rows or None, include_desc,
            device_obj, batch_size, use_amp, amp_dtype,
        )
        n_classes = len(self._classes)
        logger.info("Embedding shape: %s  n_classes=%d", train_embs.shape, n_classes)

        xgb = XGBClassifier(
            n_estimators=20, max_depth=3, learning_rate=0.3,
            subsample=1.0, colsample_bytree=1.0,
            objective="multi:softmax" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss", random_state=42,
            **({"num_class": n_classes} if n_classes > 2 else {}),
        )
        xgb.fit(train_embs, train_labels, verbose=False)
        logger.info("XGBoost head fitted on ASPIRE embeddings")
        return model, xgb

    def _predict_v2_xgb(self, X: Any, batch_size: int = 64, include_desc: bool = True) -> np.ndarray:
        device_obj = torch.device(self.device)
        use_amp = device_obj.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df = X_df.reset_index(drop=True)
        dummy_y = [self._classes[i % len(self._classes)] for i in range(len(X_df))] if self._classes else [0] * len(X_df)
        test_bundle = build_bundle_from_feature_specs(
            X=X_df, y=dummy_y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_col=self.target_column_,
        )
        test_rows = [test_bundle.df.iloc[i] for i in range(len(test_bundle.df))]
        embs, _ = _cache_h_vectors(
            self._model, test_bundle, test_rows, self.target_column_,
            self._support_rows or None, include_desc,
            device_obj, batch_size, use_amp, amp_dtype,
        )
        proba = self._xgb_clf.predict_proba(embs)
        return proba

    def get_embeddings(
        self,
        X: Any,
        batch_size: int = 64,
        include_desc: bool = True,
    ) -> np.ndarray:
        """Return backbone [N, D_MODEL] embeddings for X (requires fit() first)."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before get_embeddings()")
        device_obj = torch.device(self.device)
        use_amp = device_obj.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df = X_df.reset_index(drop=True)
        dummy_y = [self._classes[i % len(self._classes)] for i in range(len(X_df))] if self._classes else [0] * len(X_df)
        test_bundle = build_bundle_from_feature_specs(
            X=X_df, y=dummy_y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_col=self.target_column_,
        )
        test_rows = [test_bundle.df.iloc[i] for i in range(len(test_bundle.df))]
        embs, _ = _cache_h_vectors(
            self._model, test_bundle, test_rows, self.target_column_,
            self._support_rows or None, include_desc,
            device_obj, batch_size, use_amp, amp_dtype,
        )
        return embs

    def _predict_v2(self, X: Any, batch_size: int = 64, include_desc: bool = True) -> np.ndarray:
        bundle = self._bundle
        model = self._model
        head = self._v2_head
        device_obj = torch.device(self.device)
        use_amp = device_obj.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df = X_df.reset_index(drop=True)
        dummy_y = [self._classes[i % len(self._classes)] for i in range(len(X_df))] if self._classes else [0] * len(X_df)
        test_bundle = build_bundle_from_feature_specs(
            X=X_df, y=dummy_y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_col=self.target_column_,
        )
        test_rows = [test_bundle.df.iloc[i] for i in range(len(test_bundle.df))]

        embs, _ = _cache_h_vectors(
            model, test_bundle, test_rows, self.target_column_,
            self._support_rows or None, include_desc,
            device_obj, batch_size, use_amp, amp_dtype,
        )
        target_type = self._bundle.col_types.get(self.target_column_, "cat")
        if target_type == "num":
            if isinstance(head, _RidgeProbe):
                preds_norm = head.predict(embs)
            else:
                head.eval()
                X_t = torch.tensor(embs, dtype=torch.float32, device=device_obj)
                with torch.no_grad():
                    preds_norm = head(X_t).squeeze(1).cpu().numpy()
            mu_s, sigma_s = self._bundle.num_scalers.get(self.target_column_, (0.0, 1.0))
            logger.info("v2 reg unscale: mu=%.4f sigma=%.4f  pred_norm range=[%.3f, %.3f]",
                        mu_s, sigma_s, preds_norm.min(), preds_norm.max())
            return (preds_norm * sigma_s + mu_s).reshape(-1, 1)
        if isinstance(head, LogisticRegression):
            return head.predict_proba(embs)
        head.eval()
        X_t = torch.tensor(embs, dtype=torch.float32, device=device_obj)
        with torch.no_grad():
            out = head(X_t)
        return torch.softmax(out, dim=-1).cpu().numpy()

    def _predict_head_v2(self, X: Any, batch_size: int = 64, include_desc: bool = True) -> np.ndarray:
        bundle = self._bundle
        model = self._model
        device_obj = torch.device(self.device)
        use_amp = device_obj.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df = X_df.reset_index(drop=True)
        dummy_y = [self._classes[i % len(self._classes)] for i in range(len(X_df))] if self._classes else ["0"] * len(X_df)
        test_bundle = build_bundle_from_feature_specs(
            X=X_df, y=dummy_y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_col=self.target_column_,
        )
        test_rows = [test_bundle.df.iloc[i] for i in range(len(test_bundle.df))]

        target_type = bundle.col_types.get(self.target_column_, "cat")
        all_preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(test_rows), batch_size):
                br = test_rows[i: i + batch_size]
                batch = prepare_batch(test_bundle, br, self.target_column_,
                                      support_rows=self._support_rows or None,
                                      mask_prob=0.0, include_desc=include_desc)
                with autocast(device_type=device_obj.type, enabled=use_amp, dtype=amp_dtype):
                    preds = model(
                        batch["x_txt"], batch["x_num"], batch["d_output"],
                        target_type=batch["target_type"],
                        support_x_txt=batch.get("support_x_txt"),
                        support_x_num=batch.get("support_x_num"),
                        desc_txt=batch.get("desc_txt"),
                    )
                if target_type == "cat":
                    all_preds.append(torch.softmax(preds.float(), dim=-1).cpu().numpy())
                else:
                    all_preds.append(preds.float().cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    def predict_proba(
        self,
        X: Any,
        batch_size: int = 64,
        include_desc: bool = True,
    ) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict_proba()")
        if self.fit_mode_ in ("v2", "v2_ft"):
            return self._predict_v2(X, batch_size=batch_size, include_desc=include_desc)
        if self.fit_mode_ == "v2_xgb":
            return self._predict_v2_xgb(X, batch_size=batch_size, include_desc=include_desc)
        return self._predict_head_v2(X, batch_size=batch_size, include_desc=include_desc)

    def predict(self, X: Any, batch_size: int = 64, include_desc: bool = True) -> List[Any]:
        proba = self.predict_proba(X, batch_size=batch_size, include_desc=include_desc)
        target_type = self._bundle.col_types.get(self.target_column_, "cat")

        if target_type == "cat":
            pred_idx = proba.argmax(axis=1)
            return [self._classes[i] for i in pred_idx]

        # regression
        if self.fit_mode_ == "v2":
            return proba.squeeze(1).tolist()

        if self._reg_head == "mog":
            preds_norm = _decode_mog(proba)
            mu_s, sigma_s = self._bundle.num_scalers.get(self.target_column_, (0.0, 1.0))
            return (preds_norm * sigma_s + mu_s).tolist()
        edges = self._bundle.reg_bin_edges.get(self.target_column_)
        if edges is not None:
            return _decode_bins(proba.argmax(axis=1), edges).tolist()
        return proba.argmax(axis=1).tolist()

    def score(self, X: Any, y: Sequence[Any], average: str = "weighted") -> float:
        preds = self.predict(X)
        target_type = self._bundle.col_types.get(self.target_column_, "cat")
        if target_type == "cat":
            return f1_score(list(y), preds, average=average, zero_division=0)
        from sklearn.metrics import r2_score
        return r2_score(list(y), preds)
