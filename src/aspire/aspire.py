import math, os, json, logging, random, torch
import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from huggingface_hub import hf_hub_download
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from .model import ASPIREEnhanced, Example, train_examples, _attach_support_examples
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
        self._support_examples: List[Example] = []

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
            # Check if local dir exists first
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

        # handle checkpoints saved with or without "model_state_dict" key
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

            if self._support_examples:
                from .model import _attach_support_examples
                _attach_support_examples(examples, self._support_examples, num_support=min(5, len(self._support_examples)))

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
        finetune_mode: Optional[str] = None,
        max_train_samples: int = 0,
        test_fraction: float = 0.0,
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

        # Hold out test examples before training
        test_examples_ = []
        if test_fraction > 0 and len(examples) >= 5:
            rng = random.Random(random_state)
            shuffled = list(examples)
            rng.shuffle(shuffled)
            n_test = max(1, int(len(shuffled) * test_fraction))
            test_examples_ = shuffled[:n_test]
            examples = shuffled[n_test:]

        if finetune_mode is not None:
            self.fit_mode_ = finetune_mode
        else:
            self.fit_mode_ = "finetune" if self._has_loaded_weights else "scratch"

        if self.fit_mode_ in ("head_only", "finetune"):
            self.model._use_unk = False
            with torch.no_grad():
                self.model.cls_head.log_temperature.fill_(math.log(2.0))
        else:
            self.model._use_unk = True

        if self.fit_mode_ == "head_only":
            resolved_lr_head = learning_rate
            resolved_lr_backbone = 0.0
            freeze_bert = True
            freeze_backbone = True
            num_support = 5
            val_fraction = 0.0
            patience = 10
        elif self.fit_mode_ == "finetune":
            # Keep user override behavior simple: if learning_rate stays at the
            # wrapper default, switch to safer finetuning rates.
            if learning_rate == 1e-3:
                resolved_lr_head = 1e-4
                resolved_lr_backbone = 1e-5
            else:
                resolved_lr_head = learning_rate
                resolved_lr_backbone = learning_rate
            freeze_bert = True
            freeze_backbone = False
            num_support = 5
            val_fraction = 0.2
            patience = 5
        else:
            resolved_lr_head = learning_rate
            resolved_lr_backbone = learning_rate
            freeze_bert = False
            freeze_backbone = False
            num_support = 0
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
            freeze_backbone=freeze_backbone,
            num_support=num_support,
            val_fraction=val_fraction,
            patience=patience,
            show_progress=show_progress,
            max_train_samples=max_train_samples,
        )

        self._support_examples = examples

        # Evaluate on held-out test set if requested
        self.fit_metrics_: Dict[str, Any] = {}
        if test_examples_:
            if num_support > 0:
                _attach_support_examples(test_examples_, examples, num_support=num_support)
            self.fit_metrics_ = self._evaluate_examples(test_examples_)

        target_name_set = {self.feature_desc_[idx]["name"] for idx in self.target_indices_ if 0 <= idx < len(self.feature_desc_)}
        self.feature_names_in_ = [f["name"] for f in self.feature_desc_ if f["name"] not in target_name_set]
        self.dataset_description_ = normalized_metadata["dataset_description"]
        self.task_description_ = normalized_metadata["task_description"]
        self.target_description_ = normalized_metadata["target_description"]
        self.is_fitted_ = True
        return self

    def _evaluate_examples(self, test_examples: List[Example]) -> Dict[str, Any]:
        """Evaluate model on a list of examples and return metrics."""
        self.model.eval()
        cls_preds, cls_targets = [], []
        reg_preds, reg_targets = [], []

        with torch.no_grad():
            for ex in test_examples:
                try:
                    preds = self.model.predict(ex)
                    for j, t_idx in enumerate(ex.target_indices):
                        if j >= len(preds) or preds[j] is None:
                            continue
                        feat = ex.features[t_idx]
                        true_val = ex.values[t_idx]
                        pred_val = preds[j]
                        if feat.dtype == "continuous":
                            try:
                                pv = float(pred_val)
                                tv = float(true_val)
                                if np.isfinite(pv) and np.isfinite(tv):
                                    reg_preds.append(pv)
                                    reg_targets.append(tv)
                            except Exception:
                                pass
                        else:
                            cls_preds.append(str(pred_val))
                            cls_targets.append(str(true_val))
                except Exception:
                    continue

        metrics: Dict[str, Any] = {}
        if cls_preds:
            metrics["accuracy"] = accuracy_score(cls_targets, cls_preds)
            metrics["f1_weighted"] = f1_score(cls_targets, cls_preds, average="weighted", zero_division=0)
            metrics["f1_macro"] = f1_score(cls_targets, cls_preds, average="macro", zero_division=0)
            metrics["n_cls"] = len(cls_preds)
        if reg_preds:
            mse = mean_squared_error(reg_targets, reg_preds)
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["r2"] = r2_score(reg_targets, reg_preds) if len(reg_preds) > 1 else 0.0
            metrics["n_reg"] = len(reg_preds)
        return metrics

    def score(
        self,
        X: Any,
        y: Sequence[Any],
        num_support: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate model on new data using ICL with support examples from fit().
        Returns dict of metrics (accuracy, f1, rmse, r2 as applicable).
        """
        if not self.is_fitted_:
            raise NotFittedError("Call fit() before score().")
        if not self.feature_specs_:
            raise ValueError("feature_specs is required.")

        test_examples = build_training_examples_from_feature_specs(
            X=X,
            y=y,
            feature_specs=self.feature_specs_,
            dataset_context=self.dataset_context,
            target_indices=self.target_indices_ if self.target_indices_ else None,
        )
        if not test_examples:
            raise ValueError("No valid test examples could be built from X/y.")

        if self._support_examples and num_support > 0:
            _attach_support_examples(
                test_examples, self._support_examples,
                num_support=min(num_support, len(self._support_examples)),
            )

        return self._evaluate_examples(test_examples)
