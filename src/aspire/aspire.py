import os, json, logging, torch
from typing import Any, List, Mapping, Optional, Sequence
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
    ) -> "AspireModel":
        """
        Fit/fine-tune model on a tabular dataset.

        If current instance has loaded weights (`from_pretrained`), fit runs as finetuning.
        Otherwise fit trains from scratch from the current randomly initialized weights.
        """
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

        self.fit_mode_ = "finetune" if self._has_loaded_weights else "scratch"

        if self.fit_mode_ == "finetune":
            # Keep user override behavior simple: if learning_rate stays at the
            # wrapper default, switch to safer finetuning rates.
            if learning_rate == 1e-3:
                resolved_lr_head = 1e-4
                resolved_lr_backbone = 1e-5
            else:
                resolved_lr_head = learning_rate
                resolved_lr_backbone = learning_rate
            freeze_bert = True
            num_support = 5
            val_fraction = 0.2
            patience = 5
        else:
            resolved_lr_head = learning_rate
            resolved_lr_backbone = learning_rate
            freeze_bert = False
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
            num_support=num_support,
            val_fraction=val_fraction,
            patience=patience,
        )

        target_name_set = {self.feature_desc_[idx]["name"] for idx in self.target_indices_ if 0 <= idx < len(self.feature_desc_)}
        self.feature_names_in_ = [f["name"] for f in self.feature_desc_ if f["name"] not in target_name_set]
        self.dataset_description_ = normalized_metadata["dataset_description"]
        self.task_description_ = normalized_metadata["task_description"]
        self.target_description_ = normalized_metadata["target_description"]
        self.is_fitted_ = True
        return self
