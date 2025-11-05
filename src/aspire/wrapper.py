import os, json, torch
from typing import List, Optional
from huggingface_hub import hf_hub_download
from .model import ASPIREEnhanced, Feature, Example
from .data_loader import df_to_examples

class AspireModel(torch.nn.Module):
    """User-facing wrapper for model."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = ASPIREEnhanced(**config)

    @classmethod
    def from_pretrained(cls, repo_id: str, filename: str = "best_model.pt", config_name: str = "config.json", device: str = "cpu"):
        """Load model + config from a local directory (if it exists) or the Hugging Face Hub."""
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

        model = cls(cfg).to(device)
        state_dict = torch.load(weights_path, map_location=device)

        # handle checkpoints saved with or without "model_state_dict" key
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def save_pretrained(self, save_dir: str = "./checkpoints"):
        """Save model + config locally (for later upload to the Hub)."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    @torch.no_grad()
    def predict(self, example: Example):
        """Run prediction on one Example."""
        self.model.eval()
        return self.model.predict(example)
    
    def predict_df(self, df, target: List[str] = [], batch_size: int = 32):
        """Run prediction on a pandas DataFrame."""
        examples = df_to_examples(df, target)
        predictions = []
        step = max(1, batch_size)
        for idx in range(0, len(examples), step):
            batch = examples[idx:idx + step]
            for example in batch:
                predictions.append(self.predict(example))
        return predictions
