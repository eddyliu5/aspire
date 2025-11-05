import os, json, torch
from huggingface_hub import hf_hub_download
from .model import ASPIREEnhanced, Feature, Example  # your main architecture

class AspireModel(torch.nn.Module):
    """Minimal user-facing wrapper for ASPIRE."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = ASPIREEnhanced(**config)

    @torch.no_grad()
    def predict(self, example: Example):
        """Run inference on one Example."""
        self.model.eval()
        return self.model.predict(example)

    @classmethod
    def from_pretrained(cls, repo_id: str, filename: str = "best_model.pt", config_name: str = "config.json", device: str = "cpu"):
        """Load model + config from the Hugging Face Hub or local path."""
        cfg_path = hf_hub_download(repo_id=repo_id, filename=config_name)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        model = cls(cfg).to(device)
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
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
