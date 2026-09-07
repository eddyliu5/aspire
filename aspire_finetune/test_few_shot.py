import numpy as np
import pandas as pd
import torch

import aspire_finetune.aspire as aspire_module
from aspire_finetune import ASPIRE


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.last_support_shape = None

    def forward(
        self,
        x_txt,
        x_num,
        d_output,
        target_type,
        support_x_txt=None,
        support_x_num=None,
        desc_txt=None,
        return_encoded=False,
    ):
        self.last_support_shape = (
            None if support_x_txt is None else support_x_txt.shape
        )
        batch_size = len(x_txt)
        if target_type == "cat":
            predictions = torch.arange(
                d_output, dtype=torch.float32, device=self.anchor.device
            ).repeat(batch_size, 1)
        else:
            predictions = torch.zeros((batch_size, 30), device=self.anchor.device)
        if return_encoded:
            encoded = torch.ones(
                (batch_size, d_output, 8), device=self.anchor.device
            )
            return predictions, encoded
        return predictions


def test_few_shot_uses_balanced_support_and_all_classes(monkeypatch):
    fake = FakeBackbone()
    monkeypatch.setattr(
        aspire_module, "_load_checkpoint", lambda *args, **kwargs: (fake, "mog")
    )
    feature_specs = [
        {"name": "value", "description": "value", "dtype": "continuous"},
        {
            "name": "label",
            "description": "label",
            "dtype": "categorical",
            "choices": ["a", "b", "c"],
        },
    ]
    X_support = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6]})
    y_support = ["a", "a", "b", "b", "c", "c"]
    model = ASPIRE(
        checkpoint="unused.pt",
        device="cpu",
        feature_specs=feature_specs,
        target_column="label",
    )

    model.fit_few_shot(
        X_support, y_support, shots_per_class=1, task_type="classification"
    )
    probabilities = model.predict_proba(pd.DataFrame({"value": [2.5]}))

    assert model.fit_mode_ == "few_shot"
    assert model.support_size_ == 3
    assert probabilities.shape == (1, 3)
    assert fake.last_support_shape is None
    assert isinstance(model._few_shot_probe, aspire_module.LogisticRegression)
    assert model.predict(pd.DataFrame({"value": [2.5]}))[0] in model._classes
    assert all(not parameter.requires_grad for parameter in fake.parameters())


def test_few_shot_rejects_conflicting_support_limits(monkeypatch):
    model = ASPIRE(checkpoint="unused.pt", device="cpu")
    with np.testing.assert_raises(ValueError):
        model.fit_few_shot([[1], [2]], ["a", "b"], shots_per_class=1, max_support=2)
