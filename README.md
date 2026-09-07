# Arbitrary Set-based Permutation-Invariant Reasoning Engine

## Basic finetune/prediction

```python
import pandas as pd
from aspire import AspireModel

# Define metadata
feature_specs = [
    {"name": "hours_sleep", "description": "Hours slept last night.", "dtype": "continuous", "value_range": (0.0, 12.0)},
    {"name": "caffeine_mg", "description": "Caffeine consumed today in milligrams.", "dtype": "continuous", "value_range": (0.0, 600.0)},
    {"name": "stress_level", "description": "Self-reported stress level.", "dtype": "categorical", "choices": ["low", "medium", "high"]},
    {"name": "fatigue_risk", "description": "Predicted fatigue risk class.", "dtype": "categorical", "choices": ["low", "high"]},
]
dataset_context = "Daily fatigue risk prediction dataset."

# Load dataset
X_train = pd.DataFrame(
    {
        "hours_sleep": [8.0, 6.5, 5.0, 7.5],
        "caffeine_mg": [80.0, 160.0, 280.0, 120.0],
        "stress_level": ["low", "medium", "high", "low"],
    }
)
y_train = ["low", "high", "high", "low"]

X_test = pd.DataFrame(
    {
        "hours_sleep": [8.5, 4.0],
        "caffeine_mg": [60.0, 360.0],
        "stress_level": ["low", "high"],
    }
)

model = AspireModel.from_pretrained(
    "checkpoints/best_model",
    device="cpu",
    feature_specs=feature_specs,
    dataset_context=dataset_context,
)
model.fit(X_train, y_train, num_epochs=10, batch_size=4)
preds = model.predict(X_test)
```

## Gradient-free few-shot inference

Use labeled rows as in-context support examples without updating the checkpoint:

```python
from aspire_finetune import ASPIRE

model = ASPIRE.from_pretrained(
    checkpoint="checkpoints/best_model.pt",
    feature_specs=feature_specs,
    dataset_context=dataset_context,
    target_column="fatigue_risk",
)
model.fit_few_shot(
    X_train,
    y_train,
    shots_per_class=5,
    random_state=42,
    task_type="classification",
)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

Use `max_support=20` for a total support budget. The equivalent generic call is
`model.fit(X_train, y_train, finetune_mode="few_shot", num_support=20)`.
