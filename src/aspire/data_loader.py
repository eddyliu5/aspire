from typing import List
import pandas as pd
from .model import Feature, Example

def add_metadata(df, feature_desc: List[Feature], target: List[str], dataset_desc: str = ""):
    """Adds metadata to df"""
    df.attrs["feature_desc"] = [vars(f) for f in feature_desc]
    df.attrs["target"] = target
    df.attrs["dataset_desc"] = dataset_desc
    return df

def df_to_examples(df, target: List[str] = []):
    """Convert a DataFrame into a list of Example objects"""
    metadata = getattr(df, "attrs", None)
    if metadata:
        features = [Feature(**f) for f in df.attrs["feature_desc"]]
        target_name = df.attrs.get("target")
        description = df.attrs.get("dataset_desc")

        if target:
            # Overwrite metadata target if explicitly given
            target_name = target
    else:
        features = []
        for column in df.columns:
            series = df[column]
            non_na = series.dropna()

            is_bool = pd.api.types.is_bool_dtype(series)
            numeric_series = None

            if pd.api.types.is_numeric_dtype(series) and not is_bool:
                numeric_series = non_na.astype(float, copy=False)
            elif not non_na.empty:
                coerced = pd.to_numeric(non_na, errors="coerce")
                valid = coerced.dropna()
                if not valid.empty and len(valid) >= 0.9 * len(non_na):
                    numeric_series = valid

            if numeric_series is not None and not numeric_series.empty:
                dtype = "continuous"
                value_range = (float(numeric_series.min()), float(numeric_series.max()))
                feature = Feature(
                    name=column,
                    description="",
                    dtype=dtype,
                    value_range=value_range
                )
            else:
                dtype = "categorical"
                choices = list(dict.fromkeys(str(v) for v in non_na)) if not non_na.empty else []
                feature = Feature(
                    name=column,
                    description="",
                    dtype=dtype,
                    choices=choices
                )
            features.append(feature)
        target_name = target
        description = ""

    target_idx = []
    if target_name:
        for i, f in enumerate(features):
            if f.name in target_name:
                target_idx.append(i)

    examples = []
    for _, row in df.iterrows():
        values = [row.get(f.name, None) for f in features]
        examples.append(
            Example(
                features=features,
                values=values,
                target_indices=target_idx,
                dataset_context=description,
                support_examples=None
            )
        )
    return examples
