import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
from .model import Feature, Example

def _infer_features(df):
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
            feature = Feature(name=column,description=column,dtype=dtype,value_range=value_range)
        else:
            dtype = "categorical"
            choices = list(dict.fromkeys(str(v) for v in non_na)) if not non_na.empty else []
            feature = Feature(name=column,description=column,dtype=dtype,choices=choices)
        features.append(feature)
    return features

def create_metadata_template(csv_path: str, output_path: Optional[str] = None, overwrite: bool = False):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    inferred_features = _infer_features(df)

    if output_path is not None:
        metadata_path = Path(output_path)
    else:
        metadata_path = csv_path.with_name(f"metadata_{csv_path.stem}.json")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Metadata file already exists: {metadata_path}")

    feature_desc = [vars(f) for f in inferred_features]
    target = []
    dataset_desc = "Dataset description here"
    json_content = {'dataset_desc': dataset_desc, 'target': target, 'feature_desc': feature_desc}

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4, ensure_ascii=False)
    print("Please edit metadata file at ", metadata_path)
    return metadata_path

def add_metadata(df, feature_desc: List[Feature] = [], target: List[str] = [], dataset_desc: str = "", metadata: dict = {}):
    """Adds metadata to df"""
    if metadata:
        df.attrs = metadata
        return df
    if not feature_desc:
        inferred = _infer_features(df)
        feature_desc_standardized = [vars(f) for f in inferred]
    else:
        first = feature_desc[0]
        if isinstance(first, Feature):
            inferred = list(feature_desc)
            feature_desc_standardized = [vars(f) for f in inferred]
        elif isinstance(first, dict):
            required_keys = {"name", "description", "dtype"}
            for feature in feature_desc:
                if not required_keys.issubset(feature.keys()):
                    raise ValueError("Each feature metadata dict must contain name, description, and dtype.")
                dtype = feature.get("dtype")
                has_choices = any(key in feature for key in ("choices", "categories"))
                has_range = any(key in feature for key in ("value_range", "range"))
                if dtype == "continuous" and not has_range:
                    raise ValueError(f"Continuous feature '{feature.get('name')}' missing a range/value_range.")
                if dtype in {"categorical", "discrete"} and not has_choices:
                    raise ValueError(f"Categorical feature '{feature.get('name')}' missing categories/choices.")
            feature_desc_standardized = feature_desc
        elif isinstance(first, str):
            if len(feature_desc) != len(df.columns):
                raise ValueError("Length of feature descriptions must match number of DataFrame columns.")
            inferred = _infer_features(df)
            for feature, desc in zip(inferred, feature_desc):
                feature.description = desc
            feature_desc_standardized = [vars(f) for f in inferred]
        else:
            raise TypeError("feature_desc must be a list of Feature objects or a list of strings.")

    df.attrs["feature_desc"] = feature_desc_standardized
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
        features = _infer_features(df)
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
