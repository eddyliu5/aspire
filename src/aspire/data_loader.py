import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import pandas as pd
from .model import Feature, Example

REQUIRED_TRAINING_METADATA_FIELDS = ["dataset_description", "feature_descriptions", "task_description"]

REQUIRED_FEATURE_SPEC_FIELDS = {"name", "description", "dtype"}


def validate_training_metadata(metadata: Mapping[str, Any], n_features: int) -> Dict[str, Any]:
    """Validate metadata schema required by `AspireModel.fit`."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping/dictionary.")

    missing = [key for key in REQUIRED_TRAINING_METADATA_FIELDS if key not in metadata]
    if missing:
        raise ValueError(f"metadata missing required fields: {missing}")

    dataset_description = metadata.get("dataset_description")
    feature_descriptions = metadata.get("feature_descriptions")
    task_description = metadata.get("task_description")
    target_description = metadata.get("target_description")

    if not isinstance(dataset_description, str) or not dataset_description.strip():
        raise ValueError("metadata['dataset_description'] must be a non-empty string.")
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("metadata['task_description'] must be a non-empty string.")
    if not isinstance(feature_descriptions, list):
        raise ValueError("metadata['feature_descriptions'] must be a list of strings.")
    if len(feature_descriptions) != n_features:
        raise ValueError(
            "metadata['feature_descriptions'] must have length equal to number of X columns "
            f"({n_features}), got {len(feature_descriptions)}."
        )
    if any(not isinstance(text, str) or not text.strip() for text in feature_descriptions):
        raise ValueError("Each item in metadata['feature_descriptions'] must be a non-empty string.")
    if target_description is not None and not isinstance(target_description, str):
        raise ValueError("metadata['target_description'] must be a string when provided.")

    return {
        "dataset_description": dataset_description.strip(),
        "feature_descriptions": [text.strip() for text in feature_descriptions],
        "task_description": task_description.strip(),
        "target_description": target_description.strip() if isinstance(target_description, str) else None,
    }


def validate_feature_specs(feature_specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize feature specs for manual ASPIRE inference metadata."""
    if not isinstance(feature_specs, Sequence) or isinstance(feature_specs, (str, bytes)):
        raise TypeError("feature_specs must be a sequence of dictionaries.")
    if len(feature_specs) == 0:
        raise ValueError("feature_specs cannot be empty.")

    normalized_specs: List[Dict[str, Any]] = []
    names: List[str] = []

    for idx, spec in enumerate(feature_specs):
        if not isinstance(spec, Mapping):
            raise TypeError(f"feature_specs[{idx}] must be a dictionary.")
        missing = REQUIRED_FEATURE_SPEC_FIELDS - set(spec.keys())
        if missing:
            raise ValueError(f"feature_specs[{idx}] missing required fields: {sorted(missing)}")

        name = str(spec["name"]).strip()
        description = str(spec["description"]).strip()
        dtype = str(spec["dtype"]).strip().lower()

        if not name:
            raise ValueError(f"feature_specs[{idx}]['name'] must be non-empty.")
        if not description:
            raise ValueError(f"feature_specs[{idx}]['description'] must be non-empty.")
        if dtype not in {"continuous", "categorical"}:
            raise ValueError(
                f"feature_specs[{idx}]['dtype'] must be 'continuous' or 'categorical', got '{dtype}'."
            )

        normalized: Dict[str, Any] = {
            "name": name,
            "description": description,
            "dtype": dtype,
            "choices": spec.get("choices"),
            "value_range": spec.get("value_range"),
        }

        if dtype == "continuous":
            value_range = normalized.get("value_range")
            if value_range is not None:
                if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
                    raise ValueError(
                        f"feature_specs[{idx}]['value_range'] must be a 2-item list/tuple when provided."
                    )
                normalized["value_range"] = (float(value_range[0]), float(value_range[1]))
            normalized["choices"] = None
        else:
            choices = normalized.get("choices")
            if choices is not None and not isinstance(choices, (list, tuple)):
                raise ValueError(
                    f"feature_specs[{idx}]['choices'] must be a list/tuple when provided."
                )
            if choices is not None:
                normalized["choices"] = [str(choice) for choice in choices]
            normalized["value_range"] = None

        normalized_specs.append(normalized)
        names.append(name)

    if len(set(names)) != len(names):
        raise ValueError("feature_specs contains duplicate feature names.")

    return normalized_specs


def _coerce_X_dataframe(X: Any) -> pd.DataFrame:
    """Coerce X to DataFrame with stable string column names."""
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    if X_df.empty or X_df.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature column.")

    X_df.columns = [str(col) for col in X_df.columns]
    if X_df.columns.duplicated().any():
        raise ValueError("X contains duplicate column names.")
    return X_df


def build_training_dataframe(
    X: Any,
    y: Sequence[Any],
    metadata: Mapping[str, Any],
    target_column: str = "__target__",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a metadata-enriched DataFrame suitable for `df_to_examples` during fit."""
    X_df = _coerce_X_dataframe(X)
    y_series = pd.Series(y)

    if len(y_series) != len(X_df):
        raise ValueError(f"y must have the same number of rows as X ({len(X_df)}).")

    normalized_metadata = validate_training_metadata(metadata, n_features=X_df.shape[1])

    train_df = X_df.copy()
    train_df[target_column] = y_series.values

    inferred_features = _infer_features(train_df)
    feature_desc_map = dict(zip(X_df.columns.tolist(), normalized_metadata["feature_descriptions"]))
    target_description = normalized_metadata["target_description"] or target_column

    for feature in inferred_features:
        if feature.name in feature_desc_map:
            feature.description = feature_desc_map[feature.name]
        elif feature.name == target_column:
            feature.description = target_description

    train_df = add_metadata(
        train_df,
        feature_desc=inferred_features,
        target=[target_column],
        dataset_desc=normalized_metadata["dataset_description"],
    )
    return train_df, normalized_metadata


def build_prediction_dataframe(
    X: Any,
    feature_desc: List[Dict[str, Any]],
    dataset_desc: str,
    target_column: str = "__target__",
) -> pd.DataFrame:
    """Build metadata-enriched DataFrame for prediction from raw X."""
    if not feature_desc:
        raise ValueError("feature_desc cannot be empty for prediction.")

    X_df = _coerce_X_dataframe(X)
    expected_columns = [item["name"] for item in feature_desc if item["name"] != target_column]

    if set(expected_columns).issubset(set(X_df.columns)):
        X_ordered = X_df.loc[:, expected_columns].copy()
    elif len(X_df.columns) == len(expected_columns):
        X_ordered = X_df.copy()
        X_ordered.columns = expected_columns
    else:
        raise ValueError(f"X columns do not match expected features: {expected_columns}")

    X_ordered[target_column] = None
    pred_df = add_metadata(
        X_ordered,
        feature_desc=feature_desc,
        target=[target_column],
        dataset_desc=dataset_desc,
    )
    return pred_df


def build_examples_from_feature_specs(
    X: Any,
    feature_specs: Sequence[Mapping[str, Any]],
    dataset_context: str = "",
    target_indices: Optional[Sequence[int]] = None,
) -> List[Example]:
    """
    Build `Example` objects directly from feature specs.

    This path keeps explicit dtype/range/choices from user-provided specs.
    """
    specs = validate_feature_specs(feature_specs)
    X_df = _coerce_X_dataframe(X)
    features = [Feature(**spec) for spec in specs]

    if target_indices is None:
        resolved_target_indices = [len(features) - 1]
    else:
        resolved_target_indices = [int(i) for i in target_indices]

    for idx in resolved_target_indices:
        if idx < 0 or idx >= len(features):
            raise ValueError(
                f"target_indices contains invalid index {idx} for {len(features)} feature specs."
            )

    context = str(dataset_context or "")
    examples: List[Example] = []
    feature_names = [feature.name for feature in features]

    for _, row in X_df.iterrows():
        values = [row.get(name, None) for name in feature_names]
        examples.append(
            Example(
                features=features,
                values=values,
                target_indices=resolved_target_indices,
                dataset_context=context,
                support_examples=None,
            )
    )
    return examples


def build_training_examples_from_feature_specs(
    X: Any,
    y: Sequence[Any],
    feature_specs: Sequence[Mapping[str, Any]],
    dataset_context: str = "",
    target_indices: Optional[Sequence[int]] = None,
) -> List[Example]:
    """
    Build training examples from X, y using explicit feature specs.

    This preserves user-defined dtype/range/choices in feature metadata.
    """
    specs = validate_feature_specs(feature_specs)
    X_df = _coerce_X_dataframe(X)
    y_series = pd.Series(y)
    if len(y_series) != len(X_df):
        raise ValueError(f"y must have the same number of rows as X ({len(X_df)}).")

    features = [Feature(**spec) for spec in specs]
    if target_indices is None:
        resolved_target_indices = [len(features) - 1]
    else:
        resolved_target_indices = [int(i) for i in target_indices]

    if len(resolved_target_indices) != 1:
        raise ValueError("Only single-target training is currently supported with feature_specs.")

    target_idx = resolved_target_indices[0]
    if target_idx < 0 or target_idx >= len(features):
        raise ValueError(f"Invalid target index {target_idx} for {len(features)} feature specs.")

    feature_names = [feature.name for feature in features]
    context = str(dataset_context or "")
    examples: List[Example] = []

    for row_idx, row in X_df.iterrows():
        values = [row.get(name, None) for name in feature_names]
        values[target_idx] = y_series.iloc[row_idx]
        examples.append(
            Example(
                features=features,
                values=values,
                target_indices=[target_idx],
                dataset_context=context,
                support_examples=None,
            )
        )
    return examples


def training_metadata_from_feature_specs(
    X: Any,
    feature_specs: Sequence[Mapping[str, Any]],
    dataset_context: str = "",
    target_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Derive fit metadata from feature specs.

    Feature descriptions are aligned to X columns by feature name when possible.
    """
    specs = validate_feature_specs(feature_specs)
    X_df = _coerce_X_dataframe(X)
    desc_by_name = {spec["name"]: spec["description"] for spec in specs}

    if target_indices is None:
        resolved_target_indices = [len(specs) - 1]
    else:
        resolved_target_indices = [int(i) for i in target_indices]

    target_description = None
    if resolved_target_indices:
        primary_idx = resolved_target_indices[0]
        if 0 <= primary_idx < len(specs):
            target_description = specs[primary_idx]["description"]

    feature_descriptions = [desc_by_name.get(str(col), str(col)) for col in X_df.columns]
    return {
        "dataset_description": str(dataset_context or "Dataset context not provided."),
        "feature_descriptions": feature_descriptions,
        "task_description": "Task description not provided.",
        "target_description": target_description,
    }

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
