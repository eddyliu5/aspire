from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model import DatasetBundle, N_REG_BINS, _fit_reg_bins


def _to_col_type(dtype: str) -> str:
    return "num" if dtype == "continuous" else "cat"


def _infer_feature_from_series(col: str, series: pd.Series) -> Tuple[str, Optional[Dict]]:
    cd = series.dropna()
    if len(cd) < 3:
        return "cat", None
    if str(cd.dtype) in ("object", "category"):
        return "cat", None
    nv = pd.to_numeric(cd, errors="coerce").dropna()
    if len(nv) >= len(cd) * 0.8 and cd.nunique() > 10:
        return "num", None
    return "cat", None


def build_bundle_from_feature_specs(
    X: Any,
    y: Sequence[Any],
    feature_specs: Sequence[Mapping[str, Any]],
    dataset_context: str = "",
    target_col: str = "__target__",
) -> DatasetBundle:
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)
    X_df.columns = [str(c) for c in X_df.columns]

    y_series = pd.Series(y).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    df = X_df.copy()
    df[target_col] = y_series.values

    col_types: Dict[str, str] = {}
    feature_descs: Dict[str, str] = {}
    col_names: List[str] = []

    spec_by_name = {s["name"]: s for s in feature_specs if s["name"] != target_col}
    for col in X_df.columns:
        spec = spec_by_name.get(col)
        if spec:
            col_types[col] = _to_col_type(spec["dtype"])
            feature_descs[col] = spec.get("description") or col
        else:
            ct, _ = _infer_feature_from_series(col, X_df[col])
            col_types[col] = ct
            feature_descs[col] = col.replace("_", " ")
        col_names.append(col)

    y_unique = y_series.astype(str).nunique()
    if y_series.dtype == object or y_unique <= 30:
        col_types[target_col] = "cat"
    else:
        col_types[target_col] = "num"
    feature_descs[target_col] = target_col.replace("_", " ")
    col_names.append(target_col)

    num_scalers: Dict[str, Tuple[float, float]] = {}
    reg_bin_edges: Dict[str, np.ndarray] = {}
    for col in col_names:
        if col_types[col] == "num":
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(vals) > 1:
                mu, sigma = float(vals.mean()), float(vals.std())
                if sigma < 1e-8:
                    sigma = 1.0
                num_scalers[col] = (mu, sigma)
                reg_bin_edges[col] = _fit_reg_bins(vals, N_REG_BINS)

    df = df[col_names].dropna(subset=[target_col]).reset_index(drop=True)

    return DatasetBundle(
        name=dataset_context or "dataset",
        df=df,
        columns=col_names,
        col_types=col_types,
        feature_descs=feature_descs,
        dataset_desc=dataset_context or "",
        num_scalers=num_scalers,
        reg_bin_edges=reg_bin_edges,
        fixed_targets=[target_col],
    )


def infer_feature_specs(
    df: pd.DataFrame,
    target_col: str,
    feature_descriptions: Optional[Dict[str, str]] = None,
    max_cat_unique: int = 200,
) -> List[Dict[str, Any]]:
    fd = feature_descriptions or {}
    non_target, target_spec = [], None

    for col in df.columns:
        cd = df[col].dropna()
        if len(cd) < 3:
            continue
        desc = fd.get(col, col.replace("_", " "))

        if str(cd.dtype) in ("object", "category", "bool"):
            uniq = sorted(cd.astype(str).unique())
            if 2 <= len(uniq) <= max_cat_unique:
                spec = {"name": col, "description": desc, "dtype": "categorical", "choices": uniq}
            else:
                continue
        else:
            nv = pd.to_numeric(cd, errors="coerce").dropna()
            if len(nv) < len(cd) * 0.7:
                continue
            uniq_vals = nv.unique()
            if len(uniq_vals) <= 20 and all(float(v) == int(float(v)) for v in uniq_vals if np.isfinite(float(v))):
                choices = sorted(str(int(v)) for v in uniq_vals if np.isfinite(float(v)))
                if 2 <= len(choices) <= max_cat_unique:
                    spec = {"name": col, "description": desc, "dtype": "categorical", "choices": choices}
                else:
                    spec = {"name": col, "description": desc, "dtype": "continuous",
                            "value_range": (float(nv.min()), float(nv.max()))}
            else:
                spec = {"name": col, "description": desc, "dtype": "continuous",
                        "value_range": (float(nv.min()), float(nv.max()))}

        if col == target_col:
            target_spec = spec
        else:
            non_target.append(spec)

    if target_spec is not None:
        non_target.append(target_spec)
    return non_target
