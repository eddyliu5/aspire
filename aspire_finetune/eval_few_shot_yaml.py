#!/usr/bin/env python3
"""Evaluate gradient-free ASPIRE few-shot inference on the held-out YAML suite."""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ICLR_DIR = ROOT / "TabSTAR" / "aspire_iclr_2026"
sys.path.insert(0, str(ROOT))
TABSTAR_DIR = ROOT / "TabSTAR"
sys.path.insert(0, str(ICLR_DIR))
sys.path.insert(0, str(TABSTAR_DIR))

from eval_markov_shots import DATASET_SPECS, build_bundle  # noqa: E402
from aspire_finetune import ASPIRE  # noqa: E402


DEFAULT_YAML = ROOT / "universal_machine_old" / "filtered_test_datasets.yaml"
DEFAULT_CHECKPOINT = (
    ROOT / "TabSTAR" / "checkpoints" / "tabstar_icl_v5_mog_desc08" / "best_model.pt"
)
DEFAULT_OUTPUT = ROOT / "aspire_finetune" / "few_shot_yaml_results.json"
logger = logging.getLogger("aspire_few_shot_yaml")


def feature_specs_from_bundle(bundle, target_col):
    specs = []
    for col in bundle.columns:
        spec = {
            "name": col,
            "description": bundle.feature_descs.get(col, col.replace("_", " ")),
            "dtype": "continuous" if bundle.col_types[col] == "num" else "categorical",
        }
        if bundle.col_types[col] == "cat":
            spec["choices"] = sorted(bundle.df[col].dropna().astype(str).unique().tolist())
        specs.append(spec)
    return specs


def split_rows(bundle, target_col, test_fraction, seed, max_query):
    indices = np.arange(len(bundle.df))
    stratify = None
    if bundle.col_types[target_col] == "cat":
        counts = bundle.df[target_col].astype(str).value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = bundle.df[target_col].astype(str)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(test_idx)
    test_idx = test_idx[:max_query]
    features = [col for col in bundle.columns if col != target_col]
    X_train = bundle.df.iloc[train_idx][features].reset_index(drop=True)
    y_train = bundle.df.iloc[train_idx][target_col].reset_index(drop=True)
    X_test = bundle.df.iloc[test_idx][features].reset_index(drop=True)
    y_test = bundle.df.iloc[test_idx][target_col].reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def evaluate_dataset(name, config, args):
    bundle = build_bundle(name, config, args.max_rows)
    if bundle is None:
        return None
    target_col = DATASET_SPECS[name]["target"]
    task_type = "classification" if bundle.col_types[target_col] == "cat" else "regression"
    X_train, y_train, X_test, y_test = split_rows(
        bundle, target_col, args.test_fraction, args.seed, args.max_query
    )
    model = ASPIRE.from_pretrained(
        checkpoint=args.checkpoint,
        device=args.device,
        feature_specs=feature_specs_from_bundle(bundle, target_col),
        dataset_context=bundle.dataset_desc,
        target_column=target_col,
    )
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "random_state": args.seed,
        "task_type": task_type,
    }
    if task_type == "classification":
        fit_kwargs["shots_per_class"] = args.shots
    else:
        fit_kwargs["max_support"] = args.shots
    model.fit_few_shot(**fit_kwargs)
    predictions = model.predict(X_test, batch_size=args.batch_size, include_desc=True)
    # A standard frozen-representation probe: only the selected labeled shots
    # train the probe, and embeddings omit ICL support to avoid self-label leakage.
    support_frame = pd.DataFrame(model._support_rows).reset_index(drop=True)
    feature_columns = [col for col in bundle.columns if col != target_col]
    support_X = support_frame[feature_columns]
    support_y = support_frame[target_col]
    support_embeddings = model.get_embeddings(
        support_X, batch_size=args.batch_size, include_desc=True, use_support=False
    )
    query_embeddings = model.get_embeddings(
        X_test, batch_size=args.batch_size, include_desc=True, use_support=False
    )
    if task_type == "classification":
        shot_probe = LogisticRegression(
            C=args.probe_c, class_weight="balanced", max_iter=3000, solver="lbfgs"
        ).fit(support_embeddings, support_y.astype(str))
    else:
        shot_probe = Ridge(alpha=args.probe_alpha).fit(
            support_embeddings, pd.to_numeric(support_y, errors="coerce")
        )
    shot_probe_predictions = shot_probe.predict(query_embeddings)

    if task_type == "classification":
        truth = y_test.astype(str).tolist()
        # Primary probe protocol retained from eval_icl_5shot_v5.py: fit and
        # score on the same labeled query embeddings.
        probe = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs"
        ).fit(query_embeddings, truth)
        probe_predictions = probe.predict(query_embeddings)

        pred = [str(value) for value in predictions]
        metrics = {
            "accuracy": float(accuracy_score(truth, pred)),
            "f1_weighted": float(f1_score(truth, pred, average="weighted", zero_division=0)),
            "probe_accuracy": float(accuracy_score(truth, probe_predictions)),
            "probe_f1_weighted": float(f1_score(truth, probe_predictions, average="weighted", zero_division=0)),
            "probe_f1_macro": float(f1_score(truth, probe_predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "shot_probe_accuracy": float(accuracy_score(truth, shot_probe_predictions)),
            "shot_probe_f1_weighted": float(f1_score(truth, shot_probe_predictions, average="weighted", zero_division=0)),
            "shot_probe_f1_macro": float(f1_score(truth, shot_probe_predictions, average="macro", zero_division=0)),
        }
    else:
        truth = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
        pred = np.asarray(predictions, dtype=float)
        probe = Ridge(alpha=1.0).fit(query_embeddings, truth)
        probe_predictions = probe.predict(query_embeddings)
        metrics = {
            "rmse": float(math.sqrt(mean_squared_error(truth, pred))),
            "probe_rmse": float(math.sqrt(mean_squared_error(truth, probe_predictions))),
            "probe_r2": float(r2_score(truth, probe_predictions)),
            "shot_probe_rmse": float(math.sqrt(mean_squared_error(truth, shot_probe_predictions))),
            "shot_probe_r2": float(r2_score(truth, shot_probe_predictions)),
            "r2": float(r2_score(truth, pred)),
        }
    return {
        "task_type": task_type,
        "target": target_col,
        "support_size": model.support_size_,
        "query_size": len(y_test),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", default=str(DEFAULT_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--max-query", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-c", type=float, default=1.0)
    parser.add_argument("--probe-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.yaml, "r", encoding="utf-8") as handle:
        configs = yaml.safe_load(handle)
    names = [name for name in configs if name in DATASET_SPECS]
    if args.max_datasets > 0:
        names = names[:args.max_datasets]

    results = {}
    for index, name in enumerate(names, 1):
        logger.info("[%d/%d] %s", index, len(names), name)
        try:
            result = evaluate_dataset(name, configs[name], args)
        except Exception as exc:
            logger.exception("%s failed", name)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        if result is not None:
            results[name] = result
            print(f"{name}: {result}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    classification = [r for r in results.values() if r.get("task_type") == "classification"]
    summary = {
        "datasets_requested": len(names),
        "datasets_evaluated": sum("error" not in r for r in results.values()),
        "failures": sum("error" in r for r in results.values()),
    }
    if classification:
        summary["classification_mean_accuracy"] = float(
            np.mean([r["accuracy"] for r in classification])
        )
        summary["classification_mean_f1_macro"] = float(
            np.mean([r["f1_macro"] for r in classification])
        )
        summary["classification_probe_mean_accuracy"] = float(
            np.mean([r["probe_accuracy"] for r in classification])
        )
        summary["classification_probe_mean_f1_macro"] = float(
            np.mean([r["probe_f1_macro"] for r in classification])
        )
        summary["classification_probe_mean_f1_weighted"] = float(
            np.mean([r["probe_f1_weighted"] for r in classification])
        )
        summary["classification_shot_probe_mean_accuracy"] = float(
            np.mean([r["shot_probe_accuracy"] for r in classification])
        )
        summary["classification_shot_probe_mean_f1_weighted"] = float(
            np.mean([r["shot_probe_f1_weighted"] for r in classification])
        )
        summary["classification_shot_probe_mean_f1_macro"] = float(
            np.mean([r["shot_probe_f1_macro"] for r in classification])
        )
    payload = {"settings": vars(args), "summary": summary, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
