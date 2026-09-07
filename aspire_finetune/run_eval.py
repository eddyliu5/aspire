#!/usr/bin/env python3
"""
Usage:
    # Minimal — auto-infers feature types, last column = target
    python run_eval.py \
        --checkpoint /path/to/checkpoint.pt \
        --data /path/to/data.csv \
        --target_col label

    # With dataset description and feature descriptions file
    python run_eval.py \
        --checkpoint /path/to/checkpoint.pt \
        --data /path/to/data.csv \
        --target_col DEATH_EVENT \
        --dataset_desc "Heart failure clinical records: predict in-hospital mortality" \
        --feat_descs_json /path/to/feat_descs.json \
        --num_epochs 200 \
        --seeds 42 43 44 \
        --xgb_baseline

    # feat_descs.json format:
    # {"age": "Patient age in years", "ejection_fraction": "Ejection fraction (%)", ...}
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))  

from aspire_finetune.data_loader import infer_feature_specs
from aspire_finetune.aspire import ASPIRE


def _load_data(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if p.suffix == ".parquet":
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def _run_one_seed(
    checkpoint: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_specs: List[Dict],
    dataset_desc: str,
    target_col: str,
    finetune_mode: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_support: int,
    val_fraction: float,
    include_desc: bool,
    device: str,
    seed: int,
    unfreeze_top_layers: int,
) -> Dict:
    if finetune_mode == "v2_ft" and learning_rate >= 1e-3:
        learning_rate = 1e-4
    logger.info("  [%s] seed=%d  fitting...", finetune_mode, seed)
    model = ASPIRE.from_pretrained(
        checkpoint=checkpoint,
        device=device,
        feature_specs=feature_specs,
        dataset_context=dataset_desc,
        target_column=target_col,
        unfreeze_top_layers=unfreeze_top_layers,
    )
    model.fit(
        X=X_train, y=y_train,
        finetune_mode=finetune_mode,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_support=num_support,
        test_fraction=val_fraction,
        random_state=seed,
        include_desc=include_desc,
        show_progress=False,
    )
    preds = model.predict(X_test, include_desc=include_desc)
    y_list = y_test.astype(str).tolist()
    preds_str = [str(p) for p in preds]

    acc = accuracy_score(y_list, preds_str)
    f1w = f1_score(y_list, preds_str, average="weighted", zero_division=0)
    f1m = f1_score(y_list, preds_str, average="macro",    zero_division=0)
    logger.info("  [%s] seed=%d  acc=%.4f  f1_w=%.4f  f1_m=%.4f",
                finetune_mode, seed, acc, f1w, f1m)
    return {"accuracy": acc, "f1_weighted": f1w, "f1_macro": f1m}


def _agg(results: List[Dict]) -> Dict:
    keys = list(results[0].keys())
    return {k: {"mean": float(np.mean([r[k] for r in results])),
                "std":  float(np.std( [r[k] for r in results]))} for k in keys}


def _xgb_baseline(X_train, y_train, X_test, y_test) -> Dict:
    try:
        import re
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        def _clean(df):
            df = pd.get_dummies(df).astype(float)
            df.columns = [re.sub(r"[\[\]<>]", "", c) for c in df.columns]
            return df

        Xtr = _clean(X_train); Xte = _clean(X_test)
        Xtr, Xte = Xtr.align(Xte, join="left", axis=1, fill_value=0)
        le = LabelEncoder()
        ytr = le.fit_transform(y_train.astype(str))
        yte = le.transform(y_test.astype(str))
        n_cls = len(le.classes_)
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softmax" if n_cls > 2 else "binary:logistic",
            eval_metric="mlogloss", random_state=42,
            **({"num_class": n_cls} if n_cls > 2 else {}),
        )
        xgb.fit(Xtr, ytr, verbose=False)
        p_enc = xgb.predict(Xte)
        preds = le.inverse_transform(p_enc)
        y_list = y_test.astype(str).tolist()
        return {
            "accuracy":    float(accuracy_score(y_list, preds)),
            "f1_weighted": float(f1_score(y_list, preds, average="weighted", zero_division=0)),
            "f1_macro":    float(f1_score(y_list, preds, average="macro",    zero_division=0)),
        }
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost baseline")
        return {}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--data",       required=True, help="CSV / parquet file path")
    ap.add_argument("--target_col", required=True, help="Target column name")
    ap.add_argument("--dataset_desc", type=str, default="",
                    help="Free-text dataset description (used as conditioning)")
    ap.add_argument("--feat_descs_json", type=str, default="",
                    help="JSON file mapping column names → descriptions")
    ap.add_argument("--test_split",  type=float, default=0.20)
    ap.add_argument("--val_fraction", type=float, default=0.15,
                    help="Fraction of train set used for early stopping")
    ap.add_argument("--num_epochs",  type=int,   default=200)
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--num_support", type=int,   default=0,
                    help="Support-row limit in few_shot mode (0 = use all training rows)")
    ap.add_argument("--include_desc", action="store_true",
                    help="Pass dataset description as conditioning signal")
    ap.add_argument("--unfreeze_top_layers", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--modes", type=str, nargs="+",
                    default=["v2", "head_v2"],
                    choices=["few_shot", "v2", "v2_ft", "v2_xgb", "head_v2"])
    ap.add_argument("--device",      type=str, default="cuda")
    ap.add_argument("--output_json", type=str, default="",
                    help="Optional path to save results JSON")
    ap.add_argument("--xgb_baseline", action="store_true",
                    help="Also run XGBoost as a baseline")
    ap.add_argument("--max_rows",    type=int, default=0,
                    help="Cap dataset rows (0 = no cap)")
    args = ap.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    df = _load_data(args.data)
    if args.max_rows > 0:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=42).reset_index(drop=True)
    df = df.dropna(subset=[args.target_col]).reset_index(drop=True)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], args.data)
    logger.info("Target: '%s'  classes: %s", args.target_col,
                sorted(df[args.target_col].astype(str).unique().tolist()))

    feat_descs: Dict[str, str] = {}
    if args.feat_descs_json and Path(args.feat_descs_json).exists():
        with open(args.feat_descs_json) as f:
            feat_descs = json.load(f)

    # ── feature specs ─────────────────────────────────────────────────────────
    feature_specs = infer_feature_specs(df, args.target_col, feat_descs)
    logger.info("Feature specs: %d features  target='%s'", len(feature_specs), args.target_col)

    # ── train/test split ──────────────────────────────────────────────────────
    X = df.drop(columns=[args.target_col])
    y = df[args.target_col].astype(str)

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_split, random_state=42, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_split, random_state=42)

    X_tr = X_tr.reset_index(drop=True); X_te = X_te.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True); y_te = y_te.reset_index(drop=True)
    logger.info("Train: %d  Test: %d", len(X_tr), len(X_te))

    all_results = {}

    for mode in args.modes:
        logger.info("\n%s\n  Mode: %s\n%s", "=" * 60, mode, "=" * 60)
        seed_results = []
        for seed in args.seeds:
            res = _run_one_seed(
                checkpoint=args.checkpoint,
                X_train=X_tr, y_train=y_tr,
                X_test=X_te,  y_test=y_te,
                feature_specs=feature_specs,
                dataset_desc=args.dataset_desc,
                target_col=args.target_col,
                finetune_mode=mode,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_support=args.num_support,
                val_fraction=args.val_fraction,
                include_desc=args.include_desc,
                device=args.device,
                seed=seed,
                unfreeze_top_layers=args.unfreeze_top_layers,
            )
            seed_results.append(res)

        all_results[mode] = _agg(seed_results)

    # ── XGBoost baseline ──────────────────────────────────────────────────────
    if args.xgb_baseline:
        logger.info("\n%s\n  XGBoost baseline\n%s", "=" * 60, "=" * 60)
        xgb_res = _xgb_baseline(X_tr, y_tr, X_te, y_te)
        if xgb_res:
            logger.info("  acc=%.4f  f1_w=%.4f  f1_m=%.4f",
                        xgb_res["accuracy"], xgb_res["f1_weighted"], xgb_res["f1_macro"])
            all_results["xgboost"] = xgb_res

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  RESULTS  —  {Path(args.data).name}  target='{args.target_col}'")
    print("=" * 72)
    hdr = f"  {'Model':<22} {'Accuracy':>10} {'F1 weighted':>12} {'F1 macro':>10}"
    print(hdr)
    print("  " + "─" * 58)

    def _fmt(res, key):
        v = res.get(key, {})
        if isinstance(v, dict):
            return f"{v['mean']:>10.4f}±{v['std']:.3f}"
        return f"{v:>10.4f}          "

    for name, res in all_results.items():
        print(f"  {name:<22} {_fmt(res,'accuracy')} {_fmt(res,'f1_weighted')} {_fmt(res,'f1_macro')}")

    print("=" * 72)
    print(f"  seeds={args.seeds}  epochs={args.num_epochs}  test_split={args.test_split}"
          f"  unfreeze_top_layers={args.unfreeze_top_layers}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output_json:
        os.makedirs(Path(args.output_json).parent, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()
