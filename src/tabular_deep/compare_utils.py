#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = os.getenv("PLR_DATA_CSV", "data/preprocessed_data.csv")
DEFAULT_TIME_COLS = [f"D{i}" for i in range(1, 126)]
DEFAULT_LABEL_COL = "label"


def get_seed(default: int = 1412) -> int:
    return int(os.getenv("REPRO_SEED", os.getenv("PYTHONHASHSEED", str(default))))


def load_data(
    csv_path: str = DEFAULT_DATA_PATH,
    time_cols: list[str] | None = None,
    label_col: str = DEFAULT_LABEL_COL,
):
    cols = time_cols or DEFAULT_TIME_COLS
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label column not found: {label_col}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing[:5]} ...")

    X = df[cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int64)
    return X, y, cols


def split_data(X, y, seed: int, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def preprocess_numeric(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train.astype(np.float32), X_test.astype(np.float32), imputer, scaler


def _ensure_binary_proba(y_proba_raw):
    arr = np.asarray(y_proba_raw)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def evaluate_binary(y_true, y_pred, y_proba):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_proba = _ensure_binary_proba(y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
    }


def print_metrics(metrics: dict):
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['auc']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")


def save_outputs(
    method_name: str,
    seed: int,
    metrics: dict,
    y_true,
    y_pred,
    y_proba,
    base_dir: str | Path | None = None,
):
    out_root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    out_dir = out_root / "compare_results" / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"seed{seed}"

    metrics_payload = {"method": method_name, "seed": seed, **metrics}
    with open(out_dir / f"metrics_{run_tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).reshape(-1),
            "y_pred": np.asarray(y_pred).reshape(-1),
            "y_proba": _ensure_binary_proba(y_proba),
        }
    )
    pred_df.to_csv(out_dir / f"predictions_{run_tag}.csv", index=False)

    summary_csv = out_dir / "metrics_summary.csv"
    row_df = pd.DataFrame([metrics_payload])
    if summary_csv.exists():
        hist = pd.read_csv(summary_csv)
        hist = pd.concat([hist, row_df], ignore_index=True)
        hist = hist.drop_duplicates(subset=["seed"], keep="last").sort_values("seed")
        hist.to_csv(summary_csv, index=False)
    else:
        row_df.to_csv(summary_csv, index=False)

    print(f"Saved: {out_dir}")

