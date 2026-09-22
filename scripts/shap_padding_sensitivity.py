"""Controlled missing-block sensitivity analysis for SHAP attributions.

This script is intended when the original missing-position mask cannot be
recovered. It injects identical contiguous missing blocks, compares row-median
padding with linear interpolation, refits the locked ensemble, and reports
attribution-rank stability. The analysis is a simulated perturbation and must
not be described as reconstruction of the original missingness pattern.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.cv_whsve.model_pool import build_candidate_models
from src.utils.data import load_plr_table
from src.utils.features import extract_plr_features
from src.utils.metrics import binary_metrics


def _inject_missing_blocks(
    sequences: np.ndarray, fraction: float, random_state: int
) -> np.ndarray:
    if not 0 < fraction < 1:
        raise ValueError("missing fraction must be between 0 and 1")
    rng = np.random.default_rng(random_state)
    result = np.asarray(sequences, dtype=float).copy()
    block = max(1, int(round(result.shape[1] * fraction)))
    for row in result:
        start = int(rng.integers(0, result.shape[1] - block + 1))
        row[start : start + block] = np.nan
    return result


def _impute_rows(sequences: np.ndarray, method: str) -> np.ndarray:
    result = np.asarray(sequences, dtype=float).copy()
    for row in result:
        missing = np.isnan(row)
        if not missing.any():
            continue
        observed = np.flatnonzero(~missing)
        if not len(observed):
            raise ValueError("A sequence contains no observed values")
        if method == "median":
            row[missing] = np.nanmedian(row)
        elif method == "linear":
            row[missing] = np.interp(np.flatnonzero(missing), observed, row[observed])
        else:
            raise ValueError("method must be 'median' or 'linear'")
    return result


def _weighted_probability(models: dict, weights: dict[str, float], x: np.ndarray) -> np.ndarray:
    return sum(weights[name] * models[name].predict_proba(x)[:, 1] for name in weights)


def _analyse_method(
    method: str,
    sequences: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    weights: dict[str, float],
    random_state: int,
    max_explanations: int,
) -> tuple[pd.DataFrame, dict]:
    import shap

    imputed = _impute_rows(sequences, method)
    features, names = extract_plr_features(imputed)
    candidates = build_candidate_models(random_state, profile="full")
    models = {
        name: clone(candidates[name]).fit(features[train_idx], labels[train_idx])
        for name in weights
    }
    predict = lambda values: _weighted_probability(models, weights, values)
    background_size = min(50, len(train_idx))
    rng = np.random.default_rng(random_state)
    background_idx = rng.choice(train_idx, size=background_size, replace=False)
    explanation_idx = test_idx[: min(max_explanations, len(test_idx))]
    explainer = shap.Explainer(
        predict,
        features[background_idx],
        algorithm="permutation",
        feature_names=names,
        seed=random_state,
    )
    explanation = explainer(
        features[explanation_idx], max_evals=2 * len(names) + 1
    )
    values = np.asarray(explanation.values)
    importance = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_signed_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    test_probability = predict(features[test_idx])
    metrics = binary_metrics(
        labels[test_idx], (test_probability >= 0.5).astype(int), test_probability
    )
    return importance, metrics


def run(
    data: str,
    weights_csv: str,
    out_dir: str,
    *,
    random_state: int = 42,
    missing_fraction: float = 0.05,
    max_explanations: int = 30,
) -> dict:
    sequences, labels, _ = load_plr_table(data)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=random_state
    )
    weight_table = pd.read_csv(weights_csv)
    value_column = next(
        (name for name in weight_table.columns if name.endswith("weight")), None
    )
    if "model" not in weight_table or value_column is None:
        raise ValueError("weights CSV must contain model and a column ending in 'weight'")
    weights = dict(zip(weight_table["model"], weight_table[value_column]))
    total = sum(weights.values())
    weights = {name: float(value) / total for name, value in weights.items()}

    perturbed = _inject_missing_blocks(sequences, missing_fraction, random_state)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    importances = {}
    for method in ("median", "linear"):
        importance, metrics = _analyse_method(
            method,
            perturbed,
            labels,
            train_idx,
            test_idx,
            weights,
            random_state,
            max_explanations,
        )
        importance.to_csv(output / f"shap_importance_{method}.csv", index=False)
        importances[method] = importance.set_index("feature")
        results[method] = metrics

    # Align by feature name before comparing ranks.  Each importance table is
    # sorted independently, so comparing their raw row order would pair
    # different features and produce an invalid correlation.
    common_features = importances["median"].index.intersection(
        importances["linear"].index, sort=False
    )
    median_rank = (
        importances["median"].loc[common_features, "mean_abs_shap"].rank(ascending=False)
    )
    linear_rank = (
        importances["linear"].loc[common_features, "mean_abs_shap"].rank(ascending=False)
    )
    rank_correlation = float(spearmanr(median_rank, linear_rank)[0])
    median_top = set(importances["median"].head(5).index)
    linear_top = set(importances["linear"].head(5).index)
    top5_overlap = len(median_top & linear_top) / 5
    union = median_top | linear_top
    direction_agreement = float(
        np.mean(
            np.sign(importances["median"].loc[list(union), "mean_signed_shap"])
            == np.sign(importances["linear"].loc[list(union), "mean_signed_shap"])
        )
    )
    summary = {
        "missing_fraction": missing_fraction,
        "missing_frames_per_sequence": int(round(sequences.shape[1] * missing_fraction)),
        "spearman_rank_correlation": rank_correlation,
        "top5_overlap_fraction": top5_overlap,
        "top_feature_direction_agreement": direction_agreement,
        "median_top5": "; ".join(importances["median"].head(5).index),
        "linear_top5": "; ".join(importances["linear"].head(5).index),
    }
    pd.DataFrame(
        [
            {"imputation": method, **metrics}
            for method, metrics in results.items()
        ]
    ).to_csv(output / "padding_sensitivity_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "padding_sensitivity_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="outputs/shap_padding_sensitivity")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--missing-fraction", type=float, default=0.05)
    parser.add_argument("--max-explanations", type=int, default=30)
    args = parser.parse_args()
    print(
        run(
            args.data,
            args.weights,
            args.out,
            random_state=args.random_state,
            missing_fraction=args.missing_fraction,
            max_explanations=args.max_explanations,
        )
    )
