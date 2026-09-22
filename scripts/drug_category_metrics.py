"""Calculate exploratory drug-category metrics from locked test predictions.

The category file must be author supplied and must use the same non-identifying
row key as ``test_predictions.csv``. No drug category is inferred from the
binary label or aggregate counts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from scipy.stats import beta

from src.utils.metrics import binary_metrics


def _binomial_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Clopper--Pearson interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def run(
    predictions: str,
    categories: str,
    output: str,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    prediction_table = pd.read_csv(predictions)
    category_table = pd.read_csv(categories)
    required_predictions = {"row_id", "y_true", "calibrated_probability"}
    required_categories = {"row_id", "drug_category"}
    if not required_predictions.issubset(prediction_table.columns):
        raise ValueError(f"Prediction file requires {sorted(required_predictions)}")
    if not required_categories.issubset(category_table.columns):
        raise ValueError(f"Category file requires {sorted(required_categories)}")

    category_table["drug_category"] = category_table["drug_category"].astype(str).map(
        lambda value: [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]
    )
    category_table = category_table.explode("drug_category")
    positive = prediction_table.loc[prediction_table["y_true"] == 1].merge(
        category_table, on="row_id", how="left", validate="one_to_many"
    )
    if positive["drug_category"].isna().any():
        missing = positive.loc[positive["drug_category"].isna(), "row_id"].tolist()
        raise ValueError(f"Missing drug category for positive-screen row keys: {missing[:10]}")
    controls = prediction_table.loc[prediction_table["y_true"] == 0]

    rows = []
    for category, category_positive in positive.groupby("drug_category"):
        subset = pd.concat([controls, category_positive[prediction_table.columns]], ignore_index=True)
        score = subset["calibrated_probability"].to_numpy()
        predicted = (score >= threshold).astype(int)
        metrics = binary_metrics(subset["y_true"], predicted, score)
        positive_score = category_positive["calibrated_probability"].to_numpy()
        true_positives = int((positive_score >= threshold).sum())
        false_negatives = int(len(category_positive) - true_positives)
        ci_lower, ci_upper = _binomial_interval(true_positives, len(category_positive))
        rows.append(
            {
                "drug_category": category,
                "n_positive": len(category_positive),
                "n_controls": len(controls),
                "threshold": threshold,
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "sensitivity_ci_lower": ci_lower,
                "sensitivity_ci_upper": ci_upper,
                **metrics,
            }
        )
    result = pd.DataFrame(rows).sort_values("drug_category")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--categories", required=True)
    parser.add_argument("--out", default="outputs/drug_category_metrics.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(
        run(
            args.predictions,
            args.categories,
            args.out,
            threshold=args.threshold,
        ).to_string(index=False)
    )
