"""Command-line entry point for leakage-safe CV-WHSVE validation.

The historical filename is retained so existing commands continue to work.
The current implementation performs nested model selection and weight
derivation before the independent held-out evaluation.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.cv_whsve.model_pool import build_selected_models
from src.cv_whsve.nested_validation import run_nested_evaluation


def weighted_predict_proba(models: dict, weights: dict[str, float], x) -> np.ndarray:
    """Combine fitted classifiers; retained for auxiliary scripts."""
    positive = np.zeros(len(x), dtype=float)
    for name, model in models.items():
        positive += weights[name] * model.predict_proba(x)[:, 1]
    return np.column_stack([1.0 - positive, positive])


def run(
    data: str | None,
    out_dir: str,
    random_state: int = 42,
    *,
    outer_splits: int = 5,
    inner_splits: int = 5,
    profile: str = "full",
    ga_population: int = 100,
    ga_generations: int = 100,
):
    result = run_nested_evaluation(
        data,
        out_dir,
        random_state=random_state,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        profile=profile,
        ga_options={
            "population_size": ga_population,
            "max_generations": ga_generations,
        },
    )
    print("Final classifiers:", ", ".join(result["final_models"]))
    print("Nested outer-fold weights:", result["weights"])
    print(f"Training-OOF calibrated threshold: {result['selected_threshold']:.4f}")
    print("Held-out metrics at probability 0.5:")
    for name, value in result["default_metrics"].items():
        print(f"  {name}: {value:.4f}")
    print("Held-out metrics at the development-derived threshold:")
    for name, value in result["threshold_metrics"].items():
        print(f"  {name}: {value:.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nested-validation CV-WHSVE evaluation on a held-out participant split."
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Local CSV path. Defaults to PLR_DATA_CSV or data/preprocessed_data.csv.",
    )
    parser.add_argument("--out", default="outputs", help="Generated output directory.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument(
        "--profile",
        choices=["full", "smoke"],
        default="full",
        help="Use 'full' for scientific analyses; 'smoke' is synthetic-data testing only.",
    )
    parser.add_argument("--ga-population", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=100)
    args = parser.parse_args()
    run(
        args.data,
        args.out,
        args.random_state,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        profile=args.profile,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
    )
