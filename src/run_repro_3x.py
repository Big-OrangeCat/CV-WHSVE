"""Repeat the complete nested-validation analysis across random seeds.

Despite the historical filename, the default is now ten runs, as requested
for the revision. Each seed creates a new participant-level 80:20 split and a
complete nested model-selection analysis. The across-seed summary uses the
same fixed 0.5 probability threshold in every run so that results are directly
comparable; the development-derived threshold remains available in each run's
``test_metrics.csv`` for the separate threshold analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.cv_whsve.nested_validation import run_nested_evaluation


DEFAULT_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]


def _summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "auroc",
        "brier",
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": rows[metric].mean(),
                "sd": rows[metric].std(ddof=1) if len(rows) > 1 else 0.0,
                "n_runs": len(rows),
            }
            for metric in metrics
        ]
    )


def run_repeated(
    data: str | None,
    out_dir: str,
    seeds: list[int],
    *,
    outer_splits: int = 5,
    inner_splits: int = 5,
    profile: str = "full",
    ga_population: int = 100,
    ga_generations: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in seeds:
        result = run_nested_evaluation(
            data,
            out / f"seed_{seed}",
            random_state=seed,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            profile=profile,
            ga_options={
                "population_size": ga_population,
                "max_generations": ga_generations,
            },
        )
        rows.append(
            {
                "random_state": seed,
                "selected_threshold": result["selected_threshold"],
                "final_models": ";".join(result["final_models"]),
                **result["default_metrics"],
            }
        )
        print(f"Completed seed {seed}: AUROC={rows[-1]['auroc']:.4f}")

    per_seed = pd.DataFrame(rows)
    summary = _summary_table(per_seed)
    per_seed.to_csv(out / "per_seed_metrics.csv", index=False)
    summary.to_csv(out / "repeated_split_summary.csv", index=False)
    (out / "repeated_split_summary.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "outer_splits": outer_splits,
                "inner_splits": inner_splits,
                "profile": profile,
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return per_seed, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repeated nested CV-WHSVE evaluation with mean and sample SD."
    )
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default="outputs/repeated_nested")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--ga-population", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=100)
    args = parser.parse_args()
    _, table = run_repeated(
        args.data,
        args.out,
        args.seeds,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        profile=args.profile,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
    )
    print(table.to_string(index=False))
