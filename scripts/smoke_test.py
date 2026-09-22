"""End-to-end executable check using the bundled synthetic CSV."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_synthetic_data import generate
from src.cv_whsve.nested_validation import run_nested_evaluation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = generate(root / "data" / "synthetic_plr.csv", n_samples=80, random_state=7)
    result = run_nested_evaluation(
        str(data),
        root / "outputs" / "synthetic_smoke_test",
        random_state=42,
        outer_splits=2,
        inner_splits=2,
        profile="smoke",
        ga_options={
            "population_size": 8,
            "max_generations": 3,
            "early_stop_generations": 2,
        },
    )
    required = {"accuracy", "f1", "auroc", "brier"}
    if not required.issubset(result["threshold_metrics"]):
        raise RuntimeError("Smoke test did not produce the required metrics")
    metadata = json.loads(
        (root / "outputs" / "synthetic_smoke_test" / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata["n_predefined_features"] != 44:
        raise RuntimeError("Expected the documented 44-feature pre-filter schema")
    removed = {
        "baseline_reference",
        "stimulus_onset_frame",
        "sequence_length",
    } - set(metadata["retained_features"])
    if len(removed) != 3:
        raise RuntimeError("Expected the three invariant legacy candidates to be filtered")
    if metadata["n_retained_on_full_training_partition"] != 41:
        raise RuntimeError("Expected the documented 41-feature modelling schema")
    print("Synthetic end-to-end smoke test passed.")


if __name__ == "__main__":
    main()
