"""Generate a small, non-clinical CSV for executable code checks only.

The generated values are simulated and must not be used to support any
scientific claim, performance estimate, or participant-level inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(path: str | Path, n_samples: int = 120, random_state: int = 7) -> Path:
    rng = np.random.default_rng(random_state)
    time = np.arange(125, dtype=float)
    labels = np.tile([0, 1], n_samples // 2 + 1)[:n_samples]
    rng.shuffle(labels)
    rows = []
    for index, label in enumerate(labels):
        baseline = rng.normal(4.2, 0.25)
        onset = rng.normal(38.0, 2.0)
        depth = rng.normal(0.75 + 0.12 * label, 0.10)
        recovery = rng.normal(38.0 + 5.0 * label, 4.0)
        constriction = depth / (1.0 + np.exp(-(time - onset) / 2.3))
        rebound = depth * (1.0 - np.exp(-np.maximum(time - onset, 0) / recovery))
        sequence = baseline - constriction + rebound + rng.normal(0.0, 0.025, size=125)
        rows.append([f"synthetic_{index:04d}", *sequence.tolist(), int(label)])
    columns = ["sample_id", *[f"D{i}" for i in range(1, 126)], "label"]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_plr.csv")
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()
    print(generate(args.out, args.n_samples, args.random_state))
