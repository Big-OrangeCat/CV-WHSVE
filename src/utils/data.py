"""Data loading utilities for PLR experiments.

The repository intentionally does not include participant-level data.
Set PLR_DATA_CSV or pass --data to each script.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_DATA_PATH = os.getenv("PLR_DATA_CSV", "data/preprocessed_data.csv")
TIME_COLUMNS = [f"D{i}" for i in range(1, 126)]
LABEL_COLUMN = "label"


def load_plr_csv(path: str | os.PathLike[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV containing D1-D125 and label columns."""
    csv_path = Path(path or DEFAULT_DATA_PATH)
    df = pd.read_csv(csv_path)
    missing = [c for c in TIME_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    x = df[TIME_COLUMNS].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    return x, y


def participant_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Participant-level stratified split used by the manuscript."""
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)
