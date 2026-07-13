"""Ablation-style evaluation for the five selected CV-WHSVE base learners."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

from src.cv_whsve.run_cv_whsve_with_shap import build_selected_models
from src.utils.data import load_plr_csv, participant_split
from src.utils.features import extract_plr_features, remove_zero_variance
from src.utils.metrics import binary_metrics


def run(data: str | None, out_dir: str, random_state: int = 42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_raw, y = load_plr_csv(data)
    x_train_raw, x_test_raw, y_train, y_test = participant_split(x_raw, y, random_state=random_state)
    x_train, names = extract_plr_features(x_train_raw)
    x_test, _ = extract_plr_features(x_test_raw)
    x_train, x_test, _ = remove_zero_variance(x_train, x_test, names)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    rows = []
    for name, model in build_selected_models(random_state).items():
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        rows.append({"model": name, **binary_metrics(y_test, y_pred, y_score)})
    pd.DataFrame(rows).to_csv(out / "base_learner_ablation.csv", index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(args.data, args.out, args.random_state)
