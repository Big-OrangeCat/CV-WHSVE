"""Run the GA-guided CV-weighted heterogeneous soft-voting ensemble.

This is a cleaned public script aligned with the manuscript. It does not ship
with data and does not write participant-level data to the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.utils.data import load_plr_csv, participant_split
from src.utils.features import extract_plr_features, remove_zero_variance
from src.utils.metrics import binary_metrics


SELECTED_CLASSIFIERS = ["GaussianProcess", "ExtraTrees", "CatBoost", "GaussianNB", "AdaBoost"]


def build_selected_models(random_state: int = 42):
    return {
        "GaussianProcess": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gpc", GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), optimizer="fmin_l_bfgs_b", random_state=random_state)),
            ]
        ),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=15, criterion="gini", random_state=random_state, n_jobs=-1),
        "CatBoost": CatBoostClassifier(depth=6, learning_rate=0.1, loss_function="Logloss", random_seed=random_state, verbose=0),
        "GaussianNB": GaussianNB(var_smoothing=1e-9),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
        ),
    }


def cv_weights(models: dict, x_train, y_train, random_state: int = 42) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = {}
    for name, model in models.items():
        scores[name] = float(np.mean(cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)))
    total = sum(scores.values())
    return {name: score / total for name, score in scores.items()}


def weighted_predict_proba(models: dict, weights: dict, x_test):
    proba = np.zeros((x_test.shape[0], 2), dtype=np.float64)
    for name, model in models.items():
        proba += weights[name] * model.predict_proba(x_test)
    return proba


def run(data: str | None, out_dir: str, random_state: int = 42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_raw, y = load_plr_csv(data)
    x_train_raw, x_test_raw, y_train, y_test = participant_split(x_raw, y, random_state=random_state)

    x_train, feature_names = extract_plr_features(x_train_raw)
    x_test, _ = extract_plr_features(x_test_raw)
    x_train, x_test, feature_names = remove_zero_variance(x_train, x_test, feature_names)

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    models = build_selected_models(random_state)
    weights = cv_weights(models, x_train, y_train, random_state)
    for model in models.values():
        model.fit(x_train, y_train)

    y_score = weighted_predict_proba(models, weights, x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    metrics = binary_metrics(y_test, y_pred, y_score)

    pd.DataFrame([metrics]).to_csv(out / "cv_whsve_metrics.csv", index=False)
    pd.DataFrame({"classifier": list(weights), "weight": list(weights.values())}).to_csv(out / "cv_weights.csv", index=False)
    pd.Series(feature_names, name="retained_feature").to_csv(out / "retained_features.csv", index=False)

    try:
        import shap

        background = shap.sample(x_train, min(100, len(x_train)), random_state=random_state)
        explainer = shap.KernelExplainer(lambda z: weighted_predict_proba(models, weights, z)[:, 1], background)
        sample = x_test[: min(100, len(x_test))]
        shap_values = explainer.shap_values(sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).to_csv(
            out / "shap_feature_importance.csv", index=False
        )
    except Exception as exc:
        print(f"SHAP analysis skipped: {exc}")

    print("CV-WHSVE metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Cross-validation weights: {weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to local CSV. Defaults to PLR_DATA_CSV or data/preprocessed_data.csv.")
    parser.add_argument("--out", default="outputs", help="Directory for generated outputs. Ignored by .gitignore.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(args.data, args.out, args.random_state)
