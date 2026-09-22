"""Leakage-aware nested validation for the CV-WHSVE ensemble.

The outer folds estimate ensemble performance and model weights. The inner
folds generate out-of-fold probabilities used by the genetic algorithm. All
fitted preprocessing steps are contained in each classifier pipeline, so the
imputer, zero-variance filter, and (where applicable) scaler see training-fold
data only.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

# Keep numerical kernels single-threaded inside each cross-validation worker.
# The five folds are parallelised explicitly below; nested BLAS/OpenMP threads
# otherwise introduce avoidable run-to-run variation and oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "12")

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.cv_whsve.ga_selection import select_classifier_subset
from src.cv_whsve.model_pool import build_candidate_models
from src.utils.data import load_plr_table
from src.utils.features import extract_plr_features
from src.utils.metrics import binary_metrics


DEFAULT_GA_OPTIONS = {
    "population_size": 100,
    "max_generations": 100,
    "crossover_probability": 0.8,
    "mutation_probability": 0.3,
    "early_stop_generations": 5,
    "minimum_relative_improvement": 0.001,
}

_FEATURE_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}


def _load_feature_table(data: str | None):
    """Load and featurize once per process for repeated-split analyses."""
    source = data or os.getenv("PLR_DATA_CSV", "data/preprocessed_data.csv")
    path = Path(source).expanduser().resolve()
    stamp = path.stat().st_mtime_ns if path.exists() else 0
    key = f"{path}|{stamp}"
    if key not in _FEATURE_CACHE:
        sequences, labels, row_ids = load_plr_table(data)
        features, feature_names = extract_plr_features(sequences)
        _FEATURE_CACHE[key] = (features, labels, row_ids, feature_names)
    return _FEATURE_CACHE[key]


def _normalise_weights(scores: dict[str, float]) -> dict[str, float]:
    clean = {name: max(float(value), 0.0) for name, value in scores.items()}
    total = sum(clean.values())
    if not clean:
        raise ValueError("No model scores were supplied for weighting")
    if total <= 0:
        equal = 1.0 / len(clean)
        return {name: equal for name in clean}
    return {name: value / total for name, value in clean.items()}


def _weighted_average(
    probability_by_model: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    return sum(weights[name] * probability_by_model[name] for name in weights)


def _inner_oof_probabilities(
    models: dict,
    x: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    probabilities: dict[str, np.ndarray] = {}
    failures: list[dict[str, str]] = []
    for name, model in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Thread-based fold parallelism keeps all estimators in the
                # same CPU-only process.  It avoids the CUDA reinitialisation
                # seen with process workers while preserving deterministic
                # fold ordering.
                with parallel_backend("threading", n_jobs=cv.n_splits):
                    probabilities[name] = cross_val_predict(
                        clone(model),
                        x,
                        y,
                        cv=cv,
                        method="predict_proba",
                        n_jobs=cv.n_splits,
                    )[:, 1]
        except Exception as exc:  # retain an auditable failure record
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
    if not probabilities:
        raise RuntimeError("All candidate classifiers failed during inner cross-validation")
    return probabilities, failures


def _youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    objective = (tpr - fpr)[finite]
    return float(thresholds[finite][int(np.argmax(objective))])


def _save_calibration_curve(y: np.ndarray, probability: np.ndarray, path: Path) -> None:
    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    pd.DataFrame(
        {"mean_predicted_probability": predicted, "observed_positive_fraction": observed}
    ).to_csv(path, index=False)


def _document_training_features(
    x_train: np.ndarray, feature_names: list[str], out: Path
) -> list[str]:
    """Document the features retained when the filter is fitted on all training data.

    This fit is for reporting only. Predictive evaluation uses separate copies
    of the same transformations fitted inside their respective folds.
    """
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(x_train)
    selector = VarianceThreshold(threshold=0.0).fit(imputed)
    retained = [name for name, keep in zip(feature_names, selector.get_support()) if keep]
    pd.DataFrame(
        {
            "feature": feature_names,
            "retained_on_full_training_partition": selector.get_support().astype(int),
        }
    ).to_csv(out / "training_feature_filter.csv", index=False)
    return retained


def run_nested_evaluation(
    data: str | None,
    out_dir: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.20,
    outer_splits: int = 5,
    inner_splits: int = 5,
    profile: str = "full",
    final_model_count: int = 5,
    ga_options: dict | None = None,
) -> dict:
    """Run one participant-level held-out evaluation with nested model selection."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ga_config = {**DEFAULT_GA_OPTIONS, **(ga_options or {})}

    all_features, labels, row_ids, feature_names = _load_feature_table(data)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    x_train, y_train = all_features[train_idx], labels[train_idx]
    x_test, y_test = all_features[test_idx], labels[test_idx]
    retained_features = _document_training_features(x_train, feature_names, out)

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    nested_oof = np.full(len(y_train), np.nan, dtype=float)
    candidate_names = list(build_candidate_models(random_state, profile=profile))
    outer_oof_by_model = {
        name: np.full(len(y_train), np.nan, dtype=float) for name in candidate_names
    }
    fold_rows: list[dict] = []
    model_rows: list[dict] = []
    failure_rows: list[dict] = []

    for fold, (outer_train, outer_valid) in enumerate(outer_cv.split(x_train, y_train), start=1):
        fold_seed = random_state + fold
        candidates = build_candidate_models(fold_seed, profile=profile)
        inner_cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=fold_seed
        )
        inner_probabilities, failures = _inner_oof_probabilities(
            candidates, x_train[outer_train], y_train[outer_train], inner_cv
        )
        for failure in failures:
            failure_rows.append({"outer_fold": fold, **failure})
        if failures and profile == "full":
            details = "; ".join(
                f"{failure['model']}: {failure['error']}" for failure in failures
            )
            raise RuntimeError(
                f"Candidate evaluation failed in outer fold {fold}; "
                f"scientific runs require the complete 17-model pool. {details}"
            )

        ga_result = select_classifier_subset(
            inner_probabilities,
            y_train[outer_train],
            random_state=fold_seed,
            **ga_config,
        )
        inner_accuracy = {
            name: accuracy_score(
                y_train[outer_train], (inner_probabilities[name] >= 0.5).astype(int)
            )
            for name in ga_result.selected_names
        }
        fold_weights = _normalise_weights(inner_accuracy)

        outer_probabilities: dict[str, np.ndarray] = {}
        for name in inner_probabilities:
            try:
                fitted = clone(candidates[name]).fit(
                    x_train[outer_train], y_train[outer_train]
                )
                probability = fitted.predict_proba(x_train[outer_valid])[:, 1]
            except Exception as exc:
                failure_rows.append(
                    {
                        "outer_fold": fold,
                        "model": name,
                        "error": f"outer fit: {type(exc).__name__}: {exc}",
                    }
                )
                if profile == "full" or name in ga_result.selected_names:
                    raise RuntimeError(
                        f"Classifier {name} failed in outer fold {fold}; "
                        "scientific runs require the complete candidate pool"
                    ) from exc
                continue
            outer_oof_by_model[name][outer_valid] = probability
            if name in ga_result.selected_names:
                outer_probabilities[name] = probability
            model_rows.append(
                {
                    "outer_fold": fold,
                    "model": name,
                    "selected_by_ga": int(name in ga_result.selected_names),
                    "inner_oof_accuracy": accuracy_score(
                        y_train[outer_train],
                        (inner_probabilities[name] >= 0.5).astype(int),
                    ),
                    "fold_weight": fold_weights.get(name, 0.0),
                    "outer_accuracy": accuracy_score(
                        y_train[outer_valid], (probability >= 0.5).astype(int)
                    ),
                }
            )

        fold_score = _weighted_average(outer_probabilities, fold_weights)
        nested_oof[outer_valid] = fold_score
        fold_metrics = binary_metrics(
            y_train[outer_valid], (fold_score >= 0.5).astype(int), fold_score
        )
        fold_rows.append(
            {
                "outer_fold": fold,
                "ga_fitness": ga_result.best_fitness,
                "ga_generations": ga_result.generations_run,
                "selected_models": ";".join(ga_result.selected_names),
                **fold_metrics,
            }
        )

    if np.isnan(nested_oof).any():
        raise RuntimeError("Nested out-of-fold predictions are incomplete")

    model_table = pd.DataFrame(model_rows)
    ranking = (
        model_table.groupby("model", as_index=False)
        .agg(
            selection_count=("selected_by_ga", "sum"),
            folds_evaluated=("outer_fold", "size"),
            mean_outer_accuracy=("outer_accuracy", "mean"),
            mean_inner_oof_accuracy=("inner_oof_accuracy", "mean"),
        )
        .sort_values(
            ["selection_count", "mean_outer_accuracy", "model"],
            ascending=[False, False, True],
        )
    )
    eligible_ranking = ranking.loc[ranking["folds_evaluated"] == outer_splits]
    number_to_select = min(final_model_count, len(eligible_ranking))
    if number_to_select == 0:
        raise RuntimeError("No classifier completed every outer validation fold")
    final_names = eligible_ranking.head(number_to_select)["model"].tolist()
    accuracy_map = ranking.set_index("model")["mean_outer_accuracy"].to_dict()
    final_weights = _normalise_weights({name: accuracy_map[name] for name in final_names})

    calibration_oof = _weighted_average(
        {name: outer_oof_by_model[name] for name in final_names}, final_weights
    )
    calibrator = LogisticRegression(solver="lbfgs", random_state=random_state)
    calibrator.fit(calibration_oof.reshape(-1, 1), y_train)
    calibrated_oof = calibrator.predict_proba(calibration_oof.reshape(-1, 1))[:, 1]
    selected_threshold = _youden_threshold(y_train, calibrated_oof)

    final_candidates = build_candidate_models(random_state, profile=profile)
    test_probability_by_model: dict[str, np.ndarray] = {}
    for name in final_names:
        fitted = clone(final_candidates[name]).fit(x_train, y_train)
        test_probability_by_model[name] = fitted.predict_proba(x_test)[:, 1]
    test_raw = _weighted_average(test_probability_by_model, final_weights)
    test_calibrated = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    default_metrics = binary_metrics(
        y_test, (test_calibrated >= 0.5).astype(int), test_calibrated
    )
    threshold_metrics = binary_metrics(
        y_test,
        (test_calibrated >= selected_threshold).astype(int),
        test_calibrated,
    )
    metric_rows = [
        {"decision_rule": "calibrated_probability_0.5", "threshold": 0.5, **default_metrics},
        {
            "decision_rule": "training_oof_youden_threshold",
            "threshold": selected_threshold,
            **threshold_metrics,
        },
    ]

    pd.DataFrame(fold_rows).to_csv(out / "nested_fold_metrics.csv", index=False)
    model_table.to_csv(out / "nested_model_fold_details.csv", index=False)
    ranking.to_csv(out / "selection_frequency.csv", index=False)
    (
        model_table.loc[model_table["selected_by_ga"] == 1]
        .groupby("model", as_index=False)
        .agg(
            folds_selected=("outer_fold", "size"),
            mean_fold_weight=("fold_weight", "mean"),
            sd_fold_weight=("fold_weight", "std"),
        )
        .fillna({"sd_fold_weight": 0.0})
        .to_csv(out / "weight_stability.csv", index=False)
    )
    pd.DataFrame(
        {"model": final_names, "nested_outer_accuracy_weight": [final_weights[n] for n in final_names]}
    ).to_csv(out / "nested_weights.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(failure_rows, columns=["outer_fold", "model", "error"]).to_csv(
        out / "candidate_failures.csv", index=False
    )
    pd.DataFrame(
        {
            "row_id": row_ids[test_idx],
            "y_true": y_test,
            "raw_ensemble_probability": test_raw,
            "calibrated_probability": test_calibrated,
            "prediction_at_0.5": (test_calibrated >= 0.5).astype(int),
            "prediction_at_selected_threshold": (
                test_calibrated >= selected_threshold
            ).astype(int),
        }
    ).to_csv(out / "test_predictions.csv", index=False)
    _save_calibration_curve(y_train, calibrated_oof, out / "calibration_curve.csv")

    metadata = {
        "random_state": random_state,
        "n_total": int(len(labels)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "candidate_profile": profile,
        "n_predefined_features": len(feature_names),
        "n_retained_on_full_training_partition": len(retained_features),
        "retained_features": retained_features,
        "final_models": final_names,
        "final_weights": final_weights,
        "selected_threshold": selected_threshold,
        "ga_options": ga_config,
        "note": (
            "The feature-filter fit reported here uses the full training partition only. "
            "During validation, preprocessing was refitted separately inside every fold."
        ),
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "random_state": random_state,
        "final_models": final_names,
        "weights": final_weights,
        "selected_threshold": selected_threshold,
        "default_metrics": default_metrics,
        "threshold_metrics": threshold_metrics,
        "output_directory": str(out),
    }
