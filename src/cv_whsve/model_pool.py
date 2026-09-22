"""Leakage-aware candidate model definitions for CV-WHSVE.

All transformations that estimate parameters from data are inside an sklearn
Pipeline. During cross-validation, SimpleImputer, VarianceThreshold, and
StandardScaler are therefore fitted on the corresponding training fold only.
"""

from __future__ import annotations

import os

# The manuscript analyses are CPU analyses.  Setting this before importing the
# optional gradient-boosting libraries prevents a CUDA-enabled wheel from
# selecting a GPU implicitly on machines where CUDA is installed but not
# available to the current process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


MANUSCRIPT_SELECTED = [
    "GaussianProcess",
    "ExtraTrees",
    "CatBoost",
    "GaussianNB",
    "AdaBoost",
]


def _pipeline(estimator, *, scale: bool = False) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("zero_variance", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def build_candidate_models(random_state: int = 42, profile: str = "full") -> dict[str, Pipeline]:
    """Return the 17 candidate classifiers described in the manuscript.

    ``profile='smoke'`` keeps the five manuscript-selected learners and is
    intended only for executable synthetic-data checks. Scientific analyses
    must use ``profile='full'``.
    """
    models = {
        "CatBoost": _pipeline(
            CatBoostClassifier(
                depth=6,
                learning_rate=0.1,
                loss_function="Logloss",
                random_seed=random_state,
                verbose=0,
                allow_writing_files=False,
                thread_count=1,
            )
        ),
        "LightGBM": _pipeline(
            LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            )
        ),
        "XGBoost": _pipeline(
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
                tree_method="hist",
                device="cpu",
                predictor="cpu_predictor",
            )
        ),
        "RandomForest": _pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=random_state,
                n_jobs=1,
            )
        ),
        "AdaBoost": _pipeline(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state,
            )
        ),
        "GradientBoosting": _pipeline(
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=random_state)
        ),
        "ExtraTrees": _pipeline(
            ExtraTreesClassifier(
                n_estimators=100,
                max_depth=15,
                criterion="gini",
                random_state=random_state,
                n_jobs=1,
            )
        ),
        "HistGradientBoosting": _pipeline(
            HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=random_state)
        ),
        "BaggingDT": _pipeline(
            BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=random_state),
                n_estimators=50,
                random_state=random_state,
                n_jobs=1,
            )
        ),
        "SVC": _pipeline(SVC(kernel="rbf", probability=True, random_state=random_state), scale=True),
        "KernelSVM": _pipeline(SVC(kernel="poly", probability=True, random_state=random_state), scale=True),
        "QDA": _pipeline(QuadraticDiscriminantAnalysis(), scale=True),
        "GaussianProcess": _pipeline(
            GaussianProcessClassifier(
                kernel=1.0 * RBF(length_scale=1.0),
                optimizer="fmin_l_bfgs_b",
                random_state=random_state,
            ),
            scale=True,
        ),
        "GaussianNB": _pipeline(GaussianNB(var_smoothing=1e-9)),
        "KNN": _pipeline(KNeighborsClassifier(n_neighbors=29), scale=True),
        "SGD": _pipeline(
            SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4, random_state=random_state),
            scale=True,
        ),
        "MLP": _pipeline(
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state),
            scale=True,
        ),
    }
    if profile == "full":
        return models
    if profile == "smoke":
        return {name: models[name] for name in MANUSCRIPT_SELECTED}
    raise ValueError("profile must be 'full' or 'smoke'")


def build_selected_models(random_state: int = 42) -> dict[str, Pipeline]:
    models = build_candidate_models(random_state, profile="full")
    return {name: models[name] for name in MANUSCRIPT_SELECTED}
