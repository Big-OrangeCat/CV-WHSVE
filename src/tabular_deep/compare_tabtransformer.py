#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd

from compare_utils import (
    evaluate_binary,
    get_seed,
    load_data,
    preprocess_numeric,
    print_metrics,
    save_outputs,
    split_data,
)


def main():
    # Windows console may default to GBK; avoid rich progress UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    seed = get_seed()
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)
    X_train, X_test, _, _ = preprocess_numeric(X_train, X_test)

    try:
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        from pytorch_tabular.models import TabTransformerConfig
    except Exception as e:
        raise RuntimeError("TabTransformer backend not found. Please install: pip install 'pytorch-tabular[extra]'") from e

    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    train_df["label"] = y_train
    test_df["label"] = y_test

    data_config = DataConfig(
        target=["label"],
        continuous_cols=feature_names,
        categorical_cols=[],
    )
    model_config = TabTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        metrics=["accuracy"],
        metrics_prob_input=[False],
        seed=seed,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=256,
        max_epochs=40,
        accelerator="cpu",
        deterministic=True,
        progress_bar="none",
        seed=seed,
    )
    optimizer_config = OptimizerConfig()

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    model.fit(train=train_df)
    pred_df = model.predict(test_df)

    if "1_probability" in pred_df.columns:
        y_proba = pred_df["1_probability"].to_numpy()
    elif "label_1_probability" in pred_df.columns:
        y_proba = pred_df["label_1_probability"].to_numpy()
    elif "0_probability" in pred_df.columns:
        y_proba = 1.0 - pred_df["0_probability"].to_numpy()
    elif "label_0_probability" in pred_df.columns:
        y_proba = 1.0 - pred_df["label_0_probability"].to_numpy()
    elif "prediction_probability" in pred_df.columns and "prediction" in pred_df.columns:
        p = pred_df["prediction_probability"].to_numpy()
        yhat = pred_df["prediction"].to_numpy().astype(int)
        y_proba = (yhat == 1) * p + (yhat == 0) * (1.0 - p)
    elif "label_prediction" in pred_df.columns and "label_0_probability" in pred_df.columns:
        # fallback if only class-0 probability is exposed together with predicted label
        y_proba = 1.0 - pred_df["label_0_probability"].to_numpy()
    else:
        raise RuntimeError(f"Cannot infer positive-class probability from columns: {list(pred_df.columns)}")

    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test, y_pred, y_proba)
    print("Method: TabTransformer")
    print(f"Seed: {seed}")
    print_metrics(metrics)
    save_outputs("TabTransformer", seed, metrics, y_test, y_pred, y_proba)


if __name__ == "__main__":
    main()

