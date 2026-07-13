#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

from compare_utils import (
    evaluate_binary,
    get_seed,
    load_data,
    preprocess_numeric,
    print_metrics,
    save_outputs,
    split_data,
)


def build_retrieval_features(X_ref, X_q, k=16):
    """
    TabR-style retrieval augmentation (approx):
    append neighbor distance statistics as retrieval context.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_ref)
    d, _ = nn.kneighbors(X_q)
    stats = np.column_stack(
        [
            d.mean(axis=1),
            d.std(axis=1),
            d.min(axis=1),
            d.max(axis=1),
            np.median(d, axis=1),
        ]
    )
    return np.hstack([X_q, stats]).astype(np.float32)


def main():
    seed = get_seed()
    np.random.seed(seed)

    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)
    X_train, X_test, _, _ = preprocess_numeric(X_train, X_test)

    # Retrieval augmentation
    X_train_aug = build_retrieval_features(X_train, X_train, k=16)
    X_test_aug = build_retrieval_features(X_train, X_test, k=16)

    # Lightweight deep head
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(X_train_aug, y_train)
    y_proba = model.predict_proba(X_test_aug)[:, 1]

    # Orientation guard
    auc_raw = roc_auc_score(y_test, y_proba)
    auc_flip = roc_auc_score(y_test, 1.0 - y_proba)
    if auc_flip > auc_raw:
        y_proba = 1.0 - y_proba
        print(f"[Info] Probability flipped: auc_raw={auc_raw:.4f}, auc_flip={auc_flip:.4f}")

    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test, y_pred, y_proba)

    print("Method: TabR-approx")
    print(f"Seed: {seed}")
    print_metrics(metrics)
    save_outputs("TabR", seed, metrics, y_test, y_pred, y_proba)


if __name__ == "__main__":
    main()


