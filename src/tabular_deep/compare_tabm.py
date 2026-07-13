#!/usr/bin/env python
# coding: utf-8

import numpy as np

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
    seed = get_seed()
    np.random.seed(seed)
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)
    X_train, X_test, _, _ = preprocess_numeric(X_train, X_test)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import tabm
    except Exception as e:
        raise RuntimeError("TabM not installed. Please run: pip install tabm torch") from e

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    device = torch.device("cpu")

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Support both tabm APIs:
    # 1) `from tabm import Model` with `Model.make_baseline(...)`
    # 2) `from tabm import TabM` constructor-style API
    model = None
    if hasattr(tabm, "Model"):
        Model = tabm.Model
        model = Model.make_baseline(
            n_num_features=X_train.shape[1],
            cat_cardinalities=[],
            n_classes=None,
            backbone={"type": "MLP", "n_blocks": 3, "d_block": 256, "dropout": 0.1},
            k=8,
        ).to(device)
    elif hasattr(tabm, "TabM"):
        TabM = tabm.TabM
        # constructor signature differs by version; use conservative args
        model = TabM(
            n_num_features=X_train.shape[1],
            cat_cardinalities=[],
            d_out=1,
            k=8,
            n_blocks=3,
            d_block=256,
            dropout=0.1,
            start_scaling_init="normal",
        ).to(device)
    else:
        raise RuntimeError("Unsupported tabm API: neither `Model` nor `TabM` found.")

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    batch_size = 256
    model.train()
    for _ in range(40):
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]

            opt.zero_grad()
            try:
                out = model(x_num=xb, x_cat=None)
            except TypeError:
                out = model(xb)
            if out.ndim == 3:
                # [B, K, 1] -> expand target to [B, K, 1]
                yb = yb.unsqueeze(1).expand(-1, out.shape[1], -1)
            elif out.ndim == 2:
                yb = yb.expand(-1, out.shape[1])
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        try:
            out = model(x_num=Xte, x_cat=None)
        except TypeError:
            out = model(Xte)
        if out.ndim == 3:
            # [B, K, 1] -> average over ensemble dimension
            y_proba = torch.sigmoid(out).mean(dim=1).reshape(-1).cpu().numpy()
        elif out.ndim == 2:
            y_proba = torch.sigmoid(out).mean(dim=1).cpu().numpy()
        else:
            y_proba = torch.sigmoid(out).reshape(-1).cpu().numpy()

    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test, y_pred, y_proba)
    print("Method: TabM")
    print(f"Seed: {seed}")
    print_metrics(metrics)
    save_outputs("TabM", seed, metrics, y_test, y_pred, y_proba)


if __name__ == "__main__":
    main()

