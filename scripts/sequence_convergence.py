"""Retrospective development-only convergence diagnostic for three 1D baselines.

The held-out 20% test partition is excluded entirely. No participant data,
checkpoint, or prediction is written; only per-epoch aggregate metrics are saved.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    labels, probabilities = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * len(yb)
            total_count += len(yb)
            labels.extend(yb.cpu().numpy().tolist())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return total_loss / total_count, roc_auc_score(labels, probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.utils.data import load_plr_csv
    from src.baselines_1d.train_1d_mobilenet_v3 import MobileNet1D
    from src.baselines_1d.train_micro_bilstm import MicroBiLSTM
    from src.baselines_1d.train_tsm_1d import TSM1DNet

    x, y = load_plr_csv(args.data)
    indices = np.arange(len(y))
    development, held_out = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42
    )
    train_idx, validation_idx = train_test_split(
        development, test_size=0.2, stratify=y[development], random_state=42
    )
    # The held-out indices are counted, but never used for normalization or training.
    print(f"development={len(development)} train={len(train_idx)} "
          f"validation={len(validation_idx)} held_out_excluded={len(held_out)}", flush=True)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x_train = ((x[train_idx] - mean) / std)[:, None, :].astype(np.float32)
    x_validation = ((x[validation_idx] - mean) / std)[:, None, :].astype(np.float32)
    y_train = y[train_idx].astype(np.int64)
    y_validation = y[validation_idx].astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "model", "epoch", "training_loss", "validation_loss", "validation_auroc",
        ])
        writer.writeheader()
        for name, constructor, smoothing in [
            ("1D-MobileNet-V3", MobileNet1D, 0.05),
            ("TSM-1D", TSM1DNet, 0.0),
            ("Micro-BiLSTM", MicroBiLSTM, 0.05),
        ]:
            set_seed(args.seed)
            model = constructor().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                batch_size=64, shuffle=True,
                generator=torch.Generator().manual_seed(args.seed),
            )
            validation_loader = DataLoader(
                TensorDataset(torch.from_numpy(x_validation), torch.from_numpy(y_validation)),
                batch_size=64, shuffle=False,
            )
            for epoch in range(1, args.epochs + 1):
                model.train()
                loss_sum = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.item()) * len(yb)
                validation_loss, validation_auroc = evaluate(
                    model, validation_loader, criterion, device
                )
                writer.writerow({
                    "model": name,
                    "epoch": epoch,
                    "training_loss": loss_sum / len(train_idx),
                    "validation_loss": validation_loss,
                    "validation_auroc": validation_auroc,
                })
                file.flush()
                if epoch == 1 or epoch % 10 == 0:
                    print(f"{name} epoch={epoch:03d} "
                          f"train_loss={loss_sum / len(train_idx):.4f} "
                          f"validation_loss={validation_loss:.4f} "
                          f"validation_auroc={validation_auroc:.4f}", flush=True)


if __name__ == "__main__":
    main()
