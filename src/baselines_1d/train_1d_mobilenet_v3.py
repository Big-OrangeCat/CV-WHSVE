"""Train the 1D-MobileNet-V3 baseline on 125-frame PLR sequences."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data import load_plr_csv, participant_split


BATCH, EPOCHS, LR, WD = 64, 80, 3e-4, 1e-4


class SE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = max(1, c // r)
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, hidden), nn.ReLU(), nn.Linear(hidden, c), nn.Hardsigmoid())

    def forward(self, x):
        return x * self.net(x).view(x.size(0), x.size(1), 1)


class DW(nn.Module):
    def __init__(self, c: int, exp: int = 2):
        super().__init__()
        hidden = int(c * exp)
        self.net = nn.Sequential(
            nn.Conv1d(c, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.Hardswish(),
            nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.Hardswish(),
            SE(hidden),
            nn.Conv1d(hidden, c, 1, bias=False),
            nn.BatchNorm1d(c),
        )

    def forward(self, x):
        return x + self.net(x)


class MobileNet1D(nn.Module):
    def __init__(self, num_classes: int = 2, width: float = 0.125):
        super().__init__()
        c1, c2 = max(8, int(64 * width)), max(8, int(128 * width))
        self.stem = nn.Sequential(nn.Conv1d(1, c1, 3, padding=1, bias=False), nn.BatchNorm1d(c1), nn.Hardswish())
        self.blocks = nn.Sequential(DW(c1), nn.Conv1d(c1, c2, 1), nn.BatchNorm1d(c2), nn.Hardswish(), DW(c2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c2, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def train(data: str | None, epochs: int = EPOCHS):
    x, y = load_plr_csv(data)
    x_train, x_test, y_train, y_test = participant_split(x, y, random_state=42)
    x_train = x_train[:, None, :].astype("float32")
    x_test = x_test[:, None, :].astype("float32")
    mean, std = x_train.mean(), x_train.std() + 1e-8
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=BATCH, shuffle=True)
    model = MobileNet1D()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test))
        proba = torch.softmax(logits, dim=1)[:, 1].numpy()
    latency_ms = (time.perf_counter() - start) * 1000 / len(x_test)
    print({"auroc": roc_auc_score(y_test, proba), "cpu_latency_ms": latency_ms})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(args.data, args.epochs)
