"""Train the TSM-1D baseline for 125-frame PLR sequences.

This public script contains the model and training settings needed to reproduce
the TSM-1D baseline structure reported in the manuscript. It intentionally does
not include participant data, trained weights, or generated result files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data import load_sequences_and_labels, stratified_train_test


class TemporalShift1D(nn.Module):
    """A compact temporal-shift block for one-dimensional PLR sequences."""

    def __init__(self, channels: int, fold_div: int = 8) -> None:
        super().__init__()
        self.fold_div = fold_div
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        fold = max(1, self.channels // self.fold_div)
        out = torch.zeros_like(x)
        out[:, :fold, :-1] = x[:, :fold, 1:]
        out[:, fold : 2 * fold, 1:] = x[:, fold : 2 * fold, :-1]
        out[:, 2 * fold :, :] = x[:, 2 * fold :, :]
        return out


class TSMBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.block = nn.Sequential(
            TemporalShift1D(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.proj(x))


class TSM1DNet(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            TSMBlock1D(32, 64, stride=2),
            TSMBlock1D(64, 128, stride=2),
            TSMBlock1D(128, 128, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def _make_loaders(data_path: str | None, batch_size: int = 64):
    sequences, labels = load_sequences_and_labels(data_path)
    x_train, x_test, y_train, y_test = stratified_train_test(sequences, labels, test_size=0.2, random_state=42)
    x_train = torch.tensor(x_train[:, None, :], dtype=torch.float32)
    x_test = torch.tensor(x_test[:, None, :], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(data_path: str | None, epochs: int = 80, output_dir: str = "outputs/baselines_1d") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = _make_loaders(data_path)
    model = TSM1DNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch={epoch:03d} loss={np.mean(losses):.4f}")

    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            y_score.extend(prob.tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_score),
    }
    print(metrics)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(output_dir) / "tsm_1d_state_dict.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="CSV path, or set PLR_DATA_CSV.")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    train(args.data, args.epochs)
