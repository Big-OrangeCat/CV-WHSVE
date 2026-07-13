"""Train the Micro-BiLSTM baseline."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data import load_plr_csv, participant_split


class MicroBiLSTM(nn.Module):
    def __init__(self, input_size: int = 125, hidden: int = 32, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = x.squeeze(1).unsqueeze(1)
        x, _ = self.lstm(x)
        return self.fc(x.mean(dim=1))


def train(data: str | None, epochs: int = 80):
    x, y = load_plr_csv(data)
    x_train, x_test, y_train, y_test = participant_split(x, y, random_state=42)
    x_train = x_train[:, None, :].astype("float32")
    x_test = x_test[:, None, :].astype("float32")
    mean, std = x_train.mean(), x_train.std() + 1e-8
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=64, shuffle=True)
    model = MicroBiLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
        proba = torch.softmax(model(torch.from_numpy(x_test)), dim=1)[:, 1].numpy()
    latency_ms = (time.perf_counter() - start) * 1000 / len(x_test)
    print({"auroc": roc_auc_score(y_test, proba), "cpu_latency_ms": latency_ms})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    train(args.data, args.epochs)
