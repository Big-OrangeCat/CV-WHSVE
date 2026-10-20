"""Plot aggregate development-only learning curves (no participant data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.metrics)
    names = ["1D-MobileNet-V3", "TSM-1D", "Micro-BiLSTM"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.6), sharex="col")
    for column, name in enumerate(names):
        frame = data.loc[data["model"] == name].sort_values("epoch")
        upper, lower = axes[:, column]
        upper.plot(frame.epoch, frame.training_loss, color="#2471A3", lw=1.5,
                   label="Training loss")
        upper.plot(frame.epoch, frame.validation_loss, color="#D35400", lw=1.5,
                   label="Validation loss")
        lower.plot(frame.epoch, frame.validation_auroc * 100, color="#397D54",
                   lw=1.5, label="Validation AUROC")
        for axis in (upper, lower):
            axis.axvline(80, color="#5E6170", lw=1, ls="--")
            axis.grid(color="#E5E7EB", linewidth=0.5)
            axis.set_xlim(1, 240)
            axis.spines[["top", "right"]].set_visible(False)
        upper.set_title(name, weight="bold")
        lower.set_xlabel("Epoch")
        lower.set_ylim(50, 95)
        if column == 0:
            upper.set_ylabel("Cross-entropy loss")
            lower.set_ylabel("Validation AUROC (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper left",
               bbox_to_anchor=(0.015, 0.995), ncol=2)
    fig.text(0.99, 0.98, "Dashed line: 80-epoch comparison",
             ha="right", va="top", fontsize=8, color="#5E6170")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
