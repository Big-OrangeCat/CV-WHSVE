#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPTS = [
    "compare_tabtransformer.py",
    "compare_tabr.py",
    "compare_tabm.py",
]

PATTERNS = {
    "accuracy": r"Accuracy\s*:\s*([0-9]*\.?[0-9]+)",
    "auc": r"ROC AUC\s*:\s*([0-9]*\.?[0-9]+)",
    "f1": r"F1-score\s*:\s*([0-9]*\.?[0-9]+)",
    "precision": r"Precision\s*:\s*([0-9]*\.?[0-9]+)",
    "recall": r"Recall\s*:\s*([0-9]*\.?[0-9]+)",
    "specificity": r"Specificity\s*:\s*([0-9]*\.?[0-9]+)",
}


def parse_metrics(text: str):
    out = {}
    for k, p in PATTERNS.items():
        m = re.findall(p, text, flags=re.IGNORECASE)
        if m:
            out[k] = float(m[-1])
    return out


def summarize(rows: list[dict]):
    keys = sorted({k for r in rows for k in r.keys()})
    data = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        data[k] = {
            "runs": [float(v) for v in arr.tolist()],
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=1412)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    logs = base / "repro_logs"
    logs.mkdir(exist_ok=True)

    final = {}
    for script in SCRIPTS:
        sp = base / script
        if not sp.exists():
            print(f"[WARN] Missing: {script}")
            continue

        print(f"\n===== {script} =====")
        per_runs = []
        for i in range(args.runs):
            seed = args.seed_start + i
            print(f"[RUN {i+1}/{args.runs}] seed={seed}")
            env = os.environ.copy()
            env["REPRO_SEED"] = str(seed)
            env["PYTHONHASHSEED"] = str(seed)
            env["REPRO_RUN_INDEX"] = str(i + 1)
            env["REPRO_RUNS_TOTAL"] = str(args.runs)

            proc = subprocess.run(
                [args.python, str(sp)],
                cwd=str(base),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                env=env,
            )
            log_file = logs / f"{sp.stem}.run{i+1}.log"
            with open(log_file, "w", encoding="utf-8", errors="ignore") as f:
                f.write(proc.stdout or "")
                f.write("\n\n=== STDERR ===\n")
                f.write(proc.stderr or "")

            if proc.returncode != 0:
                print(f"[ERROR] exit={proc.returncode}, log={log_file}")
                continue

            metrics = parse_metrics(proc.stdout or "")
            per_runs.append(metrics)
            if metrics:
                print("  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            else:
                print(f"  [WARN] no metrics parsed, log={log_file}")

        summ = summarize(per_runs)
        final[script] = summ
        if summ:
            print("  => mean卤std")
            for k, v in summ.items():
                print(f"     {k}: {v['mean']:.4f} 卤 {v['std']:.4f} (runs={v['runs']})")

    out = base / "compare_3x_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary: {out}")


if __name__ == "__main__":
    main()

