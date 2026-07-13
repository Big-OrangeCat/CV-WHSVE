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


SCRIPT_GROUP = [
    "stacking-鐬冲瓟13.1.py",
    "stacking-鐬冲瓟15.0锛堢儹鍔涘浘锛?py",
    "stacking-鐬冲瓟15.1锛堟秷铻嶏級.py",
    "stacking-鐬冲瓟15.2锛堝厓鍒嗙被瀵规瘮锛?py",
    #"train_1d_cnn.py",
    #"train_micro_bilstm.py",
    #"train_tsm_1d.py",
]


METRIC_PATTERNS = {
    "accuracy": [r"Accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)"],
    "auc": [
        r"ROC AUC\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bAUC\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"AUROC\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "f1": [
        r"F1-score\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bF1\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "precision": [r"Precision\s*[:=]\s*([0-9]*\.?[0-9]+)"],
    "recall": [r"Recall\s*[:=]\s*([0-9]*\.?[0-9]+)"],
    "specificity": [r"Specificity\s*[:=]\s*([0-9]*\.?[0-9]+)"],
    "best_auc": [r"Best AUC\s*[:=]\s*([0-9]*\.?[0-9]+)"],
}


def _extract_last_float(text: str, patterns: list[str]):
    last_val = None
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            try:
                last_val = float(matches[-1])
            except ValueError:
                pass
    return last_val


def parse_metrics(stdout_text: str):
    metrics = {}
    for name, patterns in METRIC_PATTERNS.items():
        val = _extract_last_float(stdout_text, patterns)
        if val is not None:
            metrics[name] = val
    return metrics


def summarize(metric_runs: list[dict]):
    all_keys = sorted({k for row in metric_runs for k in row.keys()})
    out = {}
    for k in all_keys:
        vals = [row[k] for row in metric_runs if k in row]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
            "runs": [float(x) for x in arr.tolist()],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Run full 15.0 suite for 3x reproduction and report mean卤std.")
    parser.add_argument("--runs", type=int, default=3, help="repeat count, default=3")
    parser.add_argument("--python", default=sys.executable, help="python executable path")
    parser.add_argument("--keep-logs", action="store_true", help="keep full per-run stdout logs")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / "repro_logs"
    logs_dir.mkdir(exist_ok=True)

    summary = {}
    for script in SCRIPT_GROUP:
        script_path = base_dir / script
        if not script_path.exists():
            print(f"[WARN] Missing script: {script}")
            continue

        print(f"\n===== {script} =====")
        metric_runs = []
        for i in range(1, args.runs + 1):
            print(f"[RUN {i}/{args.runs}] {script}")
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(1411 + i)
            env["REPRO_SEED"] = str(1411 + i)
            env["REPRO_RUN_INDEX"] = str(i)
            env["REPRO_RUNS_TOTAL"] = str(args.runs)

            proc = subprocess.run(
                [args.python, str(script_path)],
                cwd=str(base_dir),
                text=True,
                capture_output=True,
                env=env,
            )

            log_file = logs_dir / f"{script_path.stem}.run{i}.log"
            with open(log_file, "w", encoding="utf-8", errors="ignore") as f:
                f.write(proc.stdout or "")
                f.write("\n\n=== STDERR ===\n")
                f.write(proc.stderr or "")

            if proc.returncode != 0:
                print(f"[ERROR] exit_code={proc.returncode}, see: {log_file}")
                continue

            metrics = parse_metrics(proc.stdout or "")
            if not metrics:
                print(f"[WARN] no metric parsed from stdout, see: {log_file}")
            else:
                print("  parsed:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            metric_runs.append(metrics)

            if not args.keep_logs:
                pass

        script_summary = summarize(metric_runs)
        summary[script] = script_summary
        if script_summary:
            print("  => mean卤std")
            for k, v in script_summary.items():
                print(f"     {k}: {v['mean']:.4f} 卤 {v['std']:.4f} (n={v['n']})")
        else:
            print("  => no summary (no parsed metrics)")

    out_json = base_dir / "repro_3x_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary: {out_json}")
    print(f"Saved logs dir: {logs_dir}")
    print("Note: if original scripts keep fixed random_state, std may be near 0. This is expected.")


if __name__ == "__main__":
    main()

