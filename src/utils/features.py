"""Feature extraction for 125-frame pupil light reflex sequences.

The functions implement a compact public version of the engineered PLR feature
representation described in the manuscript. They avoid any participant-level
data or generated result files.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.stats import entropy


def baseline_correct(x: np.ndarray, baseline_frames: int = 30) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    baseline = np.nanmean(x[:, :baseline_frames], axis=1, keepdims=True)
    return x - baseline


def _safe_entropy(values: np.ndarray) -> float:
    hist, _ = np.histogram(values, bins=10, density=True)
    hist = hist + 1e-12
    return float(entropy(hist / hist.sum()))


def extract_plr_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract multidomain handcrafted features from PLR sequences.

    Parameters
    ----------
    x:
        Array of shape (n_samples, 125).
    """
    raw = np.asarray(x, dtype=np.float32)
    x = baseline_correct(raw)
    rows: list[list[float]] = []
    names = [
        "mean", "std", "max", "min", "range", "variance", "auc",
        "max_diff", "slope", "deriv_mean", "deriv_max", "second_deriv_mean",
        "abs_change", "mean_diff", "dtw_distance",
        "freq_peak1", "freq_peak2", "wst1_mean", "wst1_std",
        "Amax", "Amin", "AAC", "RAC", "TAmin", "SCV",
        "max_dilation_velocity", "normalized_constriction_rate",
        "pre_auc", "post_auc", "auc_ratio", "t75_recovery",
        "n_peaks", "n_troughs", "zero_crossing_rate", "final_baseline_diff",
        "seg3_seg1_ratio",
        "approximate_entropy", "sample_entropy", "multiscale_entropy",
        "wst2_max", "seg3_seg1_log_ratio",
        "baseline_mean_raw", "baseline_std_raw", "stimulus_value_raw",
    ]
    t = np.arange(x.shape[1], dtype=np.float32)
    freqs = rfftfreq(x.shape[1], d=1.0)

    for raw_seq, seq in zip(raw, x):
        raw_seq = np.nan_to_num(raw_seq, nan=float(np.nanmedian(raw_seq)))
        seq = np.nan_to_num(seq, nan=float(np.nanmedian(seq)))
        d1 = np.diff(seq)
        d2 = np.diff(d1)
        fft_mag = np.abs(rfft(seq))
        order = np.argsort(fft_mag[1:])[-2:] + 1 if len(fft_mag) > 2 else np.array([0, 0])
        peaks, _ = find_peaks(seq)
        troughs, _ = find_peaks(-seq)
        seg1 = seq[: len(seq) // 3]
        seg3 = seq[-len(seq) // 3 :]
        amin_idx = int(np.argmin(seq))
        recovery_target = seq[amin_idx] + 0.75 * (seq[0] - seq[amin_idx])
        recovery_candidates = np.where(seq[amin_idx:] >= recovery_target)[0]
        t75 = int(recovery_candidates[0]) if len(recovery_candidates) else len(seq) - amin_idx
        row = [
            np.mean(seq), np.std(seq), np.max(seq), np.min(seq), np.ptp(seq), np.var(seq), np.trapz(seq),
            np.max(np.abs(d1)), np.polyfit(t, seq, 1)[0], np.mean(d1), np.max(d1), np.mean(d2),
            np.sum(np.abs(d1)), np.mean(np.abs(d1)), np.sum(np.abs(seq - np.mean(seq))),
            freqs[order[-1]] if len(order) else 0.0, freqs[order[-2]] if len(order) > 1 else 0.0,
            np.mean(fft_mag), np.std(fft_mag),
            np.max(seq), np.min(seq), abs(np.min(seq) - seq[0]), abs(np.min(seq) - seq[0]) / (amin_idx + 1),
            amin_idx, np.min(d1) if len(d1) else 0.0,
            np.max(d1) if len(d1) else 0.0,
            abs(np.min(d1)) / (abs(seq[0]) + 1e-9),
            np.trapz(seq[:30]), np.trapz(seq[30:]), np.trapz(seq[30:]) / (np.trapz(seq[:30]) + 1e-9), t75,
            len(peaks), len(troughs), np.mean(np.diff(np.signbit(seq)) != 0), seq[-1] - np.mean(seq[:30]),
            np.mean(seg3) / (np.mean(seg1) + 1e-9),
            _safe_entropy(seq), _safe_entropy(d1), np.mean([_safe_entropy(seq[::s]) for s in (1, 2, 3)]),
            np.max(fft_mag), np.log1p(abs(np.mean(seg3) / (np.mean(seg1) + 1e-9))),
            np.mean(raw_seq[:30]), np.std(raw_seq[:30]), raw_seq[30] if len(raw_seq) > 30 else raw_seq[-1],
        ]
        rows.append([float(v) for v in row])
    return np.asarray(rows, dtype=np.float32), names


def remove_zero_variance(x_train: np.ndarray, x_test: np.ndarray, names: list[str]):
    mask = np.var(x_train, axis=0) > 0
    return x_train[:, mask], x_test[:, mask], [n for n, keep in zip(names, mask) if keep]
