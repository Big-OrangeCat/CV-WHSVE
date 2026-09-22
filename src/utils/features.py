"""Feature extraction for 125-frame pupil light-reflex sequences.

The implementation mirrors the feature families reported in the manuscript.
Missing samples are median-padded per record. Features that describe relative
change are calculated from a sequence corrected by the mean of frames 1--30;
absolute response and segment-ratio features retain the median-padded diameter
scale so their denominators remain interpretable.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import cwt, find_peaks, morlet2


EPSILON = 1e-9
BASELINE_FRAMES = 30
SCATTERING_WIDTHS = np.asarray([2.0, 4.0, 8.0, 16.0])


def _median_pad_rows(x: np.ndarray) -> np.ndarray:
    padded = np.asarray(x, dtype=np.float64).copy()
    for row in padded:
        finite = np.isfinite(row)
        fill = float(np.median(row[finite])) if finite.any() else 0.0
        row[~finite] = fill
    return padded


def baseline_correct(x: np.ndarray, baseline_frames: int = BASELINE_FRAMES) -> np.ndarray:
    padded = _median_pad_rows(x)
    baseline = np.mean(padded[:, :baseline_frames], axis=1, keepdims=True)
    return padded - baseline


def _approximate_entropy(values: np.ndarray, m: int = 2, r: float | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= m + 1:
        return 0.0
    tolerance = 0.2 * np.std(values) if r is None else float(r)
    if tolerance <= EPSILON:
        return 0.0

    def phi(order: int) -> float:
        windows = np.asarray([values[i : i + order] for i in range(n - order + 1)])
        distances = np.max(np.abs(windows[:, None, :] - windows[None, :, :]), axis=2)
        counts = np.mean(distances <= tolerance, axis=1)
        return float(np.mean(np.log(counts + EPSILON)))

    return float(phi(m) - phi(m + 1))


def _sample_entropy(values: np.ndarray, m: int = 2, r: float | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= m + 1:
        return 0.0
    tolerance = 0.2 * np.std(values) if r is None else float(r)
    if tolerance <= EPSILON:
        return 0.0

    def matches(order: int) -> int:
        windows = np.asarray([values[i : i + order] for i in range(n - order + 1)])
        distances = np.max(np.abs(windows[:, None, :] - windows[None, :, :]), axis=2)
        return int(np.sum(np.triu(distances <= tolerance, k=1)))

    b = matches(m)
    a = matches(m + 1)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


def _multiscale_entropy(values: np.ndarray, max_scale: int = 3) -> float:
    estimates: list[float] = []
    for scale in range(1, max_scale + 1):
        length = len(values) // scale
        if length <= 3:
            continue
        coarse = values[: length * scale].reshape(length, scale).mean(axis=1)
        estimates.append(_sample_entropy(coarse))
    return float(np.mean(estimates)) if estimates else 0.0


def _dtw_distance(values: np.ndarray, reference: np.ndarray) -> float:
    """Classic dynamic-time-warping distance to a fixed reference trajectory."""
    a = np.asarray(values, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    scale = np.std(a)
    a = (a - np.mean(a)) / (scale if scale > EPSILON else 1.0)
    previous = np.full(len(b) + 1, np.inf)
    previous[0] = 0.0
    for value in a:
        current = np.full(len(b) + 1, np.inf)
        for j, target in enumerate(b, start=1):
            current[j] = abs(value - target) + min(current[j - 1], previous[j], previous[j - 1])
        previous = current
    return float(previous[-1])


def _wavelet_scattering_summary(values: np.ndarray) -> tuple[float, float, float]:
    """Return first- and second-order complex-Morlet scattering summaries."""
    first_order = np.abs(cwt(values, morlet2, SCATTERING_WIDTHS, w=5.0))
    first_summary = first_order.mean(axis=1)
    second_summary: list[float] = []
    for index, width_1 in enumerate(SCATTERING_WIDTHS[:-1]):
        wider = SCATTERING_WIDTHS[SCATTERING_WIDTHS > width_1]
        second = np.abs(cwt(first_order[index], morlet2, wider, w=5.0))
        second_summary.extend(second.mean(axis=1).tolist())
    return (
        float(np.mean(first_summary)),
        float(np.std(first_summary)),
        float(np.max(second_summary)) if second_summary else 0.0,
    )


def extract_plr_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract 44 candidates; the three acquisition constants are filtered."""
    raw = _median_pad_rows(x)
    corrected = baseline_correct(raw)
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
        "seg3_seg1_ratio", "approximate_entropy", "sample_entropy",
        "multiscale_entropy", "wst2_max", "seg3_seg1_log_ratio",
        "baseline_reference", "stimulus_onset_frame", "sequence_length",
    ]
    time = np.arange(corrected.shape[1], dtype=np.float64)
    dtw_reference = np.linspace(0.0, 1.0, corrected.shape[1], dtype=np.float64)

    for raw_seq, seq in zip(raw, corrected):
        d1 = np.diff(seq)
        d2 = np.diff(d1)
        spectrum = np.abs(np.fft.rfft(seq))
        strongest = np.sort(spectrum[1:])[::-1][:2]
        if len(strongest) < 2:
            strongest = np.pad(strongest, (0, 2 - len(strongest)))
        wst1_mean, wst1_std, wst2_max = _wavelet_scattering_summary(seq)

        peaks, _ = find_peaks(seq)
        troughs, _ = find_peaks(-seq)
        direction_changes = np.diff(np.sign(d1)) != 0 if len(d1) > 1 else np.asarray([])

        pre_raw = raw_seq[:BASELINE_FRAMES]
        post_raw = raw_seq[BASELINE_FRAMES:]
        amax = float(np.max(pre_raw))
        amin_index = int(np.argmin(post_raw))
        amin = float(post_raw[amin_index])
        aac = amax - amin
        rac = aac / (abs(amax) + EPSILON)
        tamin = float(amin_index)
        contraction = np.diff(raw_seq[BASELINE_FRAMES - 1 : BASELINE_FRAMES + amin_index + 1])
        scv = float(np.min(contraction)) if len(contraction) else 0.0
        dilation = np.diff(post_raw[amin_index:])
        max_dilation_velocity = float(np.max(dilation)) if len(dilation) else 0.0
        normalized_constriction_rate = abs(scv) / (abs(amax) + EPSILON)

        recovery_level = amin + 0.75 * aac
        recovery_candidates = np.where(post_raw[amin_index:] >= recovery_level)[0]
        t75 = float(recovery_candidates[0]) if len(recovery_candidates) else float(len(post_raw) - amin_index)

        pre_auc = float(np.trapz(pre_raw))
        post_auc = float(np.trapz(post_raw))
        auc_ratio = post_auc / (abs(pre_auc) + EPSILON)
        seg1_mean = float(np.mean(raw_seq[:30]))
        seg3_mean = float(np.mean(raw_seq[60:90]))
        seg_ratio = seg3_mean / (seg1_mean + EPSILON)
        seg_log_ratio = float(np.log((abs(seg3_mean) + EPSILON) / (abs(seg1_mean) + EPSILON)))

        row = [
            np.mean(seq), np.std(seq), np.max(seq), np.min(seq), np.ptp(seq), np.var(seq), np.trapz(seq),
            np.max(np.abs(d1)), np.polyfit(time, seq, 1)[0], np.mean(d1), np.max(np.abs(d1)), np.mean(d2),
            np.mean(np.abs(d1)), np.mean(d1), _dtw_distance(seq, dtw_reference),
            strongest[0], strongest[1], wst1_mean, wst1_std,
            amax, amin, aac, rac, tamin, scv, max_dilation_velocity,
            normalized_constriction_rate, pre_auc, post_auc, auc_ratio, t75,
            len(peaks), len(troughs), np.mean(direction_changes) if len(direction_changes) else 0.0,
            np.mean(seq[-max(1, len(seq) // 10) :]), seg_ratio,
            _approximate_entropy(seq), _sample_entropy(seq), _multiscale_entropy(seq),
            wst2_max, seg_log_ratio,
            0.0, float(BASELINE_FRAMES), float(len(seq)),
        ]
        rows.append([float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in row])
    return np.asarray(rows, dtype=np.float32), names


def remove_zero_variance(x_train: np.ndarray, x_test: np.ndarray, names: list[str]):
    mask = np.var(x_train, axis=0) > 0
    return x_train[:, mask], x_test[:, mask], [n for n, keep in zip(names, mask) if keep]
