# metrics.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from model import EMGRecording


def compute_atrain_metrics(
    rec: EMGRecording,
    channels: Optional[List[str]] = None,
    peak_min_fraction: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute global + per-channel + per-train metrics from EMGRecording and A-train episodes.

    Requires:
      rec.atrains: dict[channel -> list[(start_idx, end_idx)]], end exclusive

    Adds:
      - peak features (n_peaks, peak_rate)
      - full-track energy (not gated) per channel and global
    """
    if channels is None:
        channels = rec.channel_names()

    fs = float(rec.fs)
    dt_s = 1.0 / fs
    n = rec.n_samples()
    total_duration_s = n * dt_s

    # --- normalize & merge episodes per channel ---
    atrains: Dict[str, List[Tuple[int, int]]] = {}
    for ch in channels:
        eps = (rec.atrains.get(ch, []) if getattr(rec, "atrains", None) else [])
        eps = _sanitize_intervals(eps, n)
        eps = _merge_intervals(eps)
        atrains[ch] = eps

    # --- per-train metrics ---
    per_train: Dict[str, List[Dict[str, Any]]] = {}
    for ch in channels:
        per_train[ch] = []
        sig = np.asarray(rec.emg[ch], dtype=float)
        rect = np.abs(np.nan_to_num(sig, nan=0.0))

        for (a, b) in atrains[ch]:
            ep = _compute_episode_metrics(
                rect, a, b, dt_s, peak_min_fraction=peak_min_fraction
            )
            ep["start_idx"] = int(a)
            ep["end_idx"] = int(b)
            ep["duration_s"] = float((b - a) * dt_s)
            per_train[ch].append(ep)

    # --- per-channel metrics (aggregate over episodes) ---
    per_channel: Dict[str, Dict[str, Any]] = {}
    sum_train_time = 0.0
    sum_energy_trains = 0.0
    sum_energy_full = 0.0
    n_bursts_total = 0
    longest_burst_any_s = 0.0

    for ch in channels:
        eps = atrains[ch]
        train_samples = _intervals_total_length(eps)
        train_time_s = train_samples * dt_s
        duty_cycle = (train_time_s / total_duration_s) if total_duration_s > 0 else 0.0

        sig = np.asarray(rec.emg[ch], dtype=float)
        rect = np.abs(np.nan_to_num(sig, nan=0.0))

        # full-track energy (NOT gated)
        # energy = integral rect^2 dt  (rect in whatever units; if microvolts => uV^2*s)
        full_sumsq = float(np.sum(rect * rect))
        energy_full = float(full_sumsq * dt_s)

        # gated (train-only) energy + RMS
        if train_samples > 0:
            sumsq_train = 0.0
            count_train = 0
            for (a, b) in eps:
                x = rect[a:b]
                sumsq_train += float(np.sum(x * x))
                count_train += (b - a)
            energy_trains = float(sumsq_train * dt_s)
            rms_trains = float(np.sqrt(sumsq_train / max(count_train, 1)))
        else:
            energy_trains = 0.0
            rms_trains = None

        n_bursts = len(eps)
        n_bursts_total += n_bursts

        if n_bursts > 0:
            durations = [(b - a) * dt_s for (a, b) in eps]
            longest_burst_s = float(max(durations))
            mean_burst_s = float(np.mean(durations))
        else:
            longest_burst_s = 0.0
            mean_burst_s = 0.0

        longest_burst_any_s = max(longest_burst_any_s, longest_burst_s)

        # pool freqs from all episodes for this channel
        freqs_pool = []
        for ep in per_train[ch]:
            f = ep.get("freqs_hz", None)
            if isinstance(f, np.ndarray) and f.size > 0:
                freqs_pool.append(f)
        if freqs_pool:
            freqs_pool = np.concatenate(freqs_pool)
            freq_mean_hz = float(np.mean(freqs_pool))
            freq_median_hz = float(np.median(freqs_pool))
            freq_std_hz = float(np.std(freqs_pool))
            freq_cv = float(freq_std_hz / freq_mean_hz) if freq_mean_hz > 0 else None
        else:
            freq_mean_hz = None
            freq_median_hz = None
            freq_std_hz = None
            freq_cv = None

        # peak aggregates (per channel)
        total_peaks_ch = int(sum(ep.get("n_peaks", 0) for ep in per_train[ch]))
        peaks_per_train_second = (total_peaks_ch / train_time_s) if train_time_s > 0 else None
        peaks_per_total_second = (total_peaks_ch / total_duration_s) if total_duration_s > 0 else None

        per_channel[ch] = {
            "train_time_s": float(train_time_s),
            "duty_cycle": float(duty_cycle),
            "energy_trains": float(energy_trains),
            "rms_trains": rms_trains,
            "n_bursts": int(n_bursts),
            "longest_burst_s": float(longest_burst_s),
            "mean_burst_s": float(mean_burst_s),
            "freq_mean_hz": freq_mean_hz,
            "freq_median_hz": freq_median_hz,
            "freq_std_hz": freq_std_hz,
            "freq_cv": freq_cv,
            "total_peaks": total_peaks_ch,
            "peaks_per_train_second": peaks_per_train_second,
            "peaks_per_total_second": peaks_per_total_second,
            "energy_full": float(energy_full),  # <-- NEW
        }

        sum_train_time += train_time_s
        sum_energy_trains += energy_trains
        sum_energy_full += energy_full

    # --- global synchrony using intervals (no full masks) ---
    intervals_any = _union_across_channels([atrains[ch] for ch in channels])
    time_any_channel = _intervals_total_length(intervals_any) * dt_s

    if len(channels) == 0:
        time_all_channels = 0.0
    elif len(channels) == 1:
        time_all_channels = per_channel[channels[0]]["train_time_s"]
    else:
        intervals_all = _intersection_across_channels([atrains[ch] for ch in channels])
        time_all_channels = _intervals_total_length(intervals_all) * dt_s

    synchrony_fraction = (time_all_channels / time_any_channel) if time_any_channel > 0 else None

    # --- global frequency summary: average of per-channel summaries (like old script) ---
    freq_means   = [v["freq_mean_hz"]   for v in per_channel.values() if v["freq_mean_hz"]   is not None]
    freq_medians = [v["freq_median_hz"] for v in per_channel.values() if v["freq_median_hz"] is not None]
    freq_stds    = [v["freq_std_hz"]    for v in per_channel.values() if v["freq_std_hz"]    is not None]
    freq_cvs     = [v["freq_cv"]        for v in per_channel.values() if v["freq_cv"]        is not None]

    global_freq_mean_hz   = float(np.mean(freq_means))   if freq_means   else None
    global_freq_median_hz = float(np.mean(freq_medians)) if freq_medians else None
    global_freq_std_hz    = float(np.mean(freq_stds))    if freq_stds    else None
    global_freq_cv        = float(np.mean(freq_cvs))     if freq_cvs     else None

    # --- global peak aggregates ---
    total_peaks = int(sum(per_channel[ch]["total_peaks"] for ch in channels)) if channels else 0
    peaks_per_total_second = (total_peaks / total_duration_s) if total_duration_s > 0 else None
    peaks_per_any_train_second = (total_peaks / time_any_channel) if time_any_channel > 0 else None

    global_metrics = {
        "sum_train_time_s": float(sum_train_time),
        "time_any_channel_s": float(time_any_channel),
        "time_all_channels_s": float(time_all_channels),
        "synchrony_fraction": synchrony_fraction,
        "sum_energy_trains": float(sum_energy_trains),
        "sum_energy_full": float(sum_energy_full),  # <-- NEW (not gated)
        "n_bursts_total": int(n_bursts_total),
        "longest_burst_any_channel_s": float(longest_burst_any_s),
        "global_freq_mean_hz": global_freq_mean_hz,
        "global_freq_median_hz": global_freq_median_hz,
        "global_freq_std_hz": global_freq_std_hz,
        "global_freq_cv": global_freq_cv,
        "total_peaks": total_peaks,
        "peaks_per_total_second": peaks_per_total_second,
        "peaks_per_any_train_second": peaks_per_any_train_second,
    }

    return {
        "total_duration_s": float(total_duration_s),
        "global": global_metrics,
        "per_channel": per_channel,
        "per_train": per_train,
    }


# ============================================================
# Episode-level metrics
# ============================================================

def _compute_episode_metrics(
    rect: np.ndarray,
    start: int,
    end: int,
    dt_s: float,
    peak_min_fraction: float = 0.5,
) -> Dict[str, Any]:
    """
    Metrics for one episode on one channel.
    Includes:
      - n_peaks (explicit peak count)
      - peak_rate_hz
      - freqs_hz array (for pooling)
    """
    seg = rect[start:end]
    dur_s = (end - start) * dt_s

    if seg.size == 0:
        return {
            "energy": 0.0,
            "rms": None,
            "n_peaks": 0,
            "peak_rate_hz": None,
            "freqs_hz": np.array([], dtype=float),
            "freq_mean_hz": None,
            "freq_median_hz": None,
            "freq_std_hz": None,
            "freq_cv": None,
        }

    sumsq = float(np.sum(seg * seg))
    energy = float(sumsq * dt_s)
    rms = float(np.sqrt(sumsq / max(seg.size, 1)))

    n_peaks, freqs_hz = _episode_peaks_and_freqs(
        seg, dt_s=dt_s, peak_min_fraction=peak_min_fraction
    )

    peak_rate_hz = (float(n_peaks) / dur_s) if (dur_s > 0) else None

    if freqs_hz.size > 0:
        f_mean = float(np.mean(freqs_hz))
        f_med = float(np.median(freqs_hz))
        f_std = float(np.std(freqs_hz))
        f_cv = float(f_std / f_mean) if f_mean > 0 else None
    else:
        f_mean = f_med = f_std = f_cv = None

    return {
        "energy": energy,
        "rms": rms,
        "n_peaks": int(n_peaks),
        "peak_rate_hz": peak_rate_hz,
        "freqs_hz": freqs_hz,
        "freq_mean_hz": f_mean,
        "freq_median_hz": f_med,
        "freq_std_hz": f_std,
        "freq_cv": f_cv,
    }


def _episode_peaks_and_freqs(
    rect_segment: np.ndarray,
    dt_s: float,
    peak_min_fraction: float = 0.5,
) -> tuple[int, np.ndarray]:
    """
    Peak detection on rectified segment + IPI->freq conversion.

    Returns
    -------
    n_peaks : int
    freqs_hz : np.ndarray shape (n_peaks-1,)
    """
    r = rect_segment
    if r.size <= 2:
        return 0, np.array([], dtype=float)

    mid = r[1:-1]
    left = r[:-2]
    right = r[2:]

    peak_mask = (mid > left) & (mid >= right)
    if not np.any(peak_mask):
        return 0, np.array([], dtype=float)

    # amplitude gating: keep peaks >= fraction * median_peak_amp
    peak_amps = mid[peak_mask]
    amp_thresh = np.median(peak_amps) * float(peak_min_fraction)
    peak_mask = peak_mask & (mid >= amp_thresh)

    peak_idx = np.where(peak_mask)[0] + 1
    n_peaks = int(peak_idx.size)
    if n_peaks < 2:
        return n_peaks, np.array([], dtype=float)

    ipi_s = np.diff(peak_idx).astype(float) * dt_s
    ipi_s = ipi_s[ipi_s > 0]
    if ipi_s.size == 0:
        return n_peaks, np.array([], dtype=float)

    freqs_hz = (1.0 / ipi_s).astype(float)
    return n_peaks, freqs_hz


# ============================================================
# Interval helpers (no full-length masks)
# ============================================================

def _sanitize_intervals(eps: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    out = []
    for a, b in eps:
        a = int(a)
        b = int(b)
        if b <= a:
            continue
        a = max(0, a)
        b = min(n, b)
        if b > a:
            out.append((a, b))
    return out


def _merge_intervals(eps: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not eps:
        return []
    eps = sorted(eps, key=lambda x: x[0])
    merged = [list(eps[0])]
    for a, b in eps[1:]:
        if a <= merged[-1][1]:  # overlap/touch
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(int(x[0]), int(x[1])) for x in merged]


def _intervals_total_length(eps: List[Tuple[int, int]]) -> int:
    return int(sum((b - a) for a, b in eps))


def _union_across_channels(list_of_eps: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    all_eps: List[Tuple[int, int]] = []
    for eps in list_of_eps:
        all_eps.extend(eps)
    return _merge_intervals(all_eps)


def _intersect_two(a_eps: List[Tuple[int, int]], b_eps: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Intersect two interval lists (both assumed merged + sorted)."""
    out: List[Tuple[int, int]] = []
    i = j = 0
    while i < len(a_eps) and j < len(b_eps):
        a0, a1 = a_eps[i]
        b0, b1 = b_eps[j]
        lo = max(a0, b0)
        hi = min(a1, b1)
        if hi > lo:
            out.append((lo, hi))
        if a1 <= b1:
            i += 1
        else:
            j += 1
    return out


def _intersection_across_channels(list_of_eps: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    if not list_of_eps:
        return []
    cur = list_of_eps[0]
    for nxt in list_of_eps[1:]:
        cur = _intersect_two(cur, nxt)
        if not cur:
            break
    return cur
