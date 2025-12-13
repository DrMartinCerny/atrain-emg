from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

from model import EMGRecording


def detect_a_trains(
    rec: EMGRecording,
    channels: Optional[List[str]] = None,
    window_ms: float = 30.0,
    threshold_sigma: float = 2.5,
    min_duration_ms: float = 80.0,
    min_peaks: int = 6,
    peak_sigma: float = 2.0,
    min_freq_hz: float = 20.0,
    max_freq_hz: float = 200.0,
    max_cv_ipi: float = 0.7,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Detect A-trains per channel. Returns episode list per channel:
      {channel: [(start_idx, end_idx), ...]}  (end exclusive)

    Uses:
      - rectified + smoothed envelope threshold (robust MAD)
      - minimum duration
      - minimum peaks
      - approximate periodicity constraints via inter-peak interval (IPI)
    """
    if channels is None:
        channels = rec.channel_names()

    fs = float(rec.fs)
    dt_s = 1.0 / fs

    # ms -> samples
    win_samples = int(window_ms / 1000.0 / dt_s)
    win_samples = max(win_samples, 1)

    min_samples = int(min_duration_ms / 1000.0 / dt_s)
    min_samples = max(min_samples, 1)

    # frequency -> IPI bounds in samples
    min_period_s = 1.0 / max_freq_hz
    max_period_s = 1.0 / min_freq_hz
    min_period_samples = max(int(np.floor(min_period_s / dt_s)), 1)
    max_period_samples = max(int(np.ceil(max_period_s / dt_s)), 1)

    kernel = np.ones(win_samples, dtype=float) / win_samples

    episodes: Dict[str, List[Tuple[int, int]]] = {}

    for ch in channels:
        sig = rec.emg[ch]
        x = np.nan_to_num(sig.astype(float), nan=0.0)

        rect = np.abs(x)
        smooth = np.convolve(rect, kernel, mode="same")

        baseline = np.median(smooth)
        mad = np.median(np.abs(smooth - baseline))
        if mad == 0:
            mad = 1e-9

        z_env = (smooth - baseline) / mad
        env_mask = z_env > threshold_sigma

        # min duration: compute run lengths in a cheap way
        # (convolution trick gives "how many trues in window")
        env_int = env_mask.astype(np.int8)
        run_counts = np.convolve(env_int, np.ones(min_samples, dtype=np.int8), mode="same")
        candidate_mask = run_counts >= min_samples

        peak_thresh = baseline + peak_sigma * mad

        ch_eps: List[Tuple[int, int]] = []

        in_seg = False
        seg_start = 0

        # iterate over candidate segments
        for i in range(len(candidate_mask) + 1):
            if i < len(candidate_mask) and candidate_mask[i]:
                if not in_seg:
                    in_seg = True
                    seg_start = i
            else:
                if in_seg:
                    seg_end = i  # exclusive
                    in_seg = False

                    seg_len = seg_end - seg_start
                    if seg_len < min_samples:
                        continue

                    r = rect[seg_start:seg_end]

                    # peak picking (local maxima above threshold)
                    if len(r) > 2:
                        mid = r[1:-1]
                        left = r[:-2]
                        right = r[2:]
                        peak_mask_local = (mid > left) & (mid >= right) & (mid > peak_thresh)
                        peak_idx = np.where(peak_mask_local)[0] + 1  # local to segment
                    else:
                        peak_idx = np.array([], dtype=int)

                    if len(peak_idx) < min_peaks:
                        continue

                    ipi = np.diff(peak_idx)
                    ipi_valid = ipi[(ipi >= min_period_samples) & (ipi <= max_period_samples)]

                    if len(ipi_valid) < (min_peaks - 1):
                        continue

                    mean_ipi = float(ipi_valid.mean())
                    std_ipi = float(ipi_valid.std())
                    cv = (std_ipi / mean_ipi) if mean_ipi > 0 else 1e9

                    if cv <= max_cv_ipi:
                        ch_eps.append((seg_start, seg_end))

        episodes[ch] = _merge_close_episodes(ch_eps, gap_samples=max(1, int(0.01 / dt_s)))  # merge if <10ms apart

    return episodes


def _merge_close_episodes(eps: List[Tuple[int, int]], gap_samples: int) -> List[Tuple[int, int]]:
    """Merge episodes where next.start <= prev.end + gap_samples."""
    if not eps:
        return []
    eps = sorted(eps, key=lambda x: x[0])
    merged = [list(eps[0])]
    for a, b in eps[1:]:
        if a <= merged[-1][1] + gap_samples:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(int(x[0]), int(x[1])) for x in merged]
