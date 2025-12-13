# visualize.py
from __future__ import annotations

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt

from model import EMGRecording


def visualize_emg(
    rec: EMGRecording,
    channels: Optional[List[str]] = None,
    t_start_s: Optional[float] = None,
    t_end_s: Optional[float] = None,
    show_atrains: bool = True,
    figsize=(14, 8),
) -> None:
    """
    Visualize EMGRecording as stacked panels.

    - Always plots EMG traces.
    - If show_atrains=True and rec.atrains has episodes for a channel, overlays shaded regions.
    - If no atrains present (or show_atrains=False), plot is just EMG.

    Parameters
    ----------
    rec : EMGRecording
    channels : list[str] or None
        Channels to plot. Default: all.
    t_start_s, t_end_s : float or None
        Optional time window in seconds (relative to rec start).
    show_atrains : bool
        Overlay A-train episodes if available.
    """
    if channels is None:
        channels = rec.channel_names()

    # Build time axis (seconds)
    n = rec.n_samples()

    if rec.time is not None and len(rec.time) >= n:
        t = (rec.time[:n] - rec.time[0]) / rec.fs
    else:
        t = np.arange(n) / rec.fs

    # Apply time window
    i0 = 0
    i1 = n
    if t_start_s is not None:
        i0 = int(np.searchsorted(t, float(t_start_s), side="left"))
    if t_end_s is not None:
        i1 = int(np.searchsorted(t, float(t_end_s), side="right"))
    i0 = max(0, min(i0, n))
    i1 = max(0, min(i1, n))
    if i1 <= i0:
        raise ValueError("Empty time window: t_end_s must be > t_start_s.")

    t_win = t[i0:i1]

    # Build matrix in requested channel order
    X = rec.get_matrix(channels=channels)  # (n_channels, n_samples)
    X_win = X[:, i0:i1]

    # Plot
    fig, axes = plt.subplots(len(channels), 1, sharex=True, figsize=figsize)
    if len(channels) == 1:
        axes = [axes]

    has_atrains = bool(getattr(rec, "atrains", None))

    for k, ch in enumerate(channels):
        ax = axes[k]
        ax.plot(t_win, X_win[k])
        ax.set_title(ch)
        ax.set_ylabel("EMG")
        ax.grid(True, alpha=0.2)

        # Overlay A-train episodes if available
        if show_atrains and has_atrains and (ch in rec.atrains):
            for (a, b) in rec.atrains.get(ch, []):
                # intersect with window
                aa = max(int(a), i0)
                bb = min(int(b), i1)
                if bb > aa:
                    ax.axvspan(t[aa], t[bb - 1] if (bb - 1) < len(t) else t[-1], alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
