from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EMGRecording:
    """
    EMG container.

    emg:
        dict[channel_name -> np.ndarray shape (n_samples,)]
    fs:
        sampling frequency (Hz)
    time:
        raw timing column from the export (usually sample index 0..N-1)
    """

    emg: Dict[str, np.ndarray]
    fs: float

    time: Optional[np.ndarray] = None  # raw first column (e.g., sample index)
    start_timestamp: Optional[datetime] = None
    units: str = "unknown"

    meta: Dict[str, Any] = field(default_factory=dict)
    channel_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def channel_names(self) -> list[str]:
        return list(self.emg.keys())

    def n_channels(self) -> int:
        return len(self.emg)

    def n_samples(self) -> int:
        if not self.emg:
            return 0
        return int(min(len(x) for x in self.emg.values()))

    def duration_s(self) -> float:
        return (self.n_samples() / float(self.fs)) if self.fs else 0.0

    def get_matrix(self, channels: Optional[list[str]] = None) -> np.ndarray:
        """Return (n_channels, n_samples) matrix; truncates to min length."""
        if channels is None:
            channels = self.channel_names()
        n = self.n_samples()
        return np.stack([self.emg[ch][:n] for ch in channels], axis=0)

    def print_summary(self) -> None:
        n_ch = self.n_channels()
        n_samp = self.n_samples()
        dur = self.duration_s()

        print("=== EMGRecording Summary ===")
        print(f"Channels      : {n_ch}")
        print(f"Samples       : {n_samp}")
        print(f"Sampling rate : {self.fs:.6g} Hz")
        print(f"Duration      : {dur:.3f} s  ({dur/60:.3f} min)")
        if self.start_timestamp is not None:
            print(f"Start time    : {self.start_timestamp.isoformat(sep=' ')}")
        print(f"Units         : {self.units}")

        if self.time is not None:
            print(f"Timing column : present (len={len(self.time)})")
            tqc = self.meta.get("timing_qc")
            if tqc:
                print("\n--- Timing QC ---")
                for k in sorted(tqc.keys()):
                    print(f"{k:>22}: {tqc[k]}")

        if self.meta:
            print("\n--- Global metadata ---")
            for k in sorted(self.meta.keys()):
                if k == "timing_qc":
                    continue
                print(f"{k:>14}: {self.meta[k]}")

        print("===========================")

    def __repr__(self) -> str:
        st = self.start_timestamp.isoformat(sep=" ") if self.start_timestamp else "None"
        return (
            f"EMGRecording(n_channels={self.n_channels()}, "
            f"n_samples={self.n_samples()}, fs={self.fs:.6g}, "
            f"duration_s={self.duration_s():.3f}, start_timestamp={st}, units={self.units!r})"
        )
