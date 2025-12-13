from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from model import EMGRecording


def load_iomax(
    path_to_csv: str | Path,
    channels_to_keep: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    strict_timing: bool = False,
) -> EMGRecording:
    """
    Load EMG channels from Cadwell IOMax CSV export (fast).

    Your export layout:
      - Row 0: header / column names (semicolon-separated)
      - Rows 1-7: per-channel metadata (Time, Gain, Highcut, Lowcut, Notch, Period (s), Units)
      - Row 8+: sample data
      - First column is timing/sample index (usually named "Name", but we robustly detect it)

    Timing handling:
      - keeps timing column as rec.time (np.ndarray)
      - checks strict continuity: time[i+1] == time[i] + 1
      - stores QC into rec.meta["timing_qc"]
      - if strict_timing=True and noncontinuous => raises ValueError
    """
    path_to_csv = Path(path_to_csv)

    # ----------------------------
    # 1) FAST HEADER READ
    # ----------------------------
    available_cols = _read_header_columns_fast(path_to_csv)
    if not available_cols:
        raise ValueError(f"No columns found in {path_to_csv}")

    # Prefer explicit 'Name' if present; otherwise assume first column is timing
    timing_col = "Name" if "Name" in available_cols else available_cols[0]

    # ----------------------------
    # 2) SELECT CHANNELS
    # ----------------------------
    if channels_to_keep is None:
        channels = [c for c in available_cols if c != timing_col]
    else:
        req = [_norm_col(c) for c in channels_to_keep]
        missing = [c for c in req if c not in available_cols]
        if missing:
            raise ValueError(
                "Some requested channels are not present in the file.\n"
                f"Missing: {missing}\nAvailable: {available_cols}"
            )
        channels = req

    # Always include timing column
    usecols = [timing_col] + channels

    # ----------------------------
    # 3) READ METADATA BLOCK (ROWS 1..7)
    # ----------------------------
    # Small read, overhead negligible.
    meta_block = pd.read_csv(
        path_to_csv,
        sep=";",
        header=None,
        nrows=7,
        skiprows=1,  # skip header row only
        encoding="utf-8",
    )

    col_index = {name: idx for idx, name in enumerate(available_cols)}
    channel_meta: Dict[str, Dict[str, Any]] = {ch: {} for ch in channels}
    global_meta: Dict[str, Any] = {"source_file": str(path_to_csv), "format": "iomax_csv"}

    for _, row in meta_block.iterrows():
        key = str(row.iloc[0]).strip()
        if not key or key.lower() == "nan":
            continue
        for ch in channels:
            idx = col_index[ch]
            channel_meta[ch][key] = row.iloc[idx] if idx < len(row) else None

    # ----------------------------
    # 4) INFER FS, TIMESTAMP, UNITS
    # ----------------------------
    fs = _infer_fs_from_channel_meta(channel_meta, channels)
    start_ts = _infer_start_timestamp(channel_meta, channels)
    units = _infer_units(channel_meta, channels)

    # ----------------------------
    # 5) READ DATA BLOCK (FAST PATH)
    # ----------------------------
    # Keep header row 0; skip metadata rows 1..7
    skiprows = list(range(1, 8))

    dtype_map: Dict[str, Any] = {ch: np.float32 for ch in channels}
    # timing column should be integer-ish; parse as float if it might contain decimals
    dtype_map[timing_col] = np.float64

    raw_data = pd.read_csv(
        path_to_csv,
        sep=";",
        header=0,
        usecols=usecols,
        skiprows=skiprows,
        nrows=max_samples,
        decimal=",",
        encoding="utf-8",
        dtype=dtype_map,
        memory_map=True,
        low_memory=False,
    )

    # ----------------------------
    # 6) EXTRACT TIMING + QC
    # ----------------------------
    if timing_col not in raw_data.columns:
        # This should never happen if header/usecols are correct.
        raise ValueError(
            f"Timing column {timing_col!r} not found after loading. "
            f"Loaded columns: {list(raw_data.columns)}"
        )

    time_col = raw_data[timing_col].to_numpy(dtype=np.float64, copy=False)
    raw_data = raw_data.drop(columns=[timing_col])

    qc = _timing_increment_check(time_col)
    global_meta["timing_qc"] = qc
    global_meta["timing_column_name"] = timing_col

    if strict_timing and qc.get("status") != "ok":
        raise ValueError(f"Timing continuity check failed: {qc}")

    # ----------------------------
    # 7) BUILD CHANNEL DICT (FAST)
    # ----------------------------
    emg: Dict[str, np.ndarray] = {}
    for ch in channels:
        if ch not in raw_data.columns:
            raise ValueError(f"Channel {ch!r} not found in data rows.")
        emg[ch] = raw_data[ch].to_numpy(dtype=np.float32, copy=False)

    return EMGRecording(
        emg=emg,
        fs=float(fs),
        time=time_col,
        start_timestamp=start_ts,
        units=units,
        meta=global_meta,
        channel_meta=channel_meta,
    )


# ============================
# Helper functions (below)
# ============================

def _norm_col(s: str) -> str:
    # Remove UTF-8 BOM + surrounding whitespace
    return str(s).replace("\ufeff", "").strip()


def _read_header_columns_fast(path: Path) -> List[str]:
    # Read first line only; IOMax uses ';' delimiter
    with path.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline().strip("\n\r")
    return [_norm_col(x) for x in line.split(";")]


def _parse_float_maybe(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_timestamp_yyyymmddhhmmss(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _infer_fs_from_channel_meta(channel_meta: Dict[str, Dict[str, Any]], channels: List[str]) -> float:
    fs = None
    for ch in channels:
        period = _parse_float_maybe(channel_meta[ch].get("Period (s)", None))
        if period and period > 0:
            fs = 1.0 / period
            break
    if fs is None:
        raise ValueError("Could not infer sampling rate from 'Period (s)'.")
    return float(fs)


def _infer_start_timestamp(channel_meta: Dict[str, Dict[str, Any]], channels: List[str]) -> Optional[datetime]:
    for ch in channels:
        ts = _parse_timestamp_yyyymmddhhmmss(channel_meta[ch].get("Time", None))
        if ts is not None:
            return ts
    return None


def _infer_units(channel_meta: Dict[str, Dict[str, Any]], channels: List[str]) -> str:
    units_set = set()
    for ch in channels:
        u = channel_meta[ch].get("Units", None)
        if u is not None and str(u).strip():
            units_set.add(str(u).strip())
    if len(units_set) == 1:
        return units_set.pop()
    if len(units_set) > 1:
        return "mixed"
    return "unknown"


def _timing_increment_check(time_col: np.ndarray) -> Dict[str, Any]:
    """
    Strict check: each value must be exactly previous + 1
    (after rounding to int).
    """
    out: Dict[str, Any] = {}

    if time_col is None or len(time_col) < 2:
        out["status"] = "too_short_or_missing"
        return out

    if np.any(~np.isfinite(time_col)):
        bad_idx = np.where(~np.isfinite(time_col))[0]
        out["status"] = "bad_nan_or_inf"
        out["nan_or_inf_count"] = int(len(bad_idx))
        out["nan_or_inf_idx_first"] = bad_idx[:10].tolist()
        return out

    t_int = np.rint(time_col).astype(np.int64)
    dt = np.diff(t_int)
    bad = np.where(dt != 1)[0]

    out["expected_increment"] = 1
    out["noncontinuous_count"] = int(len(bad))
    out["status"] = "ok" if len(bad) == 0 else "noncontinuous"
    out["bad_transition_idx_first"] = bad[:10].tolist()

    if len(bad) > 0:
        i0 = int(bad[0])
        out["first_bad_pair"] = {
            "idx": i0,
            "t_i": int(t_int[i0]),
            "t_next": int(t_int[i0 + 1]),
            "dt": int(dt[i0]),
        }

    # extra quick stats
    out["t0"] = int(t_int[0])
    out["t_end"] = int(t_int[-1])
    out["n_samples"] = int(len(t_int))

    return out
