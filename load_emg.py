# load_emg.py
# Flat-structure loader for Cadwell IOMax CSV exports.
# Expects model.py next to this file.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd

from model import EMGRecording


# ============================================================
# Public API (load_iomax FIRST, helpers follow)
# ============================================================

def load_iomax(
    path_or_paths: Union[str, Path, List[Union[str, Path]]],
    channels_to_keep: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    strict_timing: bool = False,
) -> EMGRecording:
    """
    Load EMG channels from Cadwell IOMax CSV export.

    Supports:
      - single file path (str|Path)
      - list of file paths -> sorted by metadata timestamp and concatenated

    Concatenation uses absolute sample indices derived from:
        abs_index = round((start_ts - t0).total_seconds() * fs) + sample_index

    Overlap handling:
      - if a later file overlaps (abs_index <= last_abs_index), those overlapping samples are dropped
      - duplicate abs_index values in later parts are dropped
      - gaps are not filled; they are reported in rec.meta["parts"] and final timing_qc

    Timing QC:
      - final combined rec.time is checked for +1 increments (strict sample continuity)
      - if strict_timing=True and noncontinuous => raises ValueError
    """
    # Normalize input to list
    if isinstance(path_or_paths, (str, Path)):
        paths = [path_or_paths]
    else:
        paths = list(path_or_paths)

    if len(paths) == 0:
        raise ValueError("No paths provided.")

    paths = [Path(p) for p in paths]

    # Single file -> fast load and return
    if len(paths) == 1:
        rec = _load_single_iomax(paths[0], channels_to_keep=channels_to_keep, max_samples=max_samples)
        # QC on the timing column inside the file (as you wanted: +1 increments)
        qc = _timing_increment_check(rec.time.astype(np.float64) if rec.time is not None else None)
        rec.meta["timing_qc"] = qc
        rec.meta["timing_column_name"] = rec.meta.get("timing_column_name", "unknown")
        if strict_timing and qc.get("status") != "ok":
            raise ValueError(f"Timing continuity check failed: {qc}")
        return rec

    # ------------------------------------------------------------
    # Multi-file mode:
    # 1) read metadata first to sort by timestamp
    # ------------------------------------------------------------
    metas: List[Dict[str, Any]] = []
    for p in paths:
        m = _read_iomax_metadata_only(p)
        metas.append(m)

    # sort by start_ts; if missing, keep original order at end
    metas_sorted = sorted(
        metas,
        key=lambda d: (d["start_ts"] is None, d["start_ts"] if d["start_ts"] is not None else datetime.max),
    )

    # sanity: consistent fs across parts (within tolerance)
    fs0 = metas_sorted[0]["fs"]
    for m in metas_sorted[1:]:
        if m["fs"] is None or fs0 is None:
            continue
        if abs(m["fs"] - fs0) > 1e-6:
            raise ValueError(
                "Sampling rate mismatch across parts.\n"
                f"{metas_sorted[0]['path']} fs={fs0}\n"
                f"{m['path']} fs={m['fs']}"
            )

    fs = float(fs0) if fs0 is not None else None
    if fs is None:
        raise ValueError("Could not infer sampling rate from metadata in multi-file mode.")

    # choose global t0 (earliest start timestamp)
    # if any is None -> we cannot do absolute alignment reliably
    if any(m["start_ts"] is None for m in metas_sorted):
        raise ValueError(
            "At least one file is missing metadata start timestamp (Time row). "
            "Cannot sort/align parts reliably."
        )
    t0 = metas_sorted[0]["start_ts"]

    # ------------------------------------------------------------
    # 2) load each file, convert to absolute sample index, glue
    # ------------------------------------------------------------
    parts_info: List[Dict[str, Any]] = []

    combined_time: List[np.ndarray] = []
    combined_emg: Dict[str, List[np.ndarray]] = {}
    combined_channel_meta: Dict[str, Dict[str, Any]] = {}

    last_abs_end: Optional[int] = None
    all_channels_ref: Optional[List[str]] = None

    for mi in metas_sorted:
        p = mi["path"]
        rec = _load_single_iomax(p, channels_to_keep=channels_to_keep, max_samples=max_samples)

        # channels consistency: require exact same set in multi-file glue
        chs = rec.channel_names()
        if all_channels_ref is None:
            all_channels_ref = chs
            for ch in all_channels_ref:
                combined_emg[ch] = []
                # keep per-channel meta from first part
                combined_channel_meta[ch] = (rec.channel_meta.get(ch, {}).copy() if rec.channel_meta else {})
        else:
            if chs != all_channels_ref:
                raise ValueError(
                    "Channel mismatch across parts.\n"
                    f"First part channels: {all_channels_ref}\n"
                    f"This part channels : {chs}\n"
                    "Make channels_to_keep explicit and consistent, or fix export."
                )

        if rec.time is None:
            raise ValueError(f"Timing column not found in data for {p}")

        # Convert per-file sample index to integer (tolerant to 0.0, 1.0 formatting)
        sample_idx = np.rint(rec.time.astype(np.float64)).astype(np.int64)

        # Derive absolute offset from timestamps
        start_ts = rec.start_timestamp
        if start_ts is None:
            raise ValueError(f"Missing start_timestamp in metadata for {p} (Time row).")

        offset = int(round((start_ts - t0).total_seconds() * fs))
        abs_idx = sample_idx + offset

        # Enforce monotonicity within part (by abs index)
        # If export is weird, sort by abs_idx (stable) and reorder signals accordingly.
        order = np.argsort(abs_idx, kind="mergesort")
        abs_idx = abs_idx[order]

        # Reorder EMG arrays accordingly
        part_emg: Dict[str, np.ndarray] = {}
        for ch in all_channels_ref:
            x = rec.emg[ch]
            if len(x) != len(sample_idx):
                # truncate to min
                n = min(len(x), len(sample_idx))
                x = x[:n]
                abs_idx = abs_idx[:n]
                order = np.argsort(abs_idx, kind="mergesort")
                abs_idx = abs_idx[order]
            part_emg[ch] = x[order]

        # Deduplicate within part (same abs index appears multiple times)
        # Keep first occurrence.
        uniq_mask = np.ones(len(abs_idx), dtype=bool)
        if len(abs_idx) > 1:
            dup = np.where(np.diff(abs_idx) == 0)[0]
            # mark second (and later) occurrences as False
            for i in dup:
                uniq_mask[i + 1] = False
        abs_idx_u = abs_idx[uniq_mask]
        for ch in all_channels_ref:
            part_emg[ch] = part_emg[ch][uniq_mask]

        dropped_overlap = 0
        dropped_dup = int(len(abs_idx) - len(abs_idx_u))

        # Handle overlap with previous glued data:
        # drop all samples with abs_idx <= last_abs_end
        if last_abs_end is not None:
            keep = abs_idx_u > last_abs_end
            dropped_overlap = int(np.sum(~keep))
            abs_idx_u = abs_idx_u[keep]
            for ch in all_channels_ref:
                part_emg[ch] = part_emg[ch][keep]

        # Compute gap info (if we have previous end)
        gap = None
        if last_abs_end is not None and len(abs_idx_u) > 0:
            gap = int(abs_idx_u[0] - last_abs_end - 1)

        # Append to combined
        if len(abs_idx_u) > 0:
            combined_time.append(abs_idx_u)
            for ch in all_channels_ref:
                combined_emg[ch].append(part_emg[ch])
            last_abs_end = int(abs_idx_u[-1])

        parts_info.append({
            "path": str(p),
            "start_ts": start_ts.isoformat(sep=" "),
            "offset_samples": offset,
            "n_loaded": int(len(sample_idx)),
            "n_after_dedup": int(len(abs_idx_u) + dropped_overlap),
            "dropped_duplicates_within_part": dropped_dup,
            "dropped_overlap_with_previous": dropped_overlap,
            "gap_samples_from_previous": gap,
            "abs_start": int(abs_idx_u[0]) if len(abs_idx_u) else None,
            "abs_end": int(abs_idx_u[-1]) if len(abs_idx_u) else None,
        })

    # Final concatenate
    if not combined_time:
        raise ValueError("After overlap handling, no samples remained. Check timestamps/overlap logic.")

    time_all = np.concatenate(combined_time).astype(np.int64, copy=False)

    emg_all: Dict[str, np.ndarray] = {}
    for ch, chunks in combined_emg.items():
        emg_all[ch] = np.concatenate(chunks).astype(np.float32, copy=False)

    # Final timing QC on combined absolute index
    qc_final = _timing_increment_check(time_all.astype(np.float64))
    meta_final: Dict[str, Any] = {
        "format": "iomax_csv_multi",
        "parts": parts_info,
        "timing_qc": qc_final,
        "timing_column_name": "absolute_sample_index",
        "t0": t0.isoformat(sep=" "),
    }

    if strict_timing and qc_final.get("status") != "ok":
        raise ValueError(f"Final timing continuity check failed: {qc_final}")

    return EMGRecording(
        emg=emg_all,
        fs=fs,
        time=time_all,
        start_timestamp=t0,
        units=_infer_units(combined_channel_meta, all_channels_ref or []),
        meta=meta_final,
        channel_meta=combined_channel_meta,
    )


# ============================================================
# Helpers
# ============================================================

def _load_single_iomax(
    path_to_csv: Path,
    channels_to_keep: Optional[List[str]],
    max_samples: Optional[int],
) -> EMGRecording:
    """
    Loads a single IOMax CSV quickly.
    Returns rec.time as the raw timing column (float64), NOT QC-checked here.
    """
    path_to_csv = Path(path_to_csv)

    available_cols = _read_header_columns_fast(path_to_csv)
    if not available_cols:
        raise ValueError(f"No columns found in {path_to_csv}")

    timing_col = "Name" if "Name" in available_cols else available_cols[0]

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

    usecols = [timing_col] + channels

    # Metadata block (rows 1..7)
    meta_block = pd.read_csv(
        path_to_csv,
        sep=";",
        header=None,
        nrows=7,
        skiprows=1,
        encoding="utf-8",
    )

    col_index = {name: idx for idx, name in enumerate(available_cols)}
    channel_meta: Dict[str, Dict[str, Any]] = {ch: {} for ch in channels}
    global_meta: Dict[str, Any] = {"source_file": str(path_to_csv), "format": "iomax_csv", "timing_column_name": timing_col}

    for _, row in meta_block.iterrows():
        key = str(row.iloc[0]).strip()
        if not key or key.lower() == "nan":
            continue
        for ch in channels:
            idx = col_index[ch]
            channel_meta[ch][key] = row.iloc[idx] if idx < len(row) else None

    fs = _infer_fs_from_channel_meta(channel_meta, channels)
    start_ts = _infer_start_timestamp(channel_meta, channels)
    units = _infer_units(channel_meta, channels)

    # Data block
    skiprows = list(range(1, 8))

    dtype_map: Dict[str, Any] = {ch: np.float32 for ch in channels}
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

    if timing_col not in raw_data.columns:
        raise ValueError(
            f"Timing column {timing_col!r} not found after loading. "
            f"Loaded columns: {list(raw_data.columns)}"
        )

    time_col = raw_data[timing_col].to_numpy(dtype=np.float64, copy=False)
    raw_data = raw_data.drop(columns=[timing_col])

    emg: Dict[str, np.ndarray] = {}
    for ch in channels:
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


def _read_iomax_metadata_only(path_to_csv: Path) -> Dict[str, Any]:
    """
    Fast-ish: reads only the header + metadata block to get:
      - start_ts
      - fs
      - available columns
    """
    path_to_csv = Path(path_to_csv)

    cols = _read_header_columns_fast(path_to_csv)
    if not cols:
        return {"path": str(path_to_csv), "start_ts": None, "fs": None, "cols": []}

    timing_col = "Name" if "Name" in cols else cols[0]
    channels = [c for c in cols if c != timing_col]

    meta_block = pd.read_csv(
        path_to_csv,
        sep=";",
        header=None,
        nrows=7,
        skiprows=1,
        encoding="utf-8",
    )

    col_index = {name: idx for idx, name in enumerate(cols)}
    channel_meta: Dict[str, Dict[str, Any]] = {ch: {} for ch in channels}

    for _, row in meta_block.iterrows():
        key = str(row.iloc[0]).strip()
        if not key or key.lower() == "nan":
            continue
        for ch in channels:
            idx = col_index[ch]
            channel_meta[ch][key] = row.iloc[idx] if idx < len(row) else None

    fs = None
    try:
        fs = _infer_fs_from_channel_meta(channel_meta, channels)
    except Exception:
        fs = None

    start_ts = _infer_start_timestamp(channel_meta, channels)

    return {"path": str(path_to_csv), "start_ts": start_ts, "fs": fs, "cols": cols}


def _norm_col(s: str) -> str:
    return str(s).replace("\ufeff", "").strip()


def _read_header_columns_fast(path: Path) -> List[str]:
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
    for ch in channels:
        period = _parse_float_maybe(channel_meta[ch].get("Period (s)", None))
        if period and period > 0:
            return float(1.0 / period)
    raise ValueError("Could not infer sampling rate from 'Period (s)'.")


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


def _timing_increment_check(time_col: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Strict check: each value must be exactly previous + 1 (after rounding to int).
    Works for:
      - per-file sample index
      - combined absolute sample index
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

    t_int = np.rint(time_col.astype(np.float64)).astype(np.int64)
    dt = np.diff(t_int)
    bad = np.where(dt != 1)[0]

    out["expected_increment"] = 1
    out["noncontinuous_count"] = int(len(bad))
    out["status"] = "ok" if len(bad) == 0 else "noncontinuous"
    out["bad_transition_idx_first"] = bad[:10].tolist()
    out["t0"] = int(t_int[0])
    out["t_end"] = int(t_int[-1])
    out["n_samples"] = int(len(t_int))

    if len(bad) > 0:
        i0 = int(bad[0])
        out["first_bad_pair"] = {
            "idx": i0,
            "t_i": int(t_int[i0]),
            "t_next": int(t_int[i0 + 1]),
            "dt": int(dt[i0]),
        }

    return out
