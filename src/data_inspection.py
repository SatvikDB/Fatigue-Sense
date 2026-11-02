"""
data_inspection.py
------------------
Inspects d:\FatigueSense AI\data\raw\google_cluster_2019.csv and reports
dataset statistics relevant to the FatigueSense AI forecasting task.

All findings are also saved to:
  d:\FatigueSense AI\experiments\results\inspection_summary.json
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = Path(r"d:\FatigueSense AI\data\raw\google_cluster_2019.csv")
OUTPUT_PATH = Path(r"d:\FatigueSense AI\experiments\results\inspection_summary.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── Forecasting parameters ───────────────────────────────────────────────────
LOOKBACK = 12       # required consecutive windows for the input sequence
HORIZON = 3         # required consecutive windows beyond lookback
TOTAL_SEQ = LOOKBACK + HORIZON   # 15 consecutive windows needed end-to-end
GAP_THRESHOLD = 310  # seconds; consecutive if next_start - prev_start <= this

print("=" * 70)
print("FatigueSense AI — Dataset Inspection")
print(f"File : {DATA_PATH}")
print("=" * 70)

# ─── 1. Load data ─────────────────────────────────────────────────────────────
print("\n[1] Loading CSV …")
df = pd.read_csv(DATA_PATH)
total_rows = len(df)
print(f"    Total rows : {total_rows:,}")
print(f"    Columns    : {list(df.columns)}")

# ─── 2. Unique instances and machines ─────────────────────────────────────────
unique_instances = df[["collection_id", "instance_index"]].drop_duplicates()
n_unique_instances = len(unique_instances)
n_unique_machines = df["machine_id"].nunique()

print(f"\n[2] Unique instances (collection_id + instance_index) : {n_unique_instances:,}")
print(f"    Unique machines                                     : {n_unique_machines:,}")

# ─── 3. Convert start_time from microseconds to seconds ───────────────────────
print("\n[3] Converting start_time from microseconds to seconds …")
df["start_time_s"] = df["start_time"] / 1e6
print(f"    start_time range (s): {df['start_time_s'].min():.0f} – {df['start_time_s'].max():.0f}")
print(f"    span: {(df['start_time_s'].max() - df['start_time_s'].min()) / 86400:.1f} days")

# ─── 4. Per-instance statistics ───────────────────────────────────────────────
print("\n[4] Computing per-instance statistics …")

def instance_stats(grp: pd.DataFrame) -> pd.Series:
    """Return stats for a single (collection_id, instance_index) group."""
    times = grp["start_time_s"].sort_values().values  # sorted seconds
    n = len(times)
    span_days = (times[-1] - times[0]) / 86400 if n > 1 else 0.0

    # consecutive windows: gap between successive start_times <= GAP_THRESHOLD
    if n == 1:
        # A single observation counts as a run of 1
        n_consec_windows = 1
        longest_run = 1
    else:
        diffs = np.diff(times)
        is_consec = diffs <= GAP_THRESHOLD   # bool array of length n-1

        # Count transitions and run lengths
        # Build run-length encoding over the sequence of observations
        # A "consecutive run" starts fresh whenever is_consec[i] is False
        run_length = 1
        longest_run = 1
        total_consec = 1
        for c in is_consec:
            if c:
                run_length += 1
                if run_length > longest_run:
                    longest_run = run_length
            else:
                run_length = 1
        # n_consec_windows: count of observations that belong to runs >= 2
        # For simplicity we report longest_run; separate counter not needed
        # Recount: number of windows that can serve as part of ANY consec seq
        n_consec_windows = int(np.sum(is_consec)) + 1 if np.any(is_consec) else 1

    return pd.Series({
        "record_count": n,
        "span_days": span_days,
        "n_consec_windows": n_consec_windows,
        "longest_run": longest_run,
    })

grouped = df.groupby(["collection_id", "instance_index"], sort=False)
stats = grouped.apply(instance_stats, include_groups=False).reset_index()

# ─── 5. Distribution of observation counts per instance ───────────────────────
print("\n[5] Distribution of observation counts per instance:")
rec = stats["record_count"]
pcts = np.percentile(rec, [1, 5, 10, 25, 50, 75, 90, 95, 99])
print(f"    Min    : {int(rec.min())}")
print(f"    P1     : {int(pcts[0])}")
print(f"    P5     : {int(pcts[1])}")
print(f"    P10    : {int(pcts[2])}")
print(f"    P25    : {int(pcts[3])}")
print(f"    Median : {int(pcts[4])}")
print(f"    Mean   : {rec.mean():.2f}")
print(f"    P75    : {int(pcts[5])}")
print(f"    P90    : {int(pcts[6])}")
print(f"    P95    : {int(pcts[7])}")
print(f"    P99    : {int(pcts[8])}")
print(f"    Max    : {int(rec.max())}")

# ─── 6. Instances with >= 12 consecutive windows ──────────────────────────────
n_ge12 = int((stats["longest_run"] >= LOOKBACK).sum())
print(f"\n[6] Instances with >= {LOOKBACK} consecutive windows : {n_ge12:,}")

# ─── 7. Instances with >= 15 consecutive windows (lookback+horizon) ───────────
n_ge15 = int((stats["longest_run"] >= TOTAL_SEQ).sum())
print(f"\n[7] Instances with >= {TOTAL_SEQ} consecutive windows (lookback+horizon) : {n_ge15:,}")

# ─── 8. Valid training windows (sliding window count) ─────────────────────────
print(f"\n[8] Counting valid training windows (horizon={HORIZON}, lookback={LOOKBACK}) …")

def count_training_windows(times_s: np.ndarray, lookback: int, horizon: int, gap: float) -> int:
    """
    Count how many start positions i exist in a sorted time array such that
    there is a valid consecutive run of length >= (lookback + horizon)
    starting at or containing position i.

    Equivalently: for each maximal consecutive run of length L,
    the number of valid (lookback, horizon) windows = max(0, L - lookback - horizon + 1).
    """
    if len(times_s) < lookback + horizon:
        return 0
    times_s = np.sort(times_s)
    diffs = np.diff(times_s)
    is_consec = diffs <= gap

    # Run-length encoding
    total_windows = 0
    run_len = 1
    for c in is_consec:
        if c:
            run_len += 1
        else:
            # end of run
            w = max(0, run_len - (lookback + horizon) + 1)
            total_windows += w
            run_len = 1
    # last run
    w = max(0, run_len - (lookback + horizon) + 1)
    total_windows += w
    return total_windows

all_windows = 0
for (cid, iidx), grp in grouped:
    times = grp["start_time_s"].values
    all_windows += count_training_windows(times, LOOKBACK, HORIZON, GAP_THRESHOLD)

print(f"    Valid training windows (lookback={LOOKBACK}, horizon={HORIZON}) : {all_windows:,}")

# ─── 9. Save summary JSON ─────────────────────────────────────────────────────
summary = {
    "file": str(DATA_PATH),
    "total_rows": total_rows,
    "unique_instances": n_unique_instances,
    "unique_machines": n_unique_machines,
    "start_time_range_seconds": {
        "min": float(df["start_time_s"].min()),
        "max": float(df["start_time_s"].max()),
        "span_days": float((df["start_time_s"].max() - df["start_time_s"].min()) / 86400),
    },
    "observation_count_distribution": {
        "min": int(rec.min()),
        "p1": int(pcts[0]),
        "p5": int(pcts[1]),
        "p10": int(pcts[2]),
        "p25": int(pcts[3]),
        "median": int(pcts[4]),
        "mean": float(rec.mean()),
        "p75": int(pcts[5]),
        "p90": int(pcts[6]),
        "p95": int(pcts[7]),
        "p99": int(pcts[8]),
        "max": int(rec.max()),
    },
    "forecasting_params": {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "gap_threshold_seconds": GAP_THRESHOLD,
    },
    "instances_ge_lookback": n_ge12,
    "instances_ge_total_seq": n_ge15,
    "valid_training_windows": all_windows,
    "longest_run_distribution": {
        "min": int(stats["longest_run"].min()),
        "max": int(stats["longest_run"].max()),
        "mean": float(stats["longest_run"].mean()),
        "median": float(stats["longest_run"].median()),
        "p90": float(np.percentile(stats["longest_run"], 90)),
        "p95": float(np.percentile(stats["longest_run"], 95)),
        "p99": float(np.percentile(stats["longest_run"], 99)),
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[9] Summary saved to: {OUTPUT_PATH}")

# ─── 10. Final summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total rows                          : {total_rows:,}")
print(f"  Unique instances                    : {n_unique_instances:,}")
print(f"  Unique machines                     : {n_unique_machines:,}")
print(f"  Trace span                          : {(df['start_time_s'].max() - df['start_time_s'].min()) / 86400:.1f} days")
print(f"  Obs per instance — min/mean/median/max: "
      f"{int(rec.min())} / {rec.mean():.1f} / {int(pcts[4])} / {int(rec.max())}")
print(f"  Instances with >= {LOOKBACK} consec windows : {n_ge12:,}")
print(f"  Instances with >= {TOTAL_SEQ} consec windows : {n_ge15:,}")
print(f"  Valid training windows              : {all_windows:,}")
print("=" * 70)
