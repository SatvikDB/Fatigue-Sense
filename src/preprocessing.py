"""
preprocessing.py
----------------
Preprocessing pipeline for FatigueSense AI.

Steps:
  1. Read raw CSV
  2. Convert timestamps from microseconds -> seconds
  3. Sort each instance chronologically
  4. Validate windows: keep only rows where (end_time_s - start_time_s) == 300 s (±0.001 s)
  5. Identify consecutive runs (gap between consecutive start_times == 300 ± 1 s)
  6. Filter runs shorter than 15 observations
  7. Report survival / drop stats
  8. MinMaxScaler on avg_cpu (global time-split: 70/15/15)
  9. Save cleaned_sequences.parquet, scaler.joblib, preprocessing_config.json
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV   = os.path.join(ROOT, "data", "raw",       "google_cluster_2019.csv")
OUT_DIR   = os.path.join(ROOT, "data", "processed")
CFG_DIR   = os.path.join(ROOT, "configs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)

OUT_PARQUET = os.path.join(OUT_DIR, "cleaned_sequences.parquet")
OUT_SCALER  = os.path.join(OUT_DIR, "scaler.joblib")
OUT_CONFIG  = os.path.join(CFG_DIR, "preprocessing_config.json")

# ---------------------------------------------------------------------------
# Step 1 — Read raw CSV
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — Reading raw CSV …")
df = pd.read_csv(RAW_CSV)
rows_raw = len(df)
print(f"  Rows read: {rows_raw:,}")
print(f"  Columns  : {list(df.columns)}")

# ---------------------------------------------------------------------------
# Step 2 — Convert timestamps from microseconds to seconds
# ---------------------------------------------------------------------------
print("\nSTEP 2 — Converting timestamps µs → s …")
df["start_time_s"] = df["start_time"] / 1e6
df["end_time_s"]   = df["end_time"]   / 1e6
print(f"  start_time_s range: {df['start_time_s'].min():.1f} s  –  {df['start_time_s'].max():.1f} s")
print(f"  end_time_s   range: {df['end_time_s'].min():.1f} s  –  {df['end_time_s'].max():.1f} s")

# ---------------------------------------------------------------------------
# Step 3 — Sort each instance chronologically
# ---------------------------------------------------------------------------
print("\nSTEP 3 — Sorting by (collection_id, instance_index, start_time_s) …")
df = df.sort_values(["collection_id", "instance_index", "start_time_s"]).reset_index(drop=True)
print(f"  Done. {len(df):,} rows.")

# ---------------------------------------------------------------------------
# Step 4 — Validate windows: keep rows where window == 300 s (±0.001 s)
# ---------------------------------------------------------------------------
print("\nSTEP 4 — Validating window durations …")
window_duration = df["end_time_s"] - df["start_time_s"]
WINDOW_TOL = 0.001
mask_valid_window = (window_duration - 300.0).abs() <= WINDOW_TOL
rows_after_window = mask_valid_window.sum()
rows_dropped_window = rows_raw - rows_after_window
print(f"  Rows with 300 s window  : {rows_after_window:,}")
print(f"  Rows dropped (bad window): {rows_dropped_window:,}")
df = df[mask_valid_window].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Step 5 — Identify consecutive runs within each instance
# ---------------------------------------------------------------------------
print("\nSTEP 5 — Identifying consecutive runs (gap == 300 ± 1 s) …")
STEP   = 300.0   # seconds between consecutive windows
STEP_TOL = 1.0   # tolerance in seconds

# Within each (collection_id, instance_index) group, compute the gap to the
# previous row.  Rows where the gap is outside [299, 301] start a new run.
df = df.sort_values(["collection_id", "instance_index", "start_time_s"]).reset_index(drop=True)

# Compute gap to previous row within each instance
df["_prev_start"] = df.groupby(["collection_id", "instance_index"])["start_time_s"].shift(1)
df["_gap"]        = df["start_time_s"] - df["_prev_start"]

# A new run begins at:
#  - the first row of an instance (gap is NaN)
#  - any row where the gap is not in [299, 301]
is_new_run = (df["_gap"].isna()) | ((df["_gap"] - STEP).abs() > STEP_TOL)
df["run_id"] = is_new_run.cumsum()

# Drop helper columns
df.drop(columns=["_prev_start", "_gap"], inplace=True)

n_runs_before = df["run_id"].nunique()
print(f"  Total runs before length filter: {n_runs_before:,}")

# ---------------------------------------------------------------------------
# Step 6 — Filter runs shorter than 15 observations
# ---------------------------------------------------------------------------
print("\nSTEP 6 — Filtering runs with < 15 observations …")
MIN_RUN = 15
run_lengths = df.groupby("run_id")["run_id"].transform("count")
mask_long  = run_lengths >= MIN_RUN
rows_before_filter = len(df)
df = df[mask_long].reset_index(drop=True)
rows_after_filter  = len(df)
rows_dropped_short = rows_before_filter - rows_after_filter

n_runs_after    = df["run_id"].nunique()
n_instances_after = df.groupby(["collection_id", "instance_index"]).ngroups
print(f"  Rows dropped (short runs): {rows_dropped_short:,}")
print(f"  Rows retained            : {rows_after_filter:,}")
print(f"  Runs  retained           : {n_runs_after:,}")
print(f"  Instances retained       : {n_instances_after:,}")

# ---------------------------------------------------------------------------
# Step 7 — Survival / drop summary
# ---------------------------------------------------------------------------
print("\nSTEP 7 — Drop summary:")
print(f"  Raw rows                         : {rows_raw:,}")
print(f"  Dropped — bad window duration    : {rows_dropped_window:,}")
print(f"  Dropped — short runs (<{MIN_RUN} obs) : {rows_dropped_short:,}")
print(f"  Retained rows                    : {rows_after_filter:,}  "
      f"({100*rows_after_filter/rows_raw:.1f}% of raw)")

# ---------------------------------------------------------------------------
# Step 8 — MinMaxScaler on avg_cpu with global chronological 70/15/15 split
# ---------------------------------------------------------------------------
print("\nSTEP 8 — Applying MinMaxScaler to avg_cpu …")

t_min = df["start_time_s"].min()
t_max = df["start_time_s"].max()
t_range = t_max - t_min

train_end_s = t_min + 0.70 * t_range
val_end_s   = t_min + 0.85 * t_range

print(f"  Time range  : {t_min:.1f} s  –  {t_max:.1f} s  ({t_range/86400:.2f} days)")
print(f"  Train end   : {train_end_s:.1f} s  (first 70 %)")
print(f"  Val end     : {val_end_s:.1f} s   (next 15 %)")
print(f"  Test start  : {val_end_s:.1f} s   (last 15 %)")

# Assign split labels
conditions = [
    df["start_time_s"] <= train_end_s,
    df["start_time_s"] <= val_end_s,
]
choices = ["train", "val"]
df["split"] = np.select(conditions, choices, default="test")

# Fit scaler ONLY on training rows
train_mask = df["split"] == "train"
scaler = MinMaxScaler()
scaler.fit(df.loc[train_mask, ["avg_cpu"]])

# Transform all rows
df["avg_cpu_scaled"] = scaler.transform(df[["avg_cpu"]])

# Split size report
split_counts = df["split"].value_counts().reindex(["train", "val", "test"])
print(f"\n  Split sizes:")
for split_name, cnt in split_counts.items():
    pct = 100 * cnt / len(df)
    print(f"    {split_name:5s}: {cnt:,} rows  ({pct:.1f} %)")

# ---------------------------------------------------------------------------
# Step 9 — Save cleaned_sequences.parquet
# ---------------------------------------------------------------------------
print("\nSTEP 9 — Saving parquet …")
df.to_parquet(OUT_PARQUET, index=False, engine="fastparquet")
parquet_mb = os.path.getsize(OUT_PARQUET) / 1e6
print(f"  Saved: {OUT_PARQUET}")
print(f"  File size: {parquet_mb:.2f} MB")

# ---------------------------------------------------------------------------
# Step 10 — Save scaler
# ---------------------------------------------------------------------------
print("\nSTEP 10 — Saving scaler …")
joblib.dump(scaler, OUT_SCALER)
print(f"  Saved: {OUT_SCALER}")

# ---------------------------------------------------------------------------
# Step 11 — Save preprocessing_config.json
# ---------------------------------------------------------------------------
print("\nSTEP 11 — Saving config …")
config = {
    "split_boundaries_seconds": {
        "train_end": float(train_end_s),
        "val_end":   float(val_end_s),
    },
    "scaler_type": "MinMaxScaler",
    "window_size_seconds": 300,
    "horizon_steps": 3,
    "lookback_steps": 12,
    "sampling_interval_seconds": 300,
}
with open(OUT_CONFIG, "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved: {OUT_CONFIG}")
print(f"  Config: {json.dumps(config, indent=4)}")

# ---------------------------------------------------------------------------
# Step 12 — Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Input rows                      : {rows_raw:,}")
print(f"  Dropped (bad window)            : {rows_dropped_window:,}")
print(f"  Dropped (run < {MIN_RUN} obs)       : {rows_dropped_short:,}")
print(f"  Retained rows                   : {rows_after_filter:,}")
print(f"  Retained instances              : {n_instances_after:,}")
print(f"  Retained runs                   : {n_runs_after:,}")
print(f"  avg_cpu_scaled min/max (train)  : "
      f"{df.loc[train_mask,'avg_cpu_scaled'].min():.4f} / "
      f"{df.loc[train_mask,'avg_cpu_scaled'].max():.4f}")
print(f"  Split — train : {split_counts['train']:,} rows")
print(f"  Split — val   : {split_counts['val']:,} rows")
print(f"  Split — test  : {split_counts['test']:,} rows")
print(f"  Output parquet: {OUT_PARQUET}")
print(f"  Output scaler : {OUT_SCALER}")
print(f"  Output config : {OUT_CONFIG}")
print("=" * 60)
