"""
windowing.py

Creates sliding windows from cleaned_sequences.parquet for time-series forecasting.

Window definition:
  - lookback = 12 steps (input: positions 0..11)
  - horizon  =  3 steps (target: position 14, i.e. input_end + 3)
  - Windows never cross run boundaries
  - Minimum run length required: lookback + horizon = 15 observations

Each window stores:
  instance_id      : f"{collection_id}_{instance_index}"
  run_id           : integer run counter (already present in parquet)
  window_start_time: start_time_s of the first input observation
  X                : float32 array of shape (12,)
  y                : float32 scalar (avg_cpu_scaled at input_end + 3)

Split assignment is based on window_start_time vs split_boundaries_seconds from config.
"""

import json
import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = pathlib.Path(r"d:\FatigueSense AI")
PARQUET_PATH  = BASE / "data" / "processed" / "cleaned_sequences.parquet"
CONFIG_PATH   = BASE / "configs" / "preprocessing_config.json"
OUT_DIR       = BASE / "data" / "processed"
META_PATH     = OUT_DIR / "windows_metadata.json"

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

LOOKBACK  = int(config["lookback_steps"])   # 12
HORIZON   = int(config["horizon_steps"])    # 3
TRAIN_END = float(config["split_boundaries_seconds"]["train_end"])
VAL_END   = float(config["split_boundaries_seconds"]["val_end"])

MIN_RUN_LEN = LOOKBACK + HORIZON  # 15

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading cleaned_sequences.parquet …")
df = pd.read_parquet(PARQUET_PATH)

# Ensure the data is sorted correctly within each (instance, run)
df = df.sort_values(["collection_id", "instance_index", "run_id", "start_time_s"])

# ---------------------------------------------------------------------------
# Build windows
# ---------------------------------------------------------------------------
all_X      = []
all_y      = []
all_times  = []
all_ids    = []

groups = df.groupby(["collection_id", "instance_index", "run_id"], sort=False)

for (coll_id, inst_idx, run), grp in groups:
    values = grp["avg_cpu_scaled"].to_numpy(dtype=np.float32)
    times  = grp["start_time_s"].to_numpy(dtype=np.float64)

    n = len(values)
    if n < MIN_RUN_LEN:
        continue  # not enough observations to form even one window

    instance_id = f"{coll_id}_{inst_idx}"

    # Slide over the run: last valid input window starts at index (n - MIN_RUN_LEN)
    # input  : values[i : i + LOOKBACK]          (indices i..i+11)
    # target : values[i + LOOKBACK - 1 + HORIZON] = values[i + 14]
    for i in range(n - MIN_RUN_LEN + 1):
        x = values[i : i + LOOKBACK]                  # shape (12,)
        y = values[i + LOOKBACK - 1 + HORIZON]        # scalar  (index i+14)
        t = times[i]                                   # window_start_time

        all_X.append(x)
        all_y.append(y)
        all_times.append(t)
        all_ids.append(instance_id)

print(f"Total windows created: {len(all_X)}")

# ---------------------------------------------------------------------------
# Convert to arrays
# ---------------------------------------------------------------------------
X_all     = np.stack(all_X)                          # (N, 12)
y_all     = np.array(all_y,     dtype=np.float32)    # (N,)
t_all     = np.array(all_times, dtype=np.float64)    # (N,)
ids_all   = np.array(all_ids,   dtype=object)        # (N,) string

# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------
train_mask = t_all < TRAIN_END
val_mask   = (t_all >= TRAIN_END) & (t_all < VAL_END)
test_mask  = t_all >= VAL_END

n_train = int(train_mask.sum())
n_val   = int(val_mask.sum())
n_test  = int(test_mask.sum())
n_total = len(X_all)

print(f"  train: {n_train}")
print(f"  val  : {n_val}")
print(f"  test : {n_test}")
print(f"  sum  : {n_train + n_val + n_test}  (should equal {n_total})")

# ---------------------------------------------------------------------------
# Save .npz files
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_split(path, mask):
    np.savez(
        path,
        X            = X_all[mask],
        y            = y_all[mask],
        times        = t_all[mask],
        instance_ids = ids_all[mask],
    )
    print(f"Saved {path.name}  ({mask.sum()} windows)")

save_split(OUT_DIR / "windows_train.npz", train_mask)
save_split(OUT_DIR / "windows_val.npz",   val_mask)
save_split(OUT_DIR / "windows_test.npz",  test_mask)

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
metadata = {
    "total_windows"  : n_total,
    "train_windows"  : n_train,
    "val_windows"    : n_val,
    "test_windows"   : n_test,
    "lookback_steps" : LOOKBACK,
    "horizon_steps"  : HORIZON,
    "feature"        : "avg_cpu_scaled",
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved {META_PATH.name}")
print("Done.")
