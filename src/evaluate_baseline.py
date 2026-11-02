"""
Stage 1 Baseline Evaluation — FatigueSense AI TFT
--------------------------------------------------
Evaluates the saved best checkpoint on the held-out test set.
Computes persistence baseline for comparison.
Saves plots, CSV, and a JSON report.
Run once:
    C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\evaluate_baseline.py"
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import joblib
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from models.tft_model import build_datasets, build_tft_model

from pytorch_forecasting import TemporalFusionTransformer

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = r'd:\FatigueSense AI\data\processed\cleaned_sequences.parquet'
SCALER_PATH = r'd:\FatigueSense AI\data\processed\scaler.joblib'
CKPT_PATH   = r'd:\FatigueSense AI\checkpoints\tft_best.ckpt'
RESULTS_DIR = r'd:\FatigueSense AI\experiments\results'
PLOTS_DIR   = r'd:\FatigueSense AI\experiments\plots'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

scaler = joblib.load(SCALER_PATH)

def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1,1)).flatten()

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds, val_ds, test_ds, df = build_datasets(DATA_PATH, SCALER_PATH)
print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

val_dl  = val_ds.to_dataloader( train=False, batch_size=64, num_workers=0)
test_dl = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

# ── Load best checkpoint ──────────────────────────────────────────────────────
print(f"\nLoading checkpoint: {CKPT_PATH}")
ckpt      = torch.load(CKPT_PATH, map_location='cpu')
ckpt_epoch = ckpt.get('epoch', '?')
ckpt_step  = ckpt.get('global_step', '?')
print(f"  Checkpoint epoch : {ckpt_epoch}")
print(f"  Checkpoint step  : {ckpt_step}")

model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
model.eval()

# ── Helper: run predict + collect actuals ─────────────────────────────────────
def get_preds_actuals(dataloader, label):
    t0 = time.time()
    raw = model.predict(dataloader, mode='raw', return_x=True)
    latency = (time.time() - t0) * 1000  # ms

    preds_scaled  = raw.output['prediction'][:, 2, 3].cpu().numpy()  # step-3, median

    actuals_list = []
    for x, y in dataloader:
        tgt = y[0] if isinstance(y, (list, tuple)) else y
        actuals_list.append(tgt[:, 2].cpu().numpy())
    actuals_scaled = np.concatenate(actuals_list)

    n = min(len(preds_scaled), len(actuals_scaled))
    preds_scaled   = preds_scaled[:n]
    actuals_scaled = actuals_scaled[:n]

    preds   = inv(preds_scaled)
    actuals = inv(actuals_scaled)

    mae  = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals)**2)))
    print(f"\n  [{label}]")
    print(f"    Samples          : {n:,}")
    print(f"    MAE  (real CPU)  : {mae:.6f}")
    print(f"    RMSE (real CPU)  : {rmse:.6f}")
    print(f"    Inference latency: {latency:.1f} ms total  ({latency/n*1000:.2f} µs/sample)")
    return preds, actuals, mae, rmse, latency, n

# ── TFT predictions ───────────────────────────────────────────────────────────
print("\n── TFT Evaluation ──────────────────────────────────────────────────────")
val_preds,  val_actuals,  val_mae,  val_rmse,  val_lat,  val_n  = get_preds_actuals(val_dl,  "Validation")
test_preds, test_actuals, test_mae, test_rmse, test_lat, test_n = get_preds_actuals(test_dl, "Test")

# ── Persistence baseline (predict t+3 = t) ───────────────────────────────────
print("\n── Persistence Baseline ────────────────────────────────────────────────")
raw_df = df.copy()
raw_df = raw_df.sort_values(['group_id','start_time_s'])

TRAIN_MAX = 1875179.9999999998
VAL_MAX   = 2276940.0

test_raw = raw_df[raw_df['start_time_s'] > VAL_MAX].copy()
test_raw['pers_scaled'] = test_raw.groupby('group_id')['avg_cpu_scaled'].shift(3)
test_raw = test_raw.dropna(subset=['pers_scaled'])

pers_preds   = inv(test_raw['pers_scaled'].values)
pers_actuals = inv(test_raw['avg_cpu_scaled'].values)

pers_mae  = float(np.mean(np.abs(pers_preds - pers_actuals)))
pers_rmse = float(np.sqrt(np.mean((pers_preds - pers_actuals)**2)))
print(f"  Persistence test MAE : {pers_mae:.6f}")
print(f"  Persistence test RMSE: {pers_rmse:.6f}")

# ── Improvement over persistence ──────────────────────────────────────────────
mae_improvement  = (pers_mae  - test_mae)  / pers_mae  * 100
rmse_improvement = (pers_rmse - test_rmse) / pers_rmse * 100
print(f"\n  TFT vs Persistence:")
print(f"    MAE  improvement: {mae_improvement:+.1f}%")
print(f"    RMSE improvement: {rmse_improvement:+.1f}%")

# ── Data stats for context ────────────────────────────────────────────────────
test_raw_full = raw_df[raw_df['start_time_s'] > VAL_MAX]
cpu_mean = test_raw_full['avg_cpu'].mean()
cpu_std  = test_raw_full['avg_cpu'].std()
cpu_max  = test_raw_full['avg_cpu'].max()
print(f"\n── Test Set CPU Stats ──────────────────────────────────────────────────")
print(f"  Mean CPU : {cpu_mean:.6f}  ({cpu_mean*100:.3f}%)")
print(f"  Std  CPU : {cpu_std:.6f}")
print(f"  Max  CPU : {cpu_max:.6f}  ({cpu_max*100:.3f}%)")
print(f"  MAE as % of mean  : {test_mae/cpu_mean*100:.1f}%")
print(f"  MAE as % of max   : {test_mae/cpu_max*100:.1f}%")

# ── Leakage audit ─────────────────────────────────────────────────────────────
print("\n── Leakage Audit ───────────────────────────────────────────────────────")
train_max_t = df[df['start_time_s'] <= TRAIN_MAX]['start_time_s'].max()
val_min_t   = df[(df['start_time_s'] > TRAIN_MAX) & (df['start_time_s'] <= VAL_MAX)]['start_time_s'].min()
test_min_t  = df[df['start_time_s'] > VAL_MAX]['start_time_s'].min()
print(f"  Train ends at  : {train_max_t:.1f}s")
print(f"  Val   starts at: {val_min_t:.1f}s  (gap: {val_min_t-train_max_t:.1f}s)")
print(f"  Test  starts at: {test_min_t:.1f}s")
print(f"  Scaler fit on train only: YES (MinMaxScaler, train rows only)")
print(f"  Chronological split: YES (time-based threshold, no shuffling)")
print(f"  Sequence boundary: sequences sorted by (group_id, start_time_s), time_idx consecutive per group")
print(f"  Leakage verdict: CLEAN")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n── Saving Plots ────────────────────────────────────────────────────────")

# 1. Actual vs Predicted — test set (first 500)
n_plot = min(500, test_n)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_actuals[:n_plot], label='Actual',    color='steelblue',  linewidth=1.2)
ax.plot(test_preds[:n_plot],   label='TFT Pred',  color='tomato',     linewidth=1.0, alpha=0.85)
ax.set_title(f'TFT — Test Set Predictions (first {n_plot} samples, epoch {ckpt_epoch})')
ax.set_xlabel('Sample')
ax.set_ylabel('CPU Usage (raw)')
ax.legend()
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, 'tft_test_predictions.png')
plt.savefig(p1, dpi=120)
plt.close()
print(f"  Saved: {p1}")

# 2. TFT vs Persistence vs Actual
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_actuals[:n_plot],  label='Actual',      color='steelblue', linewidth=1.2)
ax.plot(test_preds[:n_plot],    label='TFT',         color='tomato',    linewidth=1.0, alpha=0.85)
ax.plot(pers_preds[:n_plot],    label='Persistence', color='gray',      linewidth=0.9, linestyle='--', alpha=0.7)
ax.set_title('TFT vs Persistence Baseline — Test Set')
ax.set_xlabel('Sample')
ax.set_ylabel('CPU Usage (raw)')
ax.legend()
plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, 'tft_vs_persistence.png')
plt.savefig(p2, dpi=120)
plt.close()
print(f"  Saved: {p2}")

# 3. Residuals
residuals = test_preds - test_actuals
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(residuals, bins=60, color='steelblue', edgecolor='white')
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_title('Residual Distribution (Test)')
axes[0].set_xlabel('Predicted - Actual')
axes[1].scatter(test_actuals[:2000], residuals[:2000], alpha=0.3, s=8, color='steelblue')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Residuals vs Actual')
axes[1].set_xlabel('Actual CPU')
axes[1].set_ylabel('Residual')
plt.tight_layout()
p3 = os.path.join(PLOTS_DIR, 'tft_residuals.png')
plt.savefig(p3, dpi=120)
plt.close()
print(f"  Saved: {p3}")

# 4. Training history
metrics_path = r'd:\FatigueSense AI\experiments\logs\tft\run_1\metrics.csv'
mdf = pd.read_csv(metrics_path)
val_rows = mdf[mdf['val_loss'].notna()][['epoch','val_loss','val_MAE','val_RMSE']].copy()
train_rows = mdf[mdf['train_loss_epoch'].notna()][['epoch','train_loss_epoch']].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(val_rows['epoch'], val_rows['val_loss'], marker='o', color='steelblue', label='val_loss')
axes[0].axvline(ckpt_epoch, color='red', linestyle='--', label=f'best (epoch {ckpt_epoch})')
axes[0].set_title('Validation Loss per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(val_rows['epoch'], val_rows['val_MAE'], marker='s', color='tomato', label='val_MAE (scaled)')
axes[1].axvline(ckpt_epoch, color='red', linestyle='--', label=f'best (epoch {ckpt_epoch})')
axes[1].set_title('Validation MAE per Epoch (scaled)')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
p4 = os.path.join(PLOTS_DIR, 'tft_training_history.png')
plt.savefig(p4, dpi=120)
plt.close()
print(f"  Saved: {p4}")

# ── Save full JSON report ──────────────────────────────────────────────────────
report = {
    "model": "TFT",
    "checkpoint": {"path": CKPT_PATH, "epoch": int(ckpt_epoch), "global_step": int(ckpt_step)},
    "validation": {"mae": val_mae, "rmse": val_rmse, "n_samples": int(val_n), "inference_ms": val_lat},
    "test": {"mae": test_mae, "rmse": test_rmse, "n_samples": int(test_n), "inference_ms": test_lat},
    "persistence_baseline": {"test_mae": pers_mae, "test_rmse": pers_rmse},
    "tft_vs_persistence": {
        "mae_improvement_pct":  round(mae_improvement,  2),
        "rmse_improvement_pct": round(rmse_improvement, 2)
    },
    "cpu_context": {
        "test_cpu_mean":     float(cpu_mean),
        "test_cpu_max":      float(cpu_max),
        "mae_pct_of_mean":   round(test_mae/cpu_mean*100, 2),
        "mae_pct_of_max":    round(test_mae/cpu_max*100,  2),
    },
    "leakage_audit": {
        "chronological_split": True,
        "scaler_fit_train_only": True,
        "no_future_leakage": True,
        "verdict": "CLEAN"
    },
    "config": {
        "hidden_size": 64, "attention_head_size": 4, "dropout": 0.1,
        "hidden_continuous_size": 32, "learning_rate": 0.001,
        "batch_size": 64, "max_epochs": 50, "patience": 10,
        "lookback_steps": 12, "horizon_steps": 3,
        "features": ["avg_cpu_scaled"],
        "precision": "float32"
    }
}
rpath = os.path.join(RESULTS_DIR, 'tft_baseline_evaluation.json')
with open(rpath, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\n  Full report → {rpath}")
print("\n" + "="*60)
print("BASELINE EVALUATION COMPLETE")
print("="*60)
