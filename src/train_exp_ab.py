"""
Experiment C+D — FatigueSense AI TFT (Multi-Feature, Extended Lookback)
========================================================================
Improvements over baseline (run_1):
  A: bf16-mixed precision         — BF16 avoids FP16 overflow in TFT attention mask
  B: batch_size=256                 — ~2,427 steps/epoch vs 9,708 baseline
  C: 4 features                     — avg_cpu, avg_memory, max_cpu, assigned_memory
  D: lookback 12 -> 24 steps        — 2 hours history vs 1 hour baseline

Baseline test results (for comparison):
  MAE : 0.000722  RMSE: 0.001530
  Persistence baseline MAE: 0.000862

Run once from your terminal:
    C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\train_exp_ab.py"

Checkpoints  : d:\FatigueSense AI\checkpoints\tft_exp_abcd.ckpt
Logs         : d:\FatigueSense AI\experiments\logs\tft\exp_abcd\metrics.csv
Results JSON : d:\FatigueSense AI\experiments\results\tft_exp_abcd_results.json
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import joblib
import torch

torch.set_float32_matmul_precision('medium')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).parent))
from models.tft_model import build_datasets, build_tft_model

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = r'd:\FatigueSense AI\data\processed\cleaned_sequences.parquet'
SCALER_PATH = r'd:\FatigueSense AI\data\processed\scaler.joblib'
CKPT_DIR    = r'd:\FatigueSense AI\checkpoints'
LOG_DIR     = r'd:\FatigueSense AI\experiments\logs'
RESULTS_DIR = r'd:\FatigueSense AI\experiments\results'
PLOTS_DIR   = r'd:\FatigueSense AI\experiments\plots'

for d in [CKPT_DIR, LOG_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EXP_NAME   = 'exp_abcd'
BATCH_SIZE = 256
PRECISION  = 'bf16-mixed'  # FP16 overflows in TFT attention mask (-1e9 > FP16 max ~65504)
MAX_EPOCHS = 50
PATIENCE   = 10
SEED       = 42

BASELINE_TEST_MAE  = 0.000722
BASELINE_TEST_RMSE = 0.001530
PERSISTENCE_MAE    = 0.000862

# ── Reproducibility ───────────────────────────────────────────────────────────
seed_everything(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print('Building datasets...')
train_dataset, val_dataset, test_dataset, df = build_datasets(DATA_PATH, SCALER_PATH)

steps_per_epoch = len(train_dataset) // BATCH_SIZE
est_min_per_epoch = steps_per_epoch / 14.5 / 60

print(f'  Train samples  : {len(train_dataset):,}')
print(f'  Val   samples  : {len(val_dataset):,}')
print(f'  Test  samples  : {len(test_dataset):,}')
print(f'  Batch size     : {BATCH_SIZE}')
print(f'  Steps/epoch    : ~{steps_per_epoch:,}  (baseline was 9,708)')
print(f'  Est. min/epoch : ~{est_min_per_epoch:.1f} min')

train_dataloader = train_dataset.to_dataloader(
    train=True,  batch_size=BATCH_SIZE, num_workers=0
)
val_dataloader = val_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)
test_dataloader = test_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)

# ── Model ─────────────────────────────────────────────────────────────────────
print('\nBuilding TFT model...')
model = build_tft_model(train_dataset)
n_params = sum(p.numel() for p in model.parameters())
print(f'  Parameters : {n_params:,}  (baseline was 250,833)')

# ── Callbacks ─────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    mode='min',
    verbose=True,
)

checkpoint_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename='tft_exp_abcd',        # never overwrites tft_best.ckpt (baseline)
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    auto_insert_metric_name=False,
)

csv_logger = CSVLogger(
    save_dir=LOG_DIR,
    name='tft',
    version=EXP_NAME,
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    gradient_clip_val=0.1,
    accelerator='gpu',
    devices=1,
    precision=PRECISION,
    callbacks=[early_stop, checkpoint_cb],
    logger=csv_logger,
    enable_progress_bar=True,
    log_every_n_steps=50,
)

# ── Training ──────────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('Experiment A+B+C+D')
print('  A: bf16-mixed precision')
print('  B: batch_size=256')
print('  C: features = avg_cpu + avg_memory + max_cpu + assigned_memory')
print('  D: lookback = 24 steps (2 hours)')
print(f'  Expected ~{est_min_per_epoch:.1f} min/epoch  |  max {MAX_EPOCHS} epochs')
print(f'  Baseline test MAE to beat: {BASELINE_TEST_MAE}')
print('=' * 65)
print()

start_time = time.time()
trainer.fit(model, train_dataloader, val_dataloader)
duration = time.time() - start_time

print(f'\nTraining complete in {duration/60:.1f} min')
print(f'Best checkpoint : {checkpoint_cb.best_model_path}')
print(f'Best val_loss   : {checkpoint_cb.best_model_score:.6f}')

# ── Load best model ───────────────────────────────────────────────────────────
best_ckpt  = checkpoint_cb.best_model_path
ckpt_data  = torch.load(best_ckpt, map_location='cpu')
best_epoch = int(ckpt_data.get('epoch', 0))

best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
best_model.eval()

# ── Inverse transform — target only (avg_cpu_scaled) ─────────────────────────
cpu_scaler = joblib.load(SCALER_PATH)

def inv_cpu(arr):
    return cpu_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

# ── Predict helper ────────────────────────────────────────────────────────────
def predict(dataloader):
    raw = best_model.predict(dataloader, mode='raw', return_x=True)
    # shape: (N, prediction_length=3, num_quantiles=7)
    # index [:, 2, 3] = step 3 (15 min), median quantile (0.5)
    preds_s = raw.output['prediction'][:, 2, 3].cpu().numpy()

    acts = []
    for x, y in dataloader:
        tgt = y[0] if isinstance(y, (list, tuple)) else y
        acts.append(tgt[:, 2].cpu().numpy())
    acts_s = np.concatenate(acts)

    n = min(len(preds_s), len(acts_s))
    preds   = inv_cpu(preds_s[:n])
    actuals = inv_cpu(acts_s[:n])
    mae  = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    return preds, actuals, mae, rmse

# ── Evaluate ──────────────────────────────────────────────────────────────────
print('\nEvaluating on validation set...')
val_preds, val_actuals, val_mae, val_rmse = predict(val_dataloader)

print('Evaluating on test set...')
test_preds, test_actuals, test_mae, test_rmse = predict(test_dataloader)

# ── Persistence baseline (on test, 3-step lag) ────────────────────────────────
VAL_MAX = 2276940.0
test_raw = df[df['start_time_s'] > VAL_MAX].sort_values(['group_id','start_time_s']).copy()
test_raw['pers'] = test_raw.groupby('group_id')['avg_cpu_scaled'].shift(3)
test_raw = test_raw.dropna(subset=['pers'])
pers_mae  = float(np.mean(np.abs(inv_cpu(test_raw['pers'].values) - inv_cpu(test_raw['avg_cpu_scaled'].values))))
pers_rmse = float(np.sqrt(np.mean((inv_cpu(test_raw['pers'].values) - inv_cpu(test_raw['avg_cpu_scaled'].values))**2)))

# ── Results table ─────────────────────────────────────────────────────────────
mae_vs_baseline   = (test_mae - BASELINE_TEST_MAE)  / BASELINE_TEST_MAE  * 100
rmse_vs_baseline  = (test_rmse - BASELINE_TEST_RMSE) / BASELINE_TEST_RMSE * 100
mae_vs_persistence = (test_mae - PERSISTENCE_MAE) / PERSISTENCE_MAE * 100

print()
print('=' * 65)
print('FINAL RESULTS')
print('=' * 65)
print(f'{"Model":<30} {"Test MAE":>12} {"Test RMSE":>12}')
print('-' * 65)
print(f'{"Persistence baseline":<30} {PERSISTENCE_MAE:>12.6f} {"—":>12}')
print(f'{"Baseline TFT (run_1)":<30} {BASELINE_TEST_MAE:>12.6f} {BASELINE_TEST_RMSE:>12.6f}')
print(f'{"This run (A+B+C+D)":<30} {test_mae:>12.6f} {test_rmse:>12.6f}')
print('-' * 65)
print(f'{"vs baseline TFT":<30} {mae_vs_baseline:>+11.1f}% {rmse_vs_baseline:>+11.1f}%')
print(f'{"vs persistence":<30} {mae_vs_persistence:>+11.1f}%')
print('=' * 65)

# ── Plots ─────────────────────────────────────────────────────────────────────
n_plot = min(500, len(test_actuals))

# 1. Predictions vs actual
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_actuals[:n_plot], label='Actual',         color='steelblue', linewidth=1.2)
ax.plot(test_preds[:n_plot],   label='TFT (A+B+C+D)',  color='tomato',    linewidth=1.0, alpha=0.85)
ax.set_title('TFT Multi-Feature — Test Set Predictions (first 500 samples)')
ax.set_xlabel('Sample')
ax.set_ylabel('CPU Usage (raw)')
ax.legend()
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, 'tft_exp_abcd_predictions.png')
plt.savefig(p1, dpi=120)
plt.close()

# 2. vs baseline vs persistence
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_actuals[:n_plot],               label='Actual',         color='steelblue', linewidth=1.2)
ax.plot(test_preds[:n_plot],                 label='TFT (A+B+C+D)',  color='tomato',    linewidth=1.0, alpha=0.85)
ax.plot(inv_cpu(test_raw['pers'].values[:n_plot]), label='Persistence', color='gray', linewidth=0.9, linestyle='--', alpha=0.7)
ax.set_title('TFT Multi-Feature vs Persistence — Test Set')
ax.set_xlabel('Sample')
ax.legend()
plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, 'tft_exp_abcd_vs_persistence.png')
plt.savefig(p2, dpi=120)
plt.close()

# 3. Training history vs baseline
metrics_path = os.path.join(LOG_DIR, 'tft', EXP_NAME, 'metrics.csv')
mdf = pd.read_csv(metrics_path)
val_rows = mdf[mdf['val_loss'].notna()][['epoch','val_loss','val_MAE']].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(val_rows['epoch'], val_rows['val_loss'], marker='o', color='steelblue', label='exp_abcd')
axes[0].axhline(0.004617, color='gray', linestyle='--', linewidth=1, label='baseline best')
axes[0].axvline(best_epoch, color='red', linestyle=':', label=f'best epoch {best_epoch}')
axes[0].set_title('val_loss: Exp A+B+C+D vs Baseline')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(val_rows['epoch'], val_rows['val_MAE'], marker='s', color='tomato', label='val_MAE (scaled)')
axes[1].axvline(best_epoch, color='red', linestyle=':')
axes[1].set_title('val_MAE per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
p3 = os.path.join(PLOTS_DIR, 'tft_exp_abcd_history.png')
plt.savefig(p3, dpi=120)
plt.close()

print(f'\nPlots saved:')
print(f'  {p1}')
print(f'  {p2}')
print(f'  {p3}')

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    'experiment': 'A+B+C+D',
    'description': {
        'A': '16-mixed precision',
        'B': 'batch_size=256',
        'C': 'features: avg_cpu + avg_memory + max_cpu + assigned_memory',
        'D': 'lookback=24 steps (2 hours)',
    },
    'checkpoint': best_ckpt,
    'best_epoch': best_epoch,
    'best_val_loss': float(checkpoint_cb.best_model_score),
    'training_duration_minutes': round(duration / 60, 1),
    'validation': {'mae': val_mae,  'rmse': val_rmse},
    'test':       {'mae': test_mae, 'rmse': test_rmse},
    'comparison': {
        'baseline_tft':   {'test_mae': BASELINE_TEST_MAE,  'test_rmse': BASELINE_TEST_RMSE},
        'persistence':    {'test_mae': PERSISTENCE_MAE},
        'mae_vs_baseline_pct':    round(mae_vs_baseline,   2),
        'rmse_vs_baseline_pct':   round(rmse_vs_baseline,  2),
        'mae_vs_persistence_pct': round(mae_vs_persistence, 2),
    },
    'config': {
        'hidden_size': 128,
        'attention_head_size': 4,
        'dropout': 0.1,
        'hidden_continuous_size': 64,
        'learning_rate': 0.001,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'lookback_steps': 24,
        'horizon_steps': 3,
        'precision': PRECISION,
        'seed': SEED,
        'features': ['avg_cpu_scaled', 'avg_memory_scaled',
                     'max_cpu_scaled', 'assigned_memory_scaled'],
    },
}

json_path = os.path.join(RESULTS_DIR, 'tft_exp_abcd_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults JSON → {json_path}')
print('\n' + json.dumps(results, indent=2))
