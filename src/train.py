"""
Training script for the Temporal Fusion Transformer (TFT) model.

Run once from your terminal:
    C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\train.py"

Do NOT run multiple times concurrently — the checkpoint and log paths are fixed
so a second run would overwrite the first, not add a new version.
"""

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import joblib
import torch

# Silence the "set float32 matmul precision" warning and get free speed
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = r'd:\FatigueSense AI\data\processed\cleaned_sequences.parquet'
SCALER_PATH = r'd:\FatigueSense AI\data\processed\scaler.joblib'
CKPT_DIR    = r'd:\FatigueSense AI\checkpoints'
LOG_DIR     = r'd:\FatigueSense AI\experiments\logs'
RESULTS_DIR = r'd:\FatigueSense AI\experiments\results'
PLOTS_DIR   = r'd:\FatigueSense AI\experiments\plots'

for d in [CKPT_DIR, LOG_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
seed_everything(42)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print('Building datasets...')
train_dataset, val_dataset, test_dataset, df = build_datasets(DATA_PATH, SCALER_PATH)
print(f'  Train: {len(train_dataset):,} samples')
print(f'  Val  : {len(val_dataset):,} samples')
print(f'  Test : {len(test_dataset):,} samples')

train_dataloader = train_dataset.to_dataloader(
    train=True, batch_size=64, num_workers=0
)
val_dataloader = val_dataset.to_dataloader(
    train=False, batch_size=64, num_workers=0
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
print('Building TFT model...')
model = build_tft_model(train_dataset)
print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

# ---------------------------------------------------------------------------
# Callbacks & logger
# ---------------------------------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    verbose=True,
)

checkpoint_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename='tft_best',           # always writes tft_best.ckpt, never tft_best-v1
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    auto_insert_metric_name=False,  # prevents appending val_loss= to filename
)

# version_name fixes the folder to tft/run_1 — no auto-increment to version_0/1/2
csv_logger = CSVLogger(
    save_dir=LOG_DIR,
    name='tft',
    version='run_1',
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    max_epochs=50,
    gradient_clip_val=0.1,
    accelerator='gpu',
    devices=1,
    callbacks=[early_stop, checkpoint_cb],
    logger=csv_logger,
    enable_progress_bar=True,
    log_every_n_steps=50,       # only log every 50 steps — reduces console noise
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('Starting TFT training  (max 50 epochs, early stop patience=10)')
print('Each epoch ≈ 9,700 batches @ ~14 it/s  →  ~11 min/epoch')
print('Watch val_loss column for convergence.')
print('='*60 + '\n')

start_time = time.time()
trainer.fit(model, train_dataloader, val_dataloader)
duration = time.time() - start_time

print(f'\nTraining finished in {duration/60:.1f} min')

# ---------------------------------------------------------------------------
# Load best checkpoint
# ---------------------------------------------------------------------------
best_ckpt = checkpoint_cb.best_model_path
print(f'Best checkpoint : {best_ckpt}')
print(f'Best val_loss   : {checkpoint_cb.best_model_score:.6f}')

best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
best_model.eval()

# ---------------------------------------------------------------------------
# Predictions on validation set
# ---------------------------------------------------------------------------
print('\nRunning predictions on validation set...')
raw_predictions = best_model.predict(
    val_dataloader, mode='raw', return_x=True
)

# output["prediction"] → (N, prediction_length=3, num_quantiles=7)
# index 3 = 0.5 quantile (median); index 2 = step 3 of the 3-step horizon
preds_tensor  = raw_predictions.output['prediction']   # (N, 3, 7)
pred_scaled   = preds_tensor[:, 2, 3].cpu().numpy()    # step-3, median

# Actuals from val dataloader
actuals_list = []
for x, y in val_dataloader:
    target = y[0] if isinstance(y, (list, tuple)) else y
    actuals_list.append(target[:, 2].cpu().numpy())
actual_scaled = np.concatenate(actuals_list)

min_len       = min(len(pred_scaled), len(actual_scaled))
pred_scaled   = pred_scaled[:min_len]
actual_scaled = actual_scaled[:min_len]

# ---------------------------------------------------------------------------
# Inverse-transform to original CPU units
# ---------------------------------------------------------------------------
scaler    = joblib.load(SCALER_PATH)
predicted = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
actual    = scaler.inverse_transform(actual_scaled.reshape(-1, 1)).flatten()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
mae  = float(np.mean(np.abs(predicted - actual)))
rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
print(f'\nValidation MAE : {mae:.6f}')
print(f'Validation RMSE: {rmse:.6f}')

# ---------------------------------------------------------------------------
# Save predictions CSV
# ---------------------------------------------------------------------------
pred_csv = os.path.join(RESULTS_DIR, 'tft_val_predictions.csv')
pd.DataFrame({'predicted': predicted, 'actual': actual}).to_csv(pred_csv, index=False)
print(f'Predictions → {pred_csv}')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n_plot = min(500, len(actual))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(actual[:n_plot],    label='Actual')
ax.plot(predicted[:n_plot], label='Predicted')
ax.set_title('TFT Validation Predictions (first 500 points)')
ax.set_xlabel('Sample')
ax.set_ylabel('CPU Usage')
ax.legend()
plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, 'tft_val_predictions.png')
plt.savefig(plot_path, dpi=100)
plt.close()
print(f'Plot        → {plot_path}')

# ---------------------------------------------------------------------------
# Best epoch
# ---------------------------------------------------------------------------
best_epoch = trainer.current_epoch
try:
    best_epoch = int(checkpoint_cb.best_model_score.item())
except Exception:
    pass

# ---------------------------------------------------------------------------
# Save results JSON
# ---------------------------------------------------------------------------
results = {
    'model': 'TFT',
    'horizon': 3,
    'val_mae': mae,
    'val_rmse': rmse,
    'best_epoch': best_epoch,
    'training_duration_seconds': duration,
    'config': {
        'hidden_size': 64,
        'attention_head_size': 4,
        'dropout': 0.1,
        'hidden_continuous_size': 32,
        'learning_rate': 0.001,
        'batch_size': 64,
        'max_epochs': 50,
        'patience': 10,
        'lookback_steps': 12,
        'horizon_steps': 3,
    },
}
json_path = os.path.join(RESULTS_DIR, 'tft_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results JSON → {json_path}')
print('\n' + json.dumps(results, indent=2))
