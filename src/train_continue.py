"""
TFT Continuation Training — FatigueSense AI
============================================
Resumes from tft_exp_abcd_epoch41_SAFE.ckpt (best model so far).
Trains for up to 100 total epochs (50 more from epoch 49).
Patience increased 10 -> 15 to allow the model more room to improve.

SAFE checkpoint (never overwritten):
    d:\FatigueSense AI\checkpoints\tft_exp_abcd_epoch41_SAFE.ckpt

New best checkpoint (updated during this run):
    d:\FatigueSense AI\checkpoints\tft_continued.ckpt

Run once from your terminal:
    C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\train_continue.py"
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
DATA_PATH    = r'd:\FatigueSense AI\data\processed\cleaned_sequences.parquet'
SCALER_PATH  = r'd:\FatigueSense AI\data\processed\scaler.joblib'
RESUME_CKPT  = r'd:\FatigueSense AI\checkpoints\tft_exp_abcd_epoch41_SAFE.ckpt'
CKPT_DIR     = r'd:\FatigueSense AI\checkpoints'
LOG_DIR      = r'd:\FatigueSense AI\experiments\logs'
RESULTS_DIR  = r'd:\FatigueSense AI\experiments\results'
PLOTS_DIR    = r'd:\FatigueSense AI\experiments\plots'

for d in [CKPT_DIR, LOG_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EXP_NAME   = 'exp_continued'
BATCH_SIZE = 256
PRECISION  = 'bf16-mixed'
MAX_EPOCHS = 100        # 50 more on top of the 50 already done
PATIENCE   = 15         # increased from 10 — more room to find improvements
SEED       = 42

# Previous best results to compare against
PREV_BEST_VAL_LOSS = 0.003515
PREV_TEST_MAE      = 0.000665
PREV_TEST_RMSE     = 0.001503
BASELINE_TEST_MAE  = 0.000722
PERSISTENCE_MAE    = 0.000862

# ── Reproducibility ───────────────────────────────────────────────────────────
seed_everything(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print('Building datasets...')
train_dataset, val_dataset, test_dataset, df = build_datasets(DATA_PATH, SCALER_PATH)

steps_per_epoch = len(train_dataset) // BATCH_SIZE
print(f'  Train samples  : {len(train_dataset):,}')
print(f'  Val   samples  : {len(val_dataset):,}')
print(f'  Steps/epoch    : ~{steps_per_epoch:,}')
print(f'  Resuming from  : {RESUME_CKPT}')
print(f'  Max epochs     : {MAX_EPOCHS}  (was 50, adding up to 50 more)')
print(f'  Patience       : {PATIENCE}  (was 10)')

train_dataloader = train_dataset.to_dataloader(
    train=True,  batch_size=BATCH_SIZE, num_workers=0
)
val_dataloader = val_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)
test_dataloader = test_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)

# ── Model — load from checkpoint ──────────────────────────────────────────────
print('\nLoading model from checkpoint...')
model = TemporalFusionTransformer.load_from_checkpoint(RESUME_CKPT)
resume_epoch = torch.load(RESUME_CKPT, map_location='cpu').get('epoch', 0)
print(f'  Resumed from epoch : {resume_epoch}')
print(f'  Parameters         : {sum(p.numel() for p in model.parameters()):,}')

# ── Callbacks ─────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    mode='min',
    verbose=True,
)

checkpoint_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename='tft_continued',       # separate file — never touches the SAFE backup
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    auto_insert_metric_name=False,
)

csv_logger = CSVLogger(
    save_dir=LOG_DIR,
    name='tft',
    version=EXP_NAME,               # logs/tft/exp_continued/metrics.csv
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

# ── Training — resume from checkpoint ─────────────────────────────────────────
print()
print('=' * 65)
print('Continuation Training')
print(f'  Resuming from epoch  : {resume_epoch}')
print(f'  Previous best val_loss: {PREV_BEST_VAL_LOSS}  (epoch 41)')
print(f'  Previous test MAE     : {PREV_TEST_MAE}')
print(f'  Max epochs            : {MAX_EPOCHS}')
print(f'  Patience              : {PATIENCE}')
print(f'  Features              : avg_cpu + avg_memory + max_cpu + assigned_memory')
print(f'  Lookback              : 24 steps (2 hours)')
print('=' * 65)
print()

start_time = time.time()
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=RESUME_CKPT)
duration = time.time() - start_time

print(f'\nTraining complete in {duration/60:.1f} min')
print(f'Best checkpoint : {checkpoint_cb.best_model_path}')
print(f'Best val_loss   : {checkpoint_cb.best_model_score:.6f}')
print(f'Previous best   : {PREV_BEST_VAL_LOSS:.6f}')
improved = float(checkpoint_cb.best_model_score) < PREV_BEST_VAL_LOSS
print(f'Improved over previous best: {"YES" if improved else "NO"}')

# ── Load best model ───────────────────────────────────────────────────────────
best_ckpt  = checkpoint_cb.best_model_path
ckpt_data  = torch.load(best_ckpt, map_location='cpu')
best_epoch = int(ckpt_data.get('epoch', 0))

best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
best_model.eval()

cpu_scaler = joblib.load(SCALER_PATH)

def inv_cpu(arr):
    return cpu_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

def predict(dataloader):
    raw = best_model.predict(dataloader, mode='raw', return_x=True)
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

# Persistence
VAL_MAX = 2276940.0
test_raw = df[df['start_time_s'] > VAL_MAX].sort_values(['group_id','start_time_s']).copy()
test_raw['pers'] = test_raw.groupby('group_id')['avg_cpu_scaled'].shift(3)
test_raw = test_raw.dropna(subset=['pers'])
pers_preds_cpu   = inv_cpu(test_raw['pers'].values)
pers_actuals_cpu = inv_cpu(test_raw['avg_cpu_scaled'].values)

mae_vs_prev       = (test_mae - PREV_TEST_MAE)      / PREV_TEST_MAE      * 100
mae_vs_baseline   = (test_mae - BASELINE_TEST_MAE)  / BASELINE_TEST_MAE  * 100
mae_vs_persistence = (test_mae - PERSISTENCE_MAE)   / PERSISTENCE_MAE    * 100

# ── Results ───────────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('FINAL RESULTS')
print('=' * 65)
print(f'{"Model":<35} {"Test MAE":>10} {"Test RMSE":>10}')
print('-' * 65)
print(f'{"Persistence baseline":<35} {PERSISTENCE_MAE:>10.6f} {"—":>10}')
print(f'{"Baseline TFT (1 feature)":<35} {BASELINE_TEST_MAE:>10.6f} {0.001530:>10.6f}')
print(f'{"Previous best (epoch 41)":<35} {PREV_TEST_MAE:>10.6f} {PREV_TEST_RMSE:>10.6f}')
print(f'{"This run (epoch {best_epoch})":<35} {test_mae:>10.6f} {test_rmse:>10.6f}')
print('-' * 65)
print(f'{"vs previous best":<35} {mae_vs_prev:>+9.1f}%')
print(f'{"vs baseline TFT":<35} {mae_vs_baseline:>+9.1f}%')
print(f'{"vs persistence":<35} {mae_vs_persistence:>+9.1f}%')
print('=' * 65)

# ── Plots ─────────────────────────────────────────────────────────────────────
n_plot = min(500, len(test_actuals))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_actuals[:n_plot],                    label='Actual',           color='steelblue', linewidth=1.2)
ax.plot(test_preds[:n_plot],                      label='TFT (continued)',  color='tomato',    linewidth=1.0, alpha=0.85)
ax.plot(pers_preds_cpu[:n_plot],                  label='Persistence',      color='gray',      linewidth=0.9, linestyle='--', alpha=0.7)
ax.set_title(f'TFT Continued — Test Set (epoch {best_epoch})')
ax.set_xlabel('Sample')
ax.set_ylabel('CPU Usage (raw)')
ax.legend()
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, 'tft_continued_predictions.png')
plt.savefig(p1, dpi=120)
plt.close()

# Training history — both runs combined
metrics_prev = os.path.join(LOG_DIR, 'tft', 'exp_abcd', 'metrics.csv')
metrics_cont = os.path.join(LOG_DIR, 'tft', EXP_NAME,   'metrics.csv')

mdf_p = pd.read_csv(metrics_prev)
mdf_c = pd.read_csv(metrics_cont)
prev_val = mdf_p[mdf_p['val_loss'].notna()][['epoch','val_loss']].copy()
cont_val = mdf_c[mdf_c['val_loss'].notna()][['epoch','val_loss']].copy()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(prev_val['epoch'], prev_val['val_loss'], marker='o', color='steelblue', label='Run 1 (epochs 0-49)')
ax.plot(cont_val['epoch'], cont_val['val_loss'], marker='s', color='tomato',    label='Continued')
ax.axhline(PREV_BEST_VAL_LOSS, color='gray', linestyle='--', linewidth=1, label=f'Previous best {PREV_BEST_VAL_LOSS:.4f}')
ax.axhline(0.004617,           color='black', linestyle=':',  linewidth=1, label='Baseline TFT best')
ax.set_title('TFT val_loss — Full Training History')
ax.set_xlabel('Epoch')
ax.set_ylabel('val_loss')
ax.legend()
plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, 'tft_continued_full_history.png')
plt.savefig(p2, dpi=120)
plt.close()

print(f'\nPlots → {p1}')
print(f'        {p2}')

# ── Save results JSON ─────────────────────────────────────────────────────────
results = {
    'experiment': 'continued',
    'resumed_from': RESUME_CKPT,
    'resumed_from_epoch': int(resume_epoch),
    'best_epoch': best_epoch,
    'best_val_loss': float(checkpoint_cb.best_model_score),
    'improved_over_previous': improved,
    'training_duration_minutes': round(duration / 60, 1),
    'validation': {'mae': val_mae,  'rmse': val_rmse},
    'test':       {'mae': test_mae, 'rmse': test_rmse},
    'comparison': {
        'previous_best':  {'epoch': 41, 'test_mae': PREV_TEST_MAE, 'test_rmse': PREV_TEST_RMSE, 'val_loss': PREV_BEST_VAL_LOSS},
        'baseline_tft':   {'test_mae': BASELINE_TEST_MAE},
        'persistence':    {'test_mae': PERSISTENCE_MAE},
        'mae_vs_previous_pct':    round(mae_vs_prev,        2),
        'mae_vs_baseline_pct':    round(mae_vs_baseline,    2),
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
        'features': [
            'avg_cpu_scaled',
            'avg_memory_scaled',
            'max_cpu_scaled',
            'assigned_memory_scaled',
        ],
    },
}

json_path = os.path.join(RESULTS_DIR, 'tft_continued_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results JSON → {json_path}')
print('\n' + json.dumps(results, indent=2))
