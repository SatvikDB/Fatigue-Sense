"""
TFT Inference Script — FatigueSense AI
=======================================
Two ways to test the model:

  MODE 1 — Full test set evaluation (default):
      C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\predict.py"

  MODE 2 — Single custom sequence prediction:
      C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\predict.py" --custom

  MODE 3 — Specific group from test set:
      C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\predict.py" --group 1632087718_1390

What it predicts:
  Given the last 24 timesteps (2 hours) of:
    - avg_cpu       (average CPU usage)
    - avg_memory    (average memory usage)
    - max_cpu       (peak CPU in window)
    - assigned_memory (allocated memory)

  It predicts avg_cpu 15 minutes into the future (3 steps ahead)
  with 7 confidence intervals: 2%, 10%, 25%, 50%, 75%, 90%, 98%
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import joblib
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from models.tft_model import build_datasets

from pytorch_forecasting import TemporalFusionTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = r'd:\FatigueSense AI\data\processed\cleaned_sequences.parquet'
SCALER_PATH  = r'd:\FatigueSense AI\data\processed\scaler.joblib'
CKPT_PATH    = r'd:\FatigueSense AI\checkpoints\tft_exp_abcd_epoch41_SAFE.ckpt'
RESULTS_DIR  = r'd:\FatigueSense AI\experiments\results'
PLOTS_DIR    = r'd:\FatigueSense AI\experiments\plots'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

QUANTILE_LABELS = ['2%', '10%', '25%', '50%', '75%', '90%', '98%']
MEDIAN_IDX      = 3   # index of 0.5 quantile in the 7-quantile output

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    print(f'Loading model from: {CKPT_PATH}')
    model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
    model.eval()
    ckpt  = torch.load(CKPT_PATH, map_location='cpu')
    print(f'  Trained epoch : {ckpt.get("epoch")}')
    print(f'  Parameters    : {sum(p.numel() for p in model.parameters()):,}')
    return model

# ── Scalers ───────────────────────────────────────────────────────────────────
def load_scalers():
    sdir = os.path.dirname(SCALER_PATH)
    scalers = {
        'avg_cpu':          joblib.load(SCALER_PATH),
        'avg_memory':       joblib.load(os.path.join(sdir, 'scaler_avg_memory.joblib')),
        'max_cpu':          joblib.load(os.path.join(sdir, 'scaler_max_cpu.joblib')),
        'assigned_memory':  joblib.load(os.path.join(sdir, 'scaler_assigned_memory.joblib')),
    }
    return scalers

def inv_cpu(scalers, arr):
    return scalers['avg_cpu'].inverse_transform(
        np.array(arr).reshape(-1, 1)
    ).flatten()

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: Full test set
# ─────────────────────────────────────────────────────────────────────────────
def run_test_set():
    print('\n' + '='*60)
    print('MODE 1: Full Test Set Evaluation')
    print('='*60)

    model   = load_model()
    scalers = load_scalers()

    print('\nBuilding datasets...')
    train_ds, val_ds, test_ds, df = build_datasets(DATA_PATH, SCALER_PATH)
    test_dl = test_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    print('Running predictions...')
    raw = model.predict(test_dl, mode='raw', return_x=True)

    # All 7 quantiles for step 3 (15 min ahead)
    # shape: (N, prediction_length=3, num_quantiles=7)
    all_quantiles_scaled = raw.output['prediction'][:, 2, :].cpu().numpy()  # (N, 7)
    pred_scaled          = all_quantiles_scaled[:, MEDIAN_IDX]               # median

    # Actuals
    acts = []
    for x, y in test_dl:
        tgt = y[0] if isinstance(y, (list, tuple)) else y
        acts.append(tgt[:, 2].cpu().numpy())
    actual_scaled = np.concatenate(acts)

    n              = min(len(pred_scaled), len(actual_scaled))
    pred_scaled    = pred_scaled[:n]
    actual_scaled  = actual_scaled[:n]

    predicted = inv_cpu(scalers, pred_scaled)
    actual    = inv_cpu(scalers, actual_scaled)

    mae  = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual)**2)))

    # Persistence baseline
    VAL_MAX  = 2276940.0
    test_raw = df[df['start_time_s'] > VAL_MAX].sort_values(['group_id','start_time_s'])
    test_raw = test_raw.copy()
    test_raw['pers'] = test_raw.groupby('group_id')['avg_cpu_scaled'].shift(3)
    test_raw = test_raw.dropna(subset=['pers'])
    pers     = inv_cpu(scalers, test_raw['pers'].values)
    acts_p   = inv_cpu(scalers, test_raw['avg_cpu_scaled'].values)
    pers_mae = float(np.mean(np.abs(pers - acts_p)))

    print()
    print('='*60)
    print('TEST SET RESULTS')
    print('='*60)
    print(f'  Samples evaluated     : {n:,}')
    print(f'  Test MAE  (TFT)       : {mae:.6f}  ({mae*100:.4f}% CPU)')
    print(f'  Test RMSE (TFT)       : {rmse:.6f}')
    print(f'  Persistence MAE       : {pers_mae:.6f}')
    print(f'  TFT vs persistence    : {(mae-pers_mae)/pers_mae*100:+.1f}%')
    print(f'  Mean CPU in test data : {actual.mean():.6f}  ({actual.mean()*100:.3f}%)')
    print(f'  MAE as % of mean CPU  : {mae/actual.mean()*100:.1f}%')
    print('='*60)

    # Confidence intervals (inverse transform all quantiles)
    q_real = np.stack([
        inv_cpu(scalers, all_quantiles_scaled[:n, i])
        for i in range(7)
    ], axis=1)  # (N, 7)

    # Plot 1: predictions vs actual with confidence band
    n_plot = min(200, n)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(range(n_plot), q_real[:n_plot, 0], q_real[:n_plot, 6],
                    alpha=0.15, color='tomato', label='2%–98% interval')
    ax.fill_between(range(n_plot), q_real[:n_plot, 1], q_real[:n_plot, 5],
                    alpha=0.25, color='tomato', label='10%–90% interval')
    ax.fill_between(range(n_plot), q_real[:n_plot, 2], q_real[:n_plot, 4],
                    alpha=0.35, color='tomato', label='25%–75% interval')
    ax.plot(range(n_plot), actual[:n_plot],          color='steelblue', linewidth=1.4, label='Actual')
    ax.plot(range(n_plot), q_real[:n_plot, MEDIAN_IDX], color='tomato', linewidth=1.1, label='Predicted (median)')
    ax.set_title('TFT — Test Set: Predictions with Confidence Intervals (first 200 samples)')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('CPU Usage (raw fraction)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, 'tft_inference_confidence.png')
    plt.savefig(p1, dpi=130)
    plt.close()
    print(f'\nPlot saved → {p1}')

    # Plot 2: error distribution
    errors = predicted - actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(errors, bins=50, color='steelblue', edgecolor='white')
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].set_title('Prediction Error Distribution (Test Set)')
    axes[0].set_xlabel('Predicted − Actual CPU')
    axes[0].set_ylabel('Count')

    axes[1].scatter(actual[:500], errors[:500], alpha=0.4, s=10, color='steelblue')
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Residuals vs Actual CPU (first 500)')
    axes[1].set_xlabel('Actual CPU')
    axes[1].set_ylabel('Error')
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, 'tft_inference_errors.png')
    plt.savefig(p2, dpi=130)
    plt.close()
    print(f'Plot saved → {p2}')

    # Save predictions CSV
    csv_path = os.path.join(RESULTS_DIR, 'tft_test_predictions.csv')
    out_df = pd.DataFrame({
        'actual':    actual,
        'predicted': predicted,
        'error':     errors,
        'q02': q_real[:, 0], 'q10': q_real[:, 1], 'q25': q_real[:, 2],
        'q50': q_real[:, 3], 'q75': q_real[:, 4], 'q90': q_real[:, 5],
        'q98': q_real[:, 6],
    })
    out_df.to_csv(csv_path, index=False)
    print(f'Predictions CSV → {csv_path}')


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: Custom manual input
# ─────────────────────────────────────────────────────────────────────────────
def run_custom():
    print('\n' + '='*60)
    print('MODE 2: Custom Sequence Prediction')
    print('='*60)
    print('Enter 24 timesteps of data (each 5 minutes apart).')
    print('Press Enter to use example values instead.\n')

    # Example: a realistic sequence from the test set
    example_cpu = [
        0.0086, 0.0073, 0.0068, 0.0056, 0.0069, 0.0073,
        0.0117, 0.0166, 0.0129, 0.0099, 0.0057, 0.0055,
        0.0052, 0.0052, 0.0051, 0.0051, 0.0057, 0.0060,
        0.0060, 0.0057, 0.0068, 0.0059, 0.0056, 0.0054,
    ]
    example_mem = [0.0014] * 24
    example_maxcpu = [
        0.0189, 0.0237, 0.0169, 0.0249, 0.0267, 0.0207,
        0.0267, 0.0325, 0.0292, 0.0405, 0.0121, 0.0151,
        0.0128, 0.0151, 0.0091, 0.0091, 0.0113, 0.0118,
        0.0164, 0.0214, 0.0166, 0.0199, 0.0108, 0.0123,
    ]
    example_assigned = [0.00444] * 24

    user_input = input('Use example data? (Y/n): ').strip().lower()
    if user_input == 'n':
        print('\nEnter 24 avg_cpu values (space-separated):')
        avg_cpu_vals      = list(map(float, input('avg_cpu      : ').split()))
        print('Enter 24 avg_memory values (space-separated):')
        avg_memory_vals   = list(map(float, input('avg_memory   : ').split()))
        print('Enter 24 max_cpu values (space-separated):')
        max_cpu_vals      = list(map(float, input('max_cpu      : ').split()))
        print('Enter 24 assigned_memory values (space-separated):')
        assigned_mem_vals = list(map(float, input('assigned_mem : ').split()))
    else:
        avg_cpu_vals      = example_cpu
        avg_memory_vals   = example_mem
        max_cpu_vals      = example_maxcpu
        assigned_mem_vals = example_assigned
        print('Using example values.')

    scalers = load_scalers()

    # Scale inputs using saved scalers
    def scale(vals, key):
        return scalers[key].transform(np.array(vals).reshape(-1,1)).flatten()

    cpu_s    = scale(avg_cpu_vals,      'avg_cpu')
    mem_s    = scale(avg_memory_vals,   'avg_memory')
    maxcpu_s = scale(max_cpu_vals,      'max_cpu')
    asgn_s   = scale(assigned_mem_vals, 'assigned_memory')

    # Build a minimal dataframe with one group, 27 rows (24 encoder + 3 decoder)
    n_rows = 27
    rows = []
    for i in range(n_rows):
        rows.append({
            'group_id':               'custom_input',
            'time_idx':               i,
            'avg_cpu_scaled':         float(cpu_s[min(i, 23)]),
            'avg_memory_scaled':      float(mem_s[min(i, 23)]),
            'max_cpu_scaled':         float(maxcpu_s[min(i, 23)]),
            'assigned_memory_scaled': float(asgn_s[min(i, 23)]),
        })
    inp_df = pd.DataFrame(rows)

    from pytorch_forecasting import TimeSeriesDataSet
    ds = TimeSeriesDataSet(
        inp_df,
        time_idx='time_idx',
        target='avg_cpu_scaled',
        group_ids=['group_id'],
        min_encoder_length=24, max_encoder_length=24,
        min_prediction_length=3, max_prediction_length=3,
        static_categoricals=['group_id'],
        time_varying_unknown_reals=[
            'avg_cpu_scaled','avg_memory_scaled',
            'max_cpu_scaled','assigned_memory_scaled'
        ],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    dl    = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    model = load_model()

    raw   = model.predict(dl, mode='raw', return_x=True)
    # shape: (1, 3, 7) — 1 sample, 3 steps, 7 quantiles
    preds_scaled = raw.output['prediction'][0].cpu().numpy()  # (3, 7)
    preds_real   = np.stack([
        inv_cpu(scalers, preds_scaled[:, q]) for q in range(7)
    ], axis=1)  # (3, 7)

    print()
    print('='*60)
    print('PREDICTION RESULTS')
    print('='*60)
    print(f'Last known avg_cpu : {avg_cpu_vals[-1]:.6f}  ({avg_cpu_vals[-1]*100:.4f}%)')
    print()
    print(f'{"Step":<8} {"Time":^10} {"2%":>8} {"10%":>8} {"25%":>8} {"50% (median)":>13} {"75%":>8} {"90%":>8} {"98%":>8}')
    print('-'*80)
    for step in range(3):
        mins = (step+1) * 5
        q    = preds_real[step]
        print(f'Step {step+1:<3} {"+"+str(mins)+" min":^10} '
              f'{q[0]:>8.5f} {q[1]:>8.5f} {q[2]:>8.5f} '
              f'{q[3]:>13.5f} {q[4]:>8.5f} {q[5]:>8.5f} {q[6]:>8.5f}')
    print('='*60)
    print(f'\nFatigue risk interpretation:')
    median_15min = preds_real[2, MEDIAN_IDX]
    last_cpu     = avg_cpu_vals[-1]
    change_pct   = (median_15min - last_cpu) / max(last_cpu, 1e-9) * 100
    print(f'  Current CPU      : {last_cpu:.4f} ({last_cpu*100:.3f}%)')
    print(f'  Predicted in 15m : {median_15min:.4f} ({median_15min*100:.3f}%)')
    print(f'  Trend            : {change_pct:+.1f}%')
    if change_pct > 20:
        print('  Status           : ⚠ RISING — potential fatigue build-up')
    elif change_pct < -20:
        print('  Status           : ↓ FALLING — workload easing')
    else:
        print('  Status           : ✓ STABLE — no immediate fatigue concern')


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3: Specific group
# ─────────────────────────────────────────────────────────────────────────────
def run_group(group_id):
    print(f'\n{"="*60}')
    print(f'MODE 3: Group Prediction — {group_id}')
    print('='*60)

    model   = load_model()
    scalers = load_scalers()

    _, _, test_ds, df = build_datasets(DATA_PATH, SCALER_PATH)
    test_dl = test_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    VAL_MAX  = 2276940.0
    grp_data = df[(df['start_time_s'] > VAL_MAX) & (df['group_id'] == group_id)]
    grp_data = grp_data.sort_values('start_time_s')

    if len(grp_data) == 0:
        print(f'Group "{group_id}" not found in test set.')
        print('Available groups:')
        avail = df[df['start_time_s'] > VAL_MAX]['group_id'].unique()
        for g in sorted(avail)[:20]:
            print(f'  {g}')
        return

    # Get predictions for this group via full test set predict
    raw      = model.predict(test_dl, mode='raw', return_x=True)
    preds_s  = raw.output['prediction'][:, 2, MEDIAN_IDX].cpu().numpy()
    acts     = []
    for x, y in test_dl:
        tgt = y[0] if isinstance(y, (list, tuple)) else y
        acts.append(tgt[:, 2].cpu().numpy())
    acts_s = np.concatenate(acts)
    n      = min(len(preds_s), len(acts_s))

    predicted = inv_cpu(scalers, preds_s[:n])
    actual    = inv_cpu(scalers, acts_s[:n])
    mae       = float(np.mean(np.abs(predicted - actual)))

    print(f'  Test samples : {n}')
    print(f'  MAE          : {mae:.6f}')

    n_plot = min(100, n)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual[:n_plot],    label='Actual',    color='steelblue', linewidth=1.3)
    ax.plot(predicted[:n_plot], label='Predicted', color='tomato',    linewidth=1.1, alpha=0.9)
    ax.set_title(f'TFT Predictions — Group: {group_id}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('CPU Usage')
    ax.legend()
    plt.tight_layout()
    safe_name = group_id.replace('/', '_')
    p = os.path.join(PLOTS_DIR, f'tft_group_{safe_name}.png')
    plt.savefig(p, dpi=130)
    plt.close()
    print(f'  Plot → {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFT Inference — FatigueSense AI')
    parser.add_argument('--custom', action='store_true',
                        help='Run interactive custom input prediction')
    parser.add_argument('--group',  type=str, default=None,
                        help='Predict for a specific group_id from test set')
    args = parser.parse_args()

    if args.custom:
        run_custom()
    elif args.group:
        run_group(args.group)
    else:
        run_test_set()
