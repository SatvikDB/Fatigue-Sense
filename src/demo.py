"""
TFT Live Terminal Demo — FatigueSense AI
=========================================
Shows real predictions from the trained model directly in the terminal.
No plots. Just numbers, tables, and fatigue alerts.

Run:
    C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\demo.py"

Options:
    --samples 10        how many test instances to show (default: 10)
    --live              simulate live streaming (pause between each prediction)
    --all               show all 100 test instances
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import joblib
import torch

# Force UTF-8 output on Windows so special chars don't crash
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# suppress lightning info logs
os.environ['LIGHTNING_VERBOSITY'] = 'ERROR'
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from models.tft_model import build_datasets
from pytorch_forecasting import TemporalFusionTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_PATH   = r'd:\FatigueSense AI\checkpoints\tft_exp_abcd_epoch41_SAFE.ckpt'
DATA_PATH   = r'd:\FatigueSense AI\data\processed\cleaned_sequences.parquet'
SCALER_PATH = r'd:\FatigueSense AI\data\processed\scaler.joblib'

QUANTILES   = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
MEDIAN_IDX  = 3

# ── Helpers ───────────────────────────────────────────────────────────────────
def cpu_pct(val):
    """Convert raw fraction to percentage string."""
    return f'{val*100:.4f}%'

def bar(val, max_val=0.07, width=30):
    """Simple ASCII bar chart."""
    filled = int((val / max(max_val, 1e-9)) * width)
    filled = min(filled, width)
    return '#' * filled + '.' * (width - filled)

def trend_arrow(current, predicted):
    diff_pct = (predicted - current) / max(current, 1e-9) * 100
    if diff_pct > 15:
        return '++ RISING  ', '!! FATIGUE RISK'
    elif diff_pct > 5:
        return '+  Climbing', '   watch closely'
    elif diff_pct < -15:
        return '-- FALLING ', '   workload easing'
    elif diff_pct < -5:
        return '-  Dropping', '   recovering'
    else:
        return '-> Stable  ', '   OK normal'

def inv(scaler, val):
    return float(scaler.inverse_transform([[val]])[0][0])

# ── Load everything ───────────────────────────────────────────────────────────
def setup():
    print('=' * 60)
    print('  FatigueSense AI -- TFT Live Demo')
    print('  Predicting CPU usage 15 minutes ahead')
    print('=' * 60)
    print()
    print('Loading model...', end=' ', flush=True)
    model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
    model.eval()
    print('OK  (epoch 41, 1,057,595 params)')

    print('Loading scalers...', end=' ', flush=True)
    cpu_scaler = joblib.load(SCALER_PATH)
    sdir = os.path.dirname(SCALER_PATH)
    print('OK')

    print('Building test dataset...', end=' ', flush=True)
    _, _, test_ds, df = build_datasets(DATA_PATH, SCALER_PATH)
    print('OK')
    print()
    return model, cpu_scaler, test_ds, df

# ── Main demo ─────────────────────────────────────────────────────────────────
def run_demo(n_samples=10, live=False):
    model, cpu_scaler, test_ds, df = setup()

    test_dl = test_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    # Run all predictions silently
    print('Running inference on test set...', end=' ', flush=True)
    import logging
    logging.getLogger('lightning').setLevel(logging.ERROR)

    raw      = model.predict(test_dl, mode='raw', return_x=True)
    all_preds = raw.output['prediction'].cpu().numpy()   # (N, 3, 7)

    acts = []
    for x, y in test_dl:
        tgt = y[0] if isinstance(y, (list, tuple)) else y
        acts.append(tgt.cpu().numpy())
    all_acts = np.concatenate(acts, axis=0)              # (N, 3)

    n_total = min(len(all_preds), len(all_acts))
    print(f'Done  ({n_total} predictions ready)\n')

    # Inverse transform helper
    def to_real(scaled_val):
        return inv(cpu_scaler, float(scaled_val))

    # How many to show
    n_show = n_total if n_samples == -1 else min(n_samples, n_total)

    # ── Summary header ────────────────────────────────────────────────────────
    print('-' * 70)
    print('  Showing {} of {} test predictions'.format(n_show, n_total))
    print('  Each row = one instance, predicting 15 min (3 steps) ahead')
    print('  Horizon: +5 min | +10 min | +15 min')
    print('-' * 70)

    errors = []

    for i in range(n_show):
        # Ground truth for all 3 horizon steps
        act_5  = to_real(all_acts[i, 0])
        act_10 = to_real(all_acts[i, 1])
        act_15 = to_real(all_acts[i, 2])

        # Predictions — median (q50) for each step
        pred_5  = to_real(all_preds[i, 0, MEDIAN_IDX])
        pred_10 = to_real(all_preds[i, 1, MEDIAN_IDX])
        pred_15 = to_real(all_preds[i, 2, MEDIAN_IDX])

        # Confidence interval for +15 min step
        q02 = to_real(all_preds[i, 2, 0])
        q25 = to_real(all_preds[i, 2, 2])
        q75 = to_real(all_preds[i, 2, 4])
        q98 = to_real(all_preds[i, 2, 6])

        # Error on 15-min prediction
        err   = pred_15 - act_15
        errors.append(abs(err))

        trend, status = trend_arrow(act_5, pred_15)

        print('  Sample #{:>3}'.format(i+1))
        print('  ' + '-'*66)

        max_val = max(act_15, pred_15, 0.01)
        print('  ACTUAL   | +5min: {:>10}  +10min: {:>10}  +15min: {:>10}'.format(
              cpu_pct(act_5), cpu_pct(act_10), cpu_pct(act_15)))
        print('  PREDICTED| +5min: {:>10}  +10min: {:>10}  +15min: {:>10}'.format(
              cpu_pct(pred_5), cpu_pct(pred_10), cpu_pct(pred_15)))
        err_sign = '+' if err >= 0 else ''
        print('  ERROR    | +15min prediction error: {}{}'.format(err_sign, cpu_pct(err)))
        print('  INTERVAL | 2%: {}  25%: {}  [{}]  75%: {}  98%: {}'.format(
              cpu_pct(q02), cpu_pct(q25), cpu_pct(pred_15), cpu_pct(q75), cpu_pct(q98)))
        print('  ACTUAL   | {}  {}'.format(bar(act_15, max_val), cpu_pct(act_15)))
        print('  PREDICTED| {}  {}'.format(bar(pred_15, max_val), cpu_pct(pred_15)))
        print('  TREND    | {}   Status: {}'.format(trend, status))

        if live:
            time.sleep(1.2)

    # ── Overall summary ───────────────────────────────────────────────────────
    mae  = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors)**2))

    print()
    print('-' * 70)
    print('  SUMMARY')
    print('-' * 70)
    print(f'  Samples shown        : {n_show}')
    print(f'  MAE  (15min horizon) : {cpu_pct(mae)}')
    print(f'  RMSE (15min horizon) : {cpu_pct(rmse)}')
    print(f'  Persistence MAE      : 0.0862%  (dumb baseline)')
    print(f'  TFT improvement      : {(0.000862 - mae)/0.000862*100:.1f}% better than persistence')
    print()

    # Fatigue alerts summary
    rising = sum(
        1 for i in range(n_show)
        if (to_real(all_preds[i, 2, MEDIAN_IDX]) - to_real(all_acts[i, 0]))
           / max(to_real(all_acts[i, 0]), 1e-9) > 0.15
    )
    print('  Fatigue alerts fired : {} / {} instances'.format(rising, n_show))
    print('  Clean instances      : {} / {} instances'.format(n_show - rising, n_show))
    print('-' * 70)
    print()

    # Per-step accuracy breakdown
    print('  ACCURACY BY HORIZON STEP:')
    print('  {:<12} {:>12} {:>12}'.format('Step', 'MAE', 'RMSE'))
    print('  ' + '-'*40)
    for step_i, label in enumerate(['+5 min', '+10 min', '+15 min']):
        step_preds = np.array([to_real(all_preds[i, step_i, MEDIAN_IDX]) for i in range(n_show)])
        step_acts  = np.array([to_real(all_acts[i, step_i])               for i in range(n_show)])
        s_mae  = np.mean(np.abs(step_preds - step_acts))
        s_rmse = np.sqrt(np.mean((step_preds - step_acts)**2))
        print('  {:<12} {:>12} {:>12}'.format(label, cpu_pct(s_mae), cpu_pct(s_rmse)))
    print('-' * 70)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFT Terminal Demo — FatigueSense AI')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of test samples to show (default: 10)')
    parser.add_argument('--live',    action='store_true',
                        help='Pause between predictions to simulate live feed')
    parser.add_argument('--all',     action='store_true',
                        help='Show all 100 test instances')
    args = parser.parse_args()

    n = -1 if args.all else args.samples
    run_demo(n_samples=n, live=args.live)
