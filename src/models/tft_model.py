"""
Temporal Fusion Transformer model — FatigueSense AI
=====================================================
Features used:
  - avg_cpu_scaled       : average CPU usage (0-1 fraction), MinMaxScaler on train
  - avg_memory_scaled    : average memory usage (0-1 fraction), MinMaxScaler on train
  - max_cpu_scaled       : peak CPU in window (can exceed 1.0 raw), MinMaxScaler on train
  - assigned_memory_scaled: allocated memory limit, MinMaxScaler on train

Target: avg_cpu_scaled (3 steps ahead = 15 minutes)
Lookback: 24 steps (2 hours)
Horizon:  3 steps  (15 minutes)

Columns in cleaned_sequences.parquet:
    collection_id, instance_index, machine_id, start_time, end_time,
    avg_cpu, avg_memory, max_cpu, max_memory, assigned_memory,
    page_cache_memory, sample_rate, start_time_s, end_time_s,
    run_id, split, avg_cpu_scaled

Excluded features (with reasons):
  - max_memory     : 2.26% nulls across 15 groups, collinear with avg_memory
  - page_cache_memory : r=0.09 correlation with avg_cpu, no predictive value
  - network columns   : not present in this dataset (Google Cluster 2019)
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Split thresholds — derived from preprocessing.py (70/15/15 chronological)
TRAIN_MAX = 1875179.9999999998
VAL_MAX   = 2276940.0

# Features added on top of avg_cpu_scaled
EXTRA_FEATURES = ['avg_memory_scaled', 'max_cpu_scaled', 'assigned_memory_scaled']
ALL_TIME_FEATURES = ['avg_cpu_scaled'] + EXTRA_FEATURES


def _fit_and_apply_scalers(df, scaler_dir):
    """
    Fit one MinMaxScaler per extra feature on training rows only,
    transform all rows, and save scalers to scaler_dir.
    Returns df with new *_scaled columns added.
    """
    os.makedirs(scaler_dir, exist_ok=True)
    train_mask = df['start_time_s'] <= TRAIN_MAX

    feature_map = {
        'avg_memory':        'avg_memory_scaled',
        'max_cpu':           'max_cpu_scaled',
        'assigned_memory':   'assigned_memory_scaled',
    }

    for raw_col, scaled_col in feature_map.items():
        scaler = MinMaxScaler()
        scaler.fit(df.loc[train_mask, [raw_col]])
        df[scaled_col] = scaler.transform(df[[raw_col]]).flatten()
        path = os.path.join(scaler_dir, f'scaler_{raw_col}.joblib')
        joblib.dump(scaler, path)

    return df


def build_datasets(data_path, scaler_path):
    """
    Load parquet, scale all features (train-only fit), build TimeSeriesDataSets.
    Returns (train_dataset, val_dataset, test_dataset, df).
    scaler_path is the existing avg_cpu scaler — used to locate the scaler directory.
    """
    df = pd.read_parquet(data_path, engine='fastparquet')
    scaler_dir = os.path.dirname(scaler_path)

    # avg_cpu_scaled already present in parquet
    if 'avg_cpu_scaled' not in df.columns:
        scaler = joblib.load(scaler_path)
        df['avg_cpu_scaled'] = scaler.transform(df[['avg_cpu']]).flatten()

    # Scale extra features — fit on train only, transform all splits
    df = _fit_and_apply_scalers(df, scaler_dir)

    # Group identifier: one time series per (collection_id × instance_index)
    df['group_id'] = (
        df['collection_id'].astype(str) + '_' + df['instance_index'].astype(str)
    )

    # Sort so time_idx is strictly increasing within each group
    df = df.sort_values(['group_id', 'start_time_s']).reset_index(drop=True)

    # Consecutive integer time index per group
    df['time_idx'] = df.groupby('group_id').cumcount()

    # Chronological splits
    train_df = df[df['start_time_s'] <= TRAIN_MAX].copy()
    val_df   = df[df['start_time_s'] <= VAL_MAX].copy()
    test_df  = df[df['start_time_s'] >  VAL_MAX].copy()

    dataset_kwargs = dict(
        time_idx='time_idx',
        target='avg_cpu_scaled',
        group_ids=['group_id'],
        min_encoder_length=24,
        max_encoder_length=24,          # 24 steps = 2 hours lookback
        min_prediction_length=3,
        max_prediction_length=3,        # 3 steps = 15 minutes horizon
        static_categoricals=['group_id'],
        time_varying_unknown_reals=ALL_TIME_FEATURES,
        target_normalizer=None,         # already scaled
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    train_dataset = TimeSeriesDataSet(train_df, **dataset_kwargs)

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, val_df, predict=True, stop_randomization=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, test_df, predict=True, stop_randomization=True
    )

    return train_dataset, val_dataset, test_dataset, df


def build_tft_model(train_dataset):
    """
    Return a configured TemporalFusionTransformer built from train_dataset.
    Architecture is scaled up slightly to handle 4 input features.
    """
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        hidden_size=128,            # increased from 64 — more features need more capacity
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=64,  # increased from 32
        output_size=7,              # 7 quantiles (default)
        learning_rate=0.001,
        reduce_on_plateau_patience=4,
        loss=QuantileLoss(),
    )
    return model


if __name__ == '__main__':
    DATA_PATH   = r'd:\FatigueSense AI\data\processed\cleaned_sequences.parquet'
    SCALER_PATH = r'd:\FatigueSense AI\data\processed\scaler.joblib'

    train_ds, val_ds, test_ds, df = build_datasets(DATA_PATH, SCALER_PATH)
    print('Train:', len(train_ds))
    print('Val  :', len(val_ds))
    print('Test :', len(test_ds))
    print('Features:', ALL_TIME_FEATURES)

    model = build_tft_model(train_ds)
    print('Parameters:', sum(p.numel() for p in model.parameters()))
