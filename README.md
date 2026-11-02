# FatigueSense AI

CPU fatigue forecasting using a Temporal Fusion Transformer (TFT) on Google Cluster 2019 data.

---

## Project Overview

FatigueSense AI predicts short-term CPU utilization trends (15-minute horizon) for datacenter workloads. The model ingests historical CPU usage sequences from the Google Cluster 2019 dataset and produces 3-step-ahead forecasts at 5-minute resolution. The goal is to detect and anticipate compute resource fatigue before it manifests as service degradation.

---

## Dataset

| Property | Value |
|---|---|
| Source | Google Cluster 2019 — exported from BigQuery |
| File | `data/raw/google_cluster_2019.csv` |
| Total rows | 894,263 |
| Unique instances | 100 |
| Unique machines | 218 |
| Time range | 31 days (~300 s to ~2,678,700 s) |
| Sampling interval | 300 seconds (5 minutes) |
| CPU values | Normalized fractions (avg_cpu_scaled via MinMaxScaler) |

Each row represents one 5-minute window for a single instance, recording average CPU usage as a normalized fraction in [0, 1].

---

## Project Structure

```
FatigueSense AI/
├── configs/            # Experiment and preprocessing configuration files
├── checkpoints/        # Saved model checkpoints from training runs
├── data/
│   ├── raw/            # Original CSV export from BigQuery
│   └── processed/      # Cleaned sequences, NPZ windows, and scaler artifact
├── experiments/
│   ├── logs/           # Training logs (TensorBoard / Lightning)
│   ├── plots/          # Generated visualizations
│   └── results/        # JSON summaries and pipeline reports
├── src/
│   ├── models/         # TFT model definition and supporting modules
│   ├── data_inspection.py   # EDA script — profiles the raw CSV
│   ├── preprocessing.py     # Cleaning, scaling, and chronological splitting
│   └── windowing.py         # Sliding-window dataset construction
└── requirements.txt    # Pinned Python dependencies
```

---

## Environment Setup

The project uses a dedicated virtual environment at `C:\ML-Environment\.venv`.

### Activate the environment

```powershell
# PowerShell
C:\ML-Environment\.venv\Scripts\Activate.ps1
```

```cmd
# Command Prompt
C:\ML-Environment\.venv\Scripts\activate.bat
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

> **GPU note:** `torch`, `torchvision`, and `torchaudio` are pinned to the `+cu121` (CUDA 12.1) builds. If you are running on CPU only, replace them with the CPU-only wheels from [pytorch.org](https://pytorch.org).

---

## Running the Pipeline

Run the scripts in order from the project root with the venv active.

### 1. Data Inspection

```powershell
python src/data_inspection.py
```

Profiles the raw CSV: row counts, instance counts, time range, observation distributions, and estimated valid training window counts. Results are written to `experiments/results/inspection_summary.json`.

### 2. Preprocessing

```powershell
python src/preprocessing.py
```

Drops incomplete sequences, fits a MinMaxScaler on the training portion, applies it to all splits, and saves cleaned sequences to `data/processed/cleaned_sequences.parquet` and the scaler to `data/processed/scaler.joblib`. Split boundaries are persisted to `configs/preprocessing_config.json`.

### 3. Windowing

```powershell
python src/windowing.py
```

Applies a sliding window (lookback 12, horizon 3) across every instance's chronological sequence and saves the resulting NumPy arrays to `data/processed/windows_train.npz`, `windows_val.npz`, and `windows_test.npz`. Metadata is written to `data/processed/windows_metadata.json`.

---

## Experiment Configuration

All hyperparameters and dataset settings live in `configs/experiment_config.yaml`.

Key values:

| Parameter | Value | Notes |
|---|---|---|
| `lookback_steps` | 12 | 1 hour of history at 5-min resolution |
| `forecast_horizon_steps` | 3 | 15-minute forecast horizon |
| `train_split` | 0.70 | 70 % of the time axis |
| `val_split` | 0.15 | Next 15 % chronologically |
| `test_split` | 0.15 | Final 15 % chronologically |
| `device` | cuda | Switch to `cpu` if no GPU is available |
| `tft.hidden_size` | 64 | TFT hidden layer width |
| `tft.attention_head_size` | 4 | Multi-head attention heads |
| `tft.dropout` | 0.1 | Dropout rate |
| `tft.learning_rate` | 0.001 | Adam initial learning rate |
| `tft.batch_size` | 64 | Training batch size |
| `tft.max_epochs` | 50 | Maximum training epochs |
| `tft.patience` | 10 | Early stopping patience |

---

## Data Splits

Splits are **chronological and global** — the time axis is divided once across all instances rather than per-instance. This prevents any future data leaking into the training set.

| Split | Proportion | Boundary |
|---|---|---|
| Train | 70 % | up to ~1,875,180 s |
| Validation | 15 % | ~1,875,180 s – ~2,276,940 s |
| Test | 15 % | ~2,276,940 s – 2,678,700 s |

All scaling (MinMaxScaler) is fit exclusively on training data and applied to validation and test sets to avoid data leakage.
