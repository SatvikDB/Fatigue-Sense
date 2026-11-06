# FatigueSense AI

CPU fatigue forecasting using a Temporal Fusion Transformer (TFT) on Google Cluster 2019 data.

---

## What It Does

FatigueSense AI predicts CPU utilization **15 minutes into the future** for datacenter workloads.
Given the last **2 hours** of CPU usage, memory usage, peak CPU, and allocated memory for an
instance, it outputs a 3-step-ahead forecast (5 min, 10 min, 15 min) with 7 confidence
quantiles (2%, 10%, 25%, 50%, 75%, 90%, 98%).

It also fires a **fatigue alert** when the predicted 15-min CPU is more than 15% above the
current level — giving operators early warning before resource exhaustion.

---

## Dataset

| Property | Value |
|---|---|
| Source | Google Cluster 2019 — exported from BigQuery |
| File | `data/raw/google_cluster_2019.csv` |
| Total rows (raw) | 894,263 |
| Rows after cleaning | 889,993 |
| Unique instances | 100 |
| Sampling interval | 300 s (5 minutes) |
| Time range | ~31 days |

---

## Model — Temporal Fusion Transformer

### Architecture

| Parameter | Value |
|---|---|
| Model type | Temporal Fusion Transformer (pytorch-forecasting 1.1.1) |
| Hidden size | 128 |
| Attention heads | 4 |
| Hidden continuous size | 64 |
| Dropout | 0.1 |
| Output | 7 quantiles (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98) |
| Total parameters | 1,057,595 |

### Training config

| Parameter | Value |
|---|---|
| Features | avg_cpu, avg_memory, max_cpu, assigned_memory (all MinMaxScaled) |
| Target | avg_cpu (3 steps = 15 min ahead) |
| Lookback | 24 steps = 2 hours |
| Horizon | 3 steps = 15 minutes |
| Batch size | 256 |
| Precision | BF16 mixed (bf16-mixed) |
| Learning rate | 0.001 with ReduceLROnPlateau (patience=4) |
| Early stopping | patience=15 |
| Max epochs | 100 |
| Seed | 42 |
| Best epoch | 41 |

### Results

| Model | Test MAE | Test RMSE |
|---|---|---|
| Persistence baseline | 0.000862 | — |
| Baseline TFT (1 feature, 12-step lookback) | 0.000722 | 0.001530 |
| **Final TFT (4 features, 24-step lookback)** | **0.000665** | **0.001503** |

- **7.87% better MAE** than single-feature baseline
- **22.83% better** than persistence baseline
- Average prediction error: **0.0665% CPU** over a mean CPU of 0.77%

### Data splits (chronological, no leakage)

| Split | Proportion | Boundary |
|---|---|---|
| Train | 70% | up to ~1,875,180 s |
| Validation | 15% | ~1,875,180 s to ~2,276,940 s |
| Test | 15% | ~2,276,940 s onwards |

Scaler fitted on training data only. Applied to val/test without refitting.

---

## Project Structure

```
FatigueSense AI/
├── checkpoints/
│   ├── tft_exp_abcd_epoch41_SAFE.ckpt   <- best model (never overwrite)
│   └── tft_best.ckpt                    <- baseline model (1 feature)
├── configs/
│   ├── preprocessing_config.json
│   └── experiment_config.yaml
├── data/
│   ├── raw/                             <- google_cluster_2019.csv
│   └── processed/
│       ├── cleaned_sequences.parquet
│       ├── scaler.joblib                <- avg_cpu scaler
│       ├── scaler_avg_memory.joblib
│       ├── scaler_max_cpu.joblib
│       └── scaler_assigned_memory.joblib
├── experiments/
│   ├── logs/tft/                        <- CSVLogger outputs per run
│   ├── plots/                           <- saved figures
│   └── results/                         <- JSON evaluation reports
└── src/
    ├── models/
    │   ├── __init__.py
    │   └── tft_model.py                 <- model + dataset builder
    ├── preprocessing.py                 <- cleaning + scaling pipeline
    ├── windowing.py                     <- sliding window builder
    ├── train.py                         <- baseline training (1 feature)
    ├── train_exp_ab.py                  <- full training (4 features, best run)
    ├── train_continue.py                <- continuation from checkpoint
    ├── evaluate_baseline.py             <- Stage 1 evaluation + leakage audit
    ├── predict.py                       <- test set inference + plots
    └── demo.py                          <- live terminal demo
```

---

## Environment

```
Python   : 3.11
PyTorch  : 2.3.1+cu121
Lightning: 2.6.6
GPU      : NVIDIA RTX 3050 Laptop (4 GB VRAM)
CUDA     : 12.1
```

Activate the environment:
```powershell
C:\ML-Environment\.venv\Scripts\Activate.ps1
```

---

## Running the Project

### Quick demo (terminal, no setup needed)
```powershell
C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\demo.py"
```

Show all 100 test instances:
```powershell
C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\demo.py" --all
```

Simulate live feed (pauses between predictions):
```powershell
C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\demo.py" --live
```

### Full test set evaluation (saves plots + CSV)
```powershell
C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\predict.py"
```

### Re-run training from scratch (4 features, best config)
```powershell
C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\train_exp_ab.py"
```

### Re-run preprocessing (if raw data changes)
```powershell
C:\ML-Environment\.venv\Scripts\python.exe "d:\FatigueSense AI\src\preprocessing.py"
```

---

## Excluded Features (with reasons)

| Feature | Reason excluded |
|---|---|
| max_memory | 2.26% nulls across 15 groups, collinear with avg_memory |
| page_cache_memory | r=0.09 correlation with avg_cpu — no predictive value |
| Network columns | Not present in Google Cluster 2019 dataset |
