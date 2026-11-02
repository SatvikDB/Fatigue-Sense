# FatigueSense AI — Pipeline Setup Report

Generated after completing Steps 1–7 of the initial project setup.

---

## Steps Completed

| Step | Description | Status |
|---|---|---|
| 1 | Project directory scaffold | ✅ Complete |
| 2 | Data inspection (`src/data_inspection.py`) | ✅ Complete |
| 3 | Preprocessing (`src/preprocessing.py`) | ✅ Complete |
| 4 | Windowing (`src/windowing.py`) | ✅ Complete |
| 5 | Experiment config (`configs/experiment_config.yaml`) | ✅ Complete |
| 6 | Requirements file (`requirements.txt`) | ✅ Complete |
| 7 | README (`README.md`) | ✅ Complete |

---

## Key Findings — Data Inspection

Source: `experiments/results/inspection_summary.json`

- **Total rows:** 894,263
- **Unique instances:** 100
- **Unique machines:** 218
- **Time range:** 300 s – 2,678,700 s (31-day span)
- **Observations per instance:** highly uniform — median 8,924, mean 8,942.6; only a small tail of instances reached up to 9,771 observations (instances with minor duplicate or overlap entries)
- **Longest contiguous run distribution:** min 671 steps, median 2,880 steps, mean 2,919.7 steps, p95 5,433 steps, max 6,977 steps — confirming that most instances have long uninterrupted runs suitable for sequence modelling
- **Instances meeting minimum sequence length** (≥ lookback + horizon = 15 steps): all 100 instances passed
- **Estimated valid training windows** (pre-windowing): 884,489

---

## Key Findings — Preprocessing

Source: `configs/preprocessing_config.json`

- **Scaler:** MinMaxScaler fitted on the training split only (no leakage into val/test)
- **Chronological split boundaries:**
  - Train ends at: ~1,875,180 s
  - Validation ends at: ~2,276,940 s
  - Test: ~2,276,940 s to end of dataset
- **Artifacts produced:**
  - `data/processed/cleaned_sequences.parquet` — scaled sequences for all splits
  - `data/processed/scaler.joblib` — fitted MinMaxScaler for inference-time use

---

## Windowing Results

Source: `data/processed/windows_metadata.json`

| Split | Windows |
|---|---|
| Train | 616,630 |
| Validation | 133,017 |
| Test | 130,616 |
| **Total** | **880,263** |

- **Lookback:** 12 steps (60 minutes of history)
- **Horizon:** 3 steps (15-minute forecast)
- **Feature:** `avg_cpu_scaled`

The total window count (880,263) is slightly below the inspection estimate (884,489) because the sliding window cannot extend beyond each instance's available sequence boundary within each split — expected behavior.

---

## Files Created

### Configuration
- `configs/experiment_config.yaml` — TFT hyperparameters and dataset settings
- `configs/preprocessing_config.json` — split boundaries and scaler metadata

### Data
- `data/raw/google_cluster_2019.csv` — original BigQuery export (894,263 rows)
- `data/processed/cleaned_sequences.parquet` — normalized, split sequences
- `data/processed/scaler.joblib` — fitted MinMaxScaler
- `data/processed/windows_train.npz` — 616,630 training windows
- `data/processed/windows_val.npz` — 133,017 validation windows
- `data/processed/windows_test.npz` — 130,616 test windows
- `data/processed/windows_metadata.json` — window counts and parameters

### Source
- `src/data_inspection.py` — EDA and profiling script
- `src/preprocessing.py` — cleaning, scaling, splitting
- `src/windowing.py` — sliding-window dataset construction

### Results & Reports
- `experiments/results/inspection_summary.json` — data inspection output
- `experiments/results/pipeline_setup_report.md` — this file

### Project Root
- `requirements.txt` — pinned pip dependencies (from `pip freeze`)
- `README.md` — project overview, setup guide, and pipeline instructions

---

## Issues Encountered and Resolutions

| Issue | Resolution |
|---|---|
| Window count slightly lower than inspection estimate | Expected: the inspection estimates assume full sequences; windowing clips at chronological split boundaries, reducing available steps at each boundary. No data loss — correct behavior. |
| CUDA dependency in requirements.txt (`+cu121` suffix) | Documented in README: users without a CUDA 12.1 GPU should replace torch/torchvision/torchaudio with CPU-only wheels. |
| No issues with data integrity | All 100 instances passed minimum sequence length checks; no instances were dropped during preprocessing. |

---

*Report generated at end of pipeline setup phase. The project is ready for model training.*
