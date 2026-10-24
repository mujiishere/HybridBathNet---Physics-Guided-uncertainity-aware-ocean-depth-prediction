# HybridBathNet — Physics-Guided & Uncertainty-Aware Ocean Depth Prediction

This repository contains the v3 baseline pipeline and model for **HybridBathNet**: an end-to-end framework for satellite-derived bathymetry (ocean depth prediction) from Sentinel-2 imagery, incorporating physics priors, uncertainty quantification, and a dense U-Net architecture.

It includes:
1. **Stumpf (2003)** empirical Lyzenga-feature log-band-ratio regression.
2. **Random Forest** on Lyzenga bands, band ratios, and NDWI.
3. **Gradient Boosting** — stronger tabular baseline.
4. **HybridBathNet v3** — Physics-guided & uncertainty-aware hybrid U-Net.

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/mujiishere/HybridBathNet---Physics-Guided-uncertainity-aware-ocean-depth-prediction.git
cd HybridBathNet---Physics-Guided-uncertainity-aware-ocean-depth-prediction
git lfs pull

python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set Copernicus Credentials
```powershell
# Windows PowerShell:
$env:CDSE_USERNAME="your-email@example.com"
$env:CDSE_PASSWORD="your-password"
```
```bash
# Linux / macOS:
export CDSE_USERNAME="your-email@example.com"
export CDSE_PASSWORD="your-password"
```
Free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).

### 3. Run Full Pipeline
```bash
# Real data (downloads required):
python run_baseline_pipeline.py --config config.yaml

# Synthetic data (no downloads needed, for testing):
python run_baseline_pipeline.py --config config.yaml --synthetic

# Skip download if data already present:
python run_baseline_pipeline.py --config config.yaml --skip-download

# Run with multiple seeds (§8 ablation):
python run_baseline_pipeline.py --config config.yaml --seed 0
python run_baseline_pipeline.py --config config.yaml --seed 1
```

---

## 🛠️ Step-by-Step Pipeline

### Step 1: Download Datasets
```bash
python -m src.download_sentinel2 --config config.yaml
python -m src.download_gebco     --config config.yaml
python -m src.download_coastline --config config.yaml
```

### Step 2: Preprocess
```bash
# Applies Sentinel-2 L2A BOA offset (§1.3), builds median composite + temporal std
python -m src.preprocess --config config.yaml
```

### Step 3: Feature Engineering
```bash
# Lyzenga channels (§2.1), sun-glint correction (§2.3), mask channel (§2.2),
# temporal std (§2.4), buffered spatial split (§1.5)
python -m src.features --config config.yaml
```

### Step 4: Train & Evaluate
```bash
# Stumpf (Lyzenga) + RandomForest + GradientBoosting
python -m src.baseline_models --config config.yaml

# HybridBathNet v3 (U-Net, FiLM, β-NLL, EMA, AMP, snapshots, calibration)
python -m src.cnn_baseline --config config.yaml

# Comparison table + RMSE/depth-bin/calibration plots
python -m src.compare_baselines --config config.yaml
```

---

## 📁 Repository Structure

```
hybridbathnet_baseline/
├── config.yaml                     # Hyperparameters, paths, loss weights
├── requirements.txt                # Python dependencies
├── run_baseline_pipeline.py        # End-to-end pipeline orchestrator
├── monitor_and_run.py              # Live monitoring runner
├── README.md
├── src/
│   ├── utils.py                    # Geo utilities & logging
│   ├── download_sentinel2.py       # CDSE imagery downloader
│   ├── download_gebco.py           # GEBCO bathymetry converter
│   ├── download_coastline.py       # OSM coastline downloader
│   ├── preprocess.py               # BOA offset, compositing, temporal std
│   ├── features.py                 # Lyzenga, glint corr., mask, split
│   ├── baseline_models.py          # Stumpf, RF, GradientBoosting
│   ├── cnn_baseline.py             # HybridBathNet v3 training loop
│   ├── hybridbathnet.py            # Model architecture (U-Net + FiLM)
│   ├── generate_synthetic_data.py  # Synthetic data generator
│   └── compare_baselines.py        # Benchmark plots & comparison table
├── checkpoints/                    # Pre-trained weights (Git LFS)
│   ├── hybridbathnet.pt            # Best EMA checkpoint
│   ├── hybridbathnet_snapshot_*.pt # Cosine-cycle snapshots (ensemble)
│   ├── rf_model.joblib
│   ├── gb_model.joblib
│   └── stumpf_model.joblib
├── results/                        # Evaluation JSON + CSV + plots
│   ├── hybridbathnet_results.json
│   ├── tabular_baseline_results.json
│   ├── cnn_baseline_results.json
│   ├── baseline_comparison.csv
│   ├── baseline_comparison.png
│   ├── depth_bin_comparison.png
│   └── uncertainty_calibration.png
└── outputs/                        # Output rasters & diagnostic charts
```

---

## 🔬 v3 Changes at a Glance

| § | Fix | File |
|---|-----|------|
| 1.1 | σ in metres (was normalised units) | `cnn_baseline.py`, `hybridbathnet.py` |
| 1.2 | Kd from real reflectance (not z-scored) | `hybridbathnet.py` |
| 1.3 | Sentinel-2 L2A BOA offset −1000/10000 | `preprocess.py` |
| 1.4 | epochs=90 = 3×T_0=30 (no mid-cycle cutoff) | `config.yaml` |
| 1.5 | Buffered spatial split (no edge leakage) | `features.py` |
| 2.1 | Lyzenga log-linearised channels | `features.py` |
| 2.2 | Water/valid mask input channel | `features.py` |
| 2.3 | NIR sun-glint correction | `features.py` |
| 2.4 | Temporal std channel (turbidity proxy) | `preprocess.py`, `features.py` |
| 2.5 | Optional distance-to-shore channel | `features.py` |
| 3.1 | Dense U-Net output (full H×W maps) | `hybridbathnet.py` |
| 3.2 | Per-pixel Kd map | `hybridbathnet.py` |
| 3.3 | FiLM conditioning at every encoder scale | `hybridbathnet.py` |
| 3.4 | Removed non-physical feature bounding | `hybridbathnet.py` |
| 3.5 | softplus depth (non-negativity by design) | `hybridbathnet.py` |
| 3.6 | Optical reconstruction head (self-supervised) | `hybridbathnet.py` |
| 3.7 | GEBCO-scale pooled loss | `hybridbathnet.py` |
| 4.1 | One compensation: sampler OR weighting | `cnn_baseline.py`, `config.yaml` |
| 4.2 | β-NLL (β=0.5, no variance collapse) | `hybridbathnet.py` |
| 4.3 | Huber instead of MSE for mean term | `hybridbathnet.py` |
| 4.5 | EMA weights, early stopping | `cnn_baseline.py` |
| 5.2 | Snapshot ensemble (free, per cycle) | `cnn_baseline.py` |
| 5.4 | Calibration: NLL, 68/90% cov, sparsification | `hybridbathnet.py`, `cnn_baseline.py` |
| 5.5 | Conformal recalibration helper | `hybridbathnet.py` |
| 6.3 | AMP (bfloat16/fp16) training | `cnn_baseline.py` |
| 7   | Pipeline, compare_baselines, credentials | `run_baseline_pipeline.py`, `compare_baselines.py`, `download_sentinel2.py` |
| 8   | Tabular: Lyzenga features, bootstrap CI | `baseline_models.py` |

---

## 📊 Results (Baseline, test split vs GEBCO)

| Model | RMSE (m) | R² | ±1 m | ±2 m |
|-------|----------|-----|------|------|
| HybridBathNet v1 | 3.618 | — | 20.64% | 41.38% |
| HybridBathNet v2 | 3.507 | 0.916 | 23.16% | 44.67% |
| HybridBathNet v3 | *run to measure* | | | |

---

## ⚠️ Notes

- **Security**: Never commit Copernicus credentials. Always use `CDSE_USERNAME` / `CDSE_PASSWORD` env vars.
- **GPU**: PyTorch uses CUDA automatically if available. AMP (bfloat16/fp16) is enabled by default.
- **Git LFS**: Pre-trained models are tracked with `git lfs`. Run `git lfs pull` after cloning.
- **Ablations**: Use `--no-rotations`, `--no-amp`, `--seed N` flags for §8 ablation runs.
