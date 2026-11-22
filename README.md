# HybridBathNet — Physics-Guided & Uncertainty-Aware Ocean Depth Prediction

GitHub publication (7 November 2026): published the accumulated source, experiment protocols, metrics, tests and study reports to origin/main in commit 049dcef (559 files). Raw satellite/tide/photon data, feature arrays, downloaded dependencies, experiment checkpoints, external downloads and logs remain local through .gitignore; existing tracked root checkpoint remains versioned. No training or result recalculation is launched. Checks: staged payload 39.5 MB, no credential-pattern matches, no whitespace errors apart from historical trailing blank lines; Git confirmed main -> main. No training was run for publication.

<!-- OOF_UNCERTAINTY_STATUS -->
GEBCO held-region uncertainty study: **complete**. [Status](runs/2026-11-06/gebco_oof_uncertainty_v1/status.json) | [Progress](runs/2026-11-06/gebco_oof_uncertainty_v1/LIVE_PROGRESS.md) | [Report](runs/2026-11-06/gebco_oof_uncertainty_v1/REPORT.md).
<!-- /OOF_UNCERTAINTY_STATUS -->

<!-- RECOVERY_V2_STATUS -->
Recovery study: **atl24: paired_training_complete, pairing: crops_ready, visibility: assessed_uncalibrated, uncertainty: complete, tides: validation_complete**. [Work directory](runs/2026-11-06/recovery_v2/) | [Final report](runs/2026-11-06/recovery_v2/REPORT.md).
<!-- /RECOVERY_V2_STATUS -->

Project study report **completed**: [read Markdown](reports/2026-11-06_project_study/HYBRIDBATHNET_STUDY_REPORT.md) | [printable HTML](reports/2026-11-06_project_study/HYBRIDBATHNET_STUDY_REPORT.html) | [all metrics CSV](reports/2026-11-06_project_study/evaluation_metrics.csv) | [layer-by-layer reference](reports/2026-11-06_project_study/LAYER_REFERENCE.md). It covers datasets, preprocessing, actual model layers/parameters/shapes, composite loss, depth/uncertainty metrics and evidence-based explanations with limitations. Full-model shallow test accuracy is54.67% within1m and93.33% within2m (RMSE1.264m,n75);0–20m RMSE2.561m uses228 points and is a different reporting domain. About92% of primary squared error comes from above10m. The report explains negative narrow-bin R², coarse GEBCO/reference effects and confounded comparisons without claiming causal proof.30 prediction input hashes and document/numerical checks pass. No retraining or historical result overwrite; no job active.

Latest reporting request completed: [GEBCO ablation and uncertainty tables up to10m](runs/2026-11-06/gebco_shallow_ablation_report_v1/REPORT.md) | [All metrics CSV](runs/2026-11-06/gebco_shallow_ablation_report_v1/all_results.csv). On the same75 shallow test points, matched full/no-physics/no-uncertainty/neither RMSE is1.264/1.373/1.665/1.887m. Uncertainty-off values are N/A; valid mean head SD is1.802m(full)/2.071m(no physics). Validation results are separate(n72). Later ensemble/OOF variants appear in a separate table because they used447 rather than637 training points. Saved predictions only; no retraining, calibration or test-mask change. Root historical metrics remain untouched.

Spatial OOF uncertainty study **completed**; no training remains active. Three fresh full folds supplied held-region residuals for all447 training locations, with fold-specific features and strict spatial/source-cell exclusions. A small total-risk correction head was selected at epoch4. [Results, bins and plots](runs/2026-11-06/gebco_oof_uncertainty_v1/REPORT.md) | [Per-location values](runs/2026-11-06/gebco_oof_uncertainty_v1/test_predictions.csv).

The raw learned head narrows mean90% intervals **8.71→7.14m validation** and **9.33→7.73m test** (17.2% test reduction), with coverage **90.4% validation/89.0% test**. Test interval score improves9.817→9.050m, and error ranking modestly improves.162→.187. The additionally scaled variant does not narrow test intervals (9.41m at97.8%). Both variants were predefined; raw head is a promising research option, not automatically deployed. Deep-bin coverages87.5/85.4% remain below90%, and spatial coverage uncertainty is broad. Depth/physics/source ensemble predictions and RMSE2.219/2.372m remain unchanged; raw aleatoric/epistemic components are preserved.39 tests, strict risk-head reload and original ensemble/fold hash checks pass. Preserved v1 remains the selected depth reference; no optical mask or test filtering was enabled. No further run queued.

Latest GEBCO uncertainty improvement study **completed**. Literature-based Gaussian sigma scaling, fitted only on123 independent calibration samples/15 groups, changes nominal90% coverage **80.3→93.8% validation** and **70.6→96.5% test**. Mean test interval width changes **5.30→9.33m** (prior simultaneous group-max intervals:17.12m). Test interval score improves11.037→9.817m, but its spatial interval includes zero and validation CRPS slightly worsens. The result is conservative, not perfect calibration or a uniform improvement. Physics/depth/head/ensemble weights and RMSE2.219/2.372m remain unchanged. Mean calibrated total test sigma is2.836m; raw aleatoric/epistemic components stay separate. Keep this as an explicit research option for the source ensemble; preserved v1 remains selected. [Full report/plots](runs/2026-11-06/gebco_uncertainty_calibration_v2/REPORT.md) | [Calibrated per-location values](runs/2026-11-06/gebco_uncertainty_calibration_v2/test_calibrated_uncertainty.csv) | [Paper-to-code note](docs/GEBCO_UNCERTAINTY_CALIBRATION_2026-11-06.md).37 tests and inference-state reload checks passed; no training is active. No optical mask or depth-error filtering was enabled.

Latest GEBCO uncertainty request completed: uncertainty-head aleatoric + three-model deep-ensemble epistemic decomposition. On 228 test centres at 0–20 m, **average per-location SD = 1.538 m aleatoric, 0.434 m epistemic, 1.612 m total**. Variances are combined per location before averaging SDs. Validation/test RMSE stays 2.219/2.372 m; raw90% test coverage is only 70.6%, so these are not reliable calibrated error bounds. Reused preserved independent seeds 42/1/7 with all HybridBathNet modules active; no new training or model promotion. [Bin-wise report and formulas](runs/2026-11-06/gebco_uncertainty_decomposition_v1/REPORT.md) | [Per-location test values](runs/2026-11-06/gebco_uncertainty_decomposition_v1/test_components.csv). Dataset/hash/order, finite variance, total-variance identity and saved-component checks passed; no training is active. Latest requested direction is GEBCO, not further ATL24 work.

Recovery completed Nov6 at19:32IST; **no training/download job remains active**. All five new runs retain physics, encoder/depth and uncertainty with the composite objective. [Results, bins and plots](runs/2026-11-06/recovery_v2/REPORT.md) | [Full handoff](PROJECT_CONTEXT.md) | [Manual commands](RUN_TRAINING_MANUALLY.md).

| Reference/arm | Validation RMSE0–20m | Test RMSE0–20m |
|---|---:|---:|
| Preserved GEBCO full v1 | 1.963m | 2.561m |
| New GEBCO ensemble3 | 2.219m | 2.372m |
| Official ATL24 V002 composite control | 6.027m | 6.512m |
| Official ATL24 V002 matched/tide | 6.713m | 6.093m |

GEBCO test centres/labels are unchanged(n228); ATL24 arms share their own171 labels/locations, so cross-reference RMSEs are not matched gains. Ensemble test bins2.136/2.173/2.687/2.295m improve deeper bins but worsen shallow bins and validation. ATL24 matched validation also worsens; only29 test points have historical clear imagery, and those points' RMSE worsens5.207→5.869m. Both overall paired95% improvement intervals include zero. **Keep the existing GEBCO full v1 selected.** Raw GEBCO90% coverage70.6%; whole-group calibration reaches100% at17.12m mean width, not useful precision. Optical classifier held-out support is zero, so automatic masking stays disabled. All74 official granules/29 paired historical crops downloaded; EOT20 relative water-level validation passed.31 tests and all five strict same-device reload checks pass. Historical data/results preserved. Future priorities: retain shallow support during calibration allocation, improve local-clear scene selection and reference/position uncertainty, and validate optical support locally before masking. No further job queued.

<!-- HEAD_RESEARCH_STATUS -->
Integrated physics/depth/uncertainty candidate: **complete**. Decoupled Gaussian variance, patch-shared effective attenuation; fixed GEBCO data/splits. [Status](runs/2026-11-06/integrated_head_research_v1/status.json) | [Comparison](runs/2026-11-06/integrated_head_research_v1/HEAD_COMPARISON.md).
<!-- /HEAD_RESEARCH_STATUS -->

<!-- GEBCO_ABLATION_STATUS -->
GEBCO nearshore component ablation: **no_physics: complete, no_uncertainty: complete, no_both: complete**. [Run directory](runs/2026-11-06/gebco_component_ablation_v1/) and [report](runs/2026-11-06/gebco_component_ablation_v1/REPORT.md).
<!-- /GEBCO_ABLATION_STATUS -->

<!-- NO_EXT_JOB_STATUS -->
No-extinction-penalty HybridBathNet ablation: **complete**. Same nearshore data/splits and all other joint losses. [Status](runs/2026-11-05/nearshore_no_ext_v2/status.json) | [Report](runs/2026-11-05/nearshore_no_ext_v2/REPORT.md).
<!-- /NO_EXT_JOB_STATUS -->

<!-- SQUARED_JOB_STATUS -->
Squared-depth HybridBathNet ablation: **complete**. Same nearshore data/splits and all other joint losses. [Status](runs/2026-11-05/nearshore_squared_depth_v3/status.json) | [Report](runs/2026-11-05/nearshore_squared_depth_v3/REPORT.md).
<!-- /SQUARED_JOB_STATUS -->

<!-- NEARSHORE_JOB_STATUS -->
Nearshore HybridBathNet: **complete**. Original test regions, shallow sampling, joint composite loss; no other models. [Status](runs/2026-11-05/nearshore_hybrid_v1/status.json) | [Report when complete](runs/2026-11-05/nearshore_hybrid_v1/REPORT.md).
<!-- /NEARSHORE_JOB_STATUS -->

<!-- JOINT_JOB_STATUS -->
Historical joint_v1 job **stopped at user request** during control epoch8; its full joint and Sentinel-only training never started. The later nearshore HybridBathNet run is separate and reported above. [Stop details](runs/2026-11-05/joint_v1/STOPPED.md).
<!-- /JOINT_JOB_STATUS -->

This repository contains the v3 baseline pipeline and model for **HybridBathNet**: an end-to-end framework for satellite-derived bathymetry (ocean depth prediction) from Sentinel-2 imagery, incorporating physics priors, uncertainty quantification, and a dense U-Net architecture.

## Current status — HybridBathNet v2 (updated 2026-11-22)

v2 is a four-site, independently referenced study with deployable GIS products; see **[README_V2.md](README_V2.md)**. Out-of-fold RMSE: Ponce (lidar) 1.26 m, R² 0.94; Kepulauan Seribu (sonar) 0.82 m; Langebaan (ICESat-2 ATL24) 0.97 m, vs Stumpf 3.03 / 1.10 / 1.40 m. On the turbid Kerala shelf (ATL24) no method is usable (v2 4.73 m, Stumpf 4.88 m), and the maps flag it as low confidence. Cross-conformal 90 % intervals cover 88.8–89.9 % at all four sites. Cross-site transfer without local soundings remains unsolved. Run everything with `scripts_v2/run_full_study.ps1`; dashboard via `uvicorn sdb_platform.api.main:app` or `docker compose -f sdb_platform/docker-compose.yml up`.

## v1 experimental status — updated 2026-11-06

**Recovery study authorized and underway:** ATL24 scene pairing/quality repair, independent optical visibility, and uncertainty calibration/ensembles are being executed in `runs/2026-11-06/recovery_v2/`. Public catalogue/model access is being checked using [fetcher](scripts/fetch_recovery_inputs.py); CUDA and disk space are available. No new training has started yet. [Current handoff](PROJECT_CONTEXT.md).

Public imagery range reads now work and the refreshed Earthdata token is downloading74 official V002 granules. A [new independent calibration split](runs/2026-11-06/recovery_v2/gebco_uq/features/metadata.json) contains447 train/123 calibration/208 validation/343 test centres, with feature fitting restricted to training footprints. [Three-seed full-model uncertainty study](scripts/run_recovery_uncertainty.py) is prepared with a richer variance branch and separate clipping; validation/test regions remain fixed.

**Physics/uncertainty research completed:** [Primary-source review and recovery design](docs/HEADS_OPTICAL_LIMIT_RESEARCH_2026-11-06.md) | [Integrated experiment results](runs/2026-11-06/integrated_head_research_v1/HEAD_COMPARISON.md) | [Maintained context](PROJECT_CONTEXT.md). One full integrated GEBCO candidate used decoupled variance supervision and patch-shared effective attenuation; best EMA136/stop166. It is preserved as an experiment, not promoted as a reliable improvement. No training is active.

| Measure, 0–20m | Existing full model | New integrated candidate |
|---|---:|---:|
| Validation RMSE | 1.963m | 1.961m |
| Test RMSE | 2.561m | 2.527m |
| Test Gaussian NLL | 2.596 | 2.634 |
| Test nominal90% coverage | 71.05% | 68.42% |

The paired95% spatial interval for the small test RMSE difference is[−0.217,+0.153]m. Validation NLL and held-out uncertainty worsened, so a joint gain is not established.26 tests, compilation and prediction-equivalent checkpoint reload checks passed. New multiband support/precision functions and corrected finite-sample calibration are available, but optical masking remains uncalibrated and disabled. Candidate inference requires its explicit protocol model_options.

The [frozen ATL24 audit](runs/2026-11-06/physics_uncertainty_research_v1/atl24_failure_audit.json) confirms correct depth arithmetic but median nearest Sentinel time gaps755/1065days (train/test), mostly single-photon labels, and ~7.6m reported horizontal uncertainty. A [75-pass historical-scene query manifest](runs/2026-11-06/physics_uncertainty_research_v1/atl24_scene_pairing_manifest.csv) prepares temporal repair for the exact AOI; no new scenes were downloaded. Next: validated optical visibility/background and temporally/datum-matched ATL24 pairs, followed by a richer uncertainty branch/ensembles with independent calibration groups. GEBCO and ATL24 scores use different targets/locations and cannot rank reference accuracy.

**GEBCO component ablation complete:** [Full report, depth bins and paired intervals](runs/2026-11-06/gebco_component_ablation_v1/REPORT.md) | [runner](scripts/run_gebco_component_ablation.py) | [maintained context](PROJECT_CONTEXT.md). The same selected nearshore GEBCO dataset and fixed geographic splits were used for all four rows. Validation/test RMSE below is for 0–20m (208/228 patch centres).

| Model | Validation RMSE (m) | Test RMSE (m) | Test MAE (m) |
|---|---:|---:|---:|
| Full physics + depth + uncertainty | **1.963** | 2.561 | 1.894 |
| No physics | 2.121 | 2.495 | 1.902 |
| No uncertainty | 2.033 | 2.467 | 1.905 |
| Neither | 2.068 | 2.491 | 1.874 |

The full model remains selected by validation. Test RMSE differences are small and each paired ~2km spatial-bootstrap 95% interval spans zero. The report contains per-bin results. Physics-off removes engineered feature channels, conditioning and optical losses; uncertainty-off removes NLL supervision, so its placeholder variance is not an uncertainty prediction. Single-seed test results were already explored in earlier studies; GEBCO is a coarse depth reference. No training is active. CUDA, 22 unit tests, compilation and test-reference/order checks passed; checkpoints and root historical metrics were preserved.

**ATL24 full-model run complete:** [Separate ATL24 report](runs/2026-11-06/atl24_hybrid_v1/REPORT.md) | [status](runs/2026-11-06/atl24_hybrid_v1/status.json) | [all results](runs/2026-11-06/atl24_hybrid_v1/results.json). Joint encoder/physics/depth/uncertainty training selected best EMA epoch60 and stopped at epoch90. **Validation 0–20m RMSE 3.443m** (112 centres); **test 4.741m** (233; 95% approximate2km group-bootstrap CI 3.460–5.832m). Test bins 0–5/5–10/10–15/15–20m: **4.287/2.356/4.873/10.811m** on **129/55/34/15** centres. Raw 90% intervals covered84.98% of test centres. The 16-channel Sentinel input used distinct ATL24 labels from a public 2018–Nov2024 V003-advertised AOI subset, **not GEBCO targets**. Four official V002 2024 AOI granules were also downloaded/verified but yielded only32 screened shallow bathymetry photons; the archive subset had 3,189 bathymetry-class points and some granule read alerts. The ATL24 dataset has 570/112/233 train/validation/test median-depth pixels after fixed geographic blocks and 96px buffers. **These scores are a different benchmark from GEBCO v1 RMSE2.561m**; the old data/model/results are untouched. No training job is active. [Current handoff](PROJECT_CONTEXT.md).

**Nearshore/reference audit:** [AOI-specific report](runs/2026-11-05/nearshore_error_reference_audit_v1/REPORT.md). Four ~2km groups contribute 76.4% of the original GEBCO test 10–15m squared error. Centre Sentinel SCL indicates fewer clear/more cloud-shadow dates in this bin, but clear high-error groups prevent attributing the error to clouds alone. The requested AOI is lon76.30–76.55, lat8.90–9.20. The initial 2024-only ATL24 query returned 95 bathymetry-class photons, too few for training; the later full-archive experiment is summarized above. Original GEBCO results remain intact.

**Optical-limit update:** [Audit and decision](runs/2026-11-05/optical_support_audit_v1/REPORT.md). A conservative lower-bound detector and anti-bypass calibration guard were added to [src/optical_limit.py](src/optical_limit.py). The current learned-optics/temporal-noise candidate retains only **7/208 validation** and **4/228 test** 0–20m predictions; median limit is0m, and the predefined coverage/spatial validation gate fails. **Operational masking remains disabled** and official full-set RMSE is unchanged. Deep-water controls vary substantially with geography/quality, so local `R_inf`, Kd and noise cannot yet be certified from this composite. Independent optical and depth calibration plus individual-scene quality work are next. Twenty-two tests pass; no training job is active. [Current context](PROJECT_CONTEXT.md).

**Planning update:** [Physics and uncertainty improvement plan](docs/PHYSICS_UNCERTAINTY_IMPROVEMENT_PLAN.md) traces the current combined Kd heuristic and learned per-band reconstruction attenuation, then prioritizes optical calibration, matched HybridBathNet component checks, independent depth data and geographic uncertainty validation. The user also requested a copyable prompt to transfer **data/preprocessing/training fixes** into the preserved six-input-channel version without changing its network architecture. The historical six-channel3.5066m result and nearshore16-channel2.5613m result are different benchmarks, so they are not a measured architecture improvement. No new experiment was launched; the completed v1 remains selected.

All three nearshore HybridBathNet-only training runs are complete; no training job is active. The original v1 remains selected by its lower validation RMSE. [Full follow-up report](runs/2026-11-05/nearshore_followup/REPORT.md) includes all test depth bins, cutoff sensitivity, matched spatial-bootstrap comparisons, uncertainty and physics assessment. [Maintained project context](PROJECT_CONTEXT.md) explains the experiment and next steps.

| Run | Change | Validation RMSE 0–20 m | Test RMSE 0–20 m |
|---|---|---:|---:|
| [Selected v1](runs/2026-11-05/nearshore_hybrid_v1/REPORT.md) | original joint composite | **1.963 m** | 2.561 m |
| [v2](runs/2026-11-05/nearshore_no_ext_v2/REPORT.md) | extinction penalty removed | 2.095 m | 2.550 m |
| [v3](runs/2026-11-05/nearshore_squared_depth_v3/REPORT.md) | squared depth term | 2.100 m | 2.590 m |

The v2 0.011 m lower test RMSE is within a paired spatial-bootstrap interval of −0.250 to +0.252 m and validation worsened. The selected v1 test-bin RMSE for 0–5/5–10/10–15/15–20 m is **1.094/1.310/3.255/2.800 m** on **17/58/64/89** patch centres. All test regions and difficult samples were retained. A sample is one overlapping 64x64 Sentinel-2 patch with a centre GEBCO label; these are not independent 10 m seabed soundings. The root results/hybridbathnet_results.json is historical.

The model jointly trains physics conditioning/reconstruction, depth and uncertainty. Selected v1 nominal90% uncertainty intervals cover71.05% of the 0–20 m test set; a validation-only interval scale raises this to78.07%, still under nominal. Held-out reflectance reconstruction is accurate, but the heuristic optical extinction limit is uncalibrated and is not used to clip predictions. All21 regression tests passed; no tabular baselines or Sentinel-only training were run in this follow-up. Older studies and checkpoints remain preserved.

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
