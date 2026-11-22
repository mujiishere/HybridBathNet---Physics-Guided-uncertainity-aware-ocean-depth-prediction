# HybridBathNet v2 — manuscript draft

Draft of 22 November 2026. Live, editable copy: Claude doc "HybridBathNet v2 — manuscript draft". All numbers come from `reports_v2/study_metrics.json` and `reports_v2/significance.json` (four-site study) and `reports_v2_3site/` (three-site transfer rows).

## Abstract

**Working title:** HybridBathNet v2: physics-guided deep ensembles with conformal calibration for operational satellite-derived bathymetry across lidar-, sonar- and ICESat-2-referenced coasts

Satellite-derived bathymetry (SDB) networks are usually validated on one site with random splits and return depths without usable confidence. We present HybridBathNet v2, a Sentinel-2 patch network that couples Lyzenga/Stumpf physics features and a differentiable Maritorena et al. (1994) shallow-water reflectance model with scene-conditioned (FiLM) convolution, a heteroscedastic deep ensemble and Mondrian cross-conformal calibration. We evaluate on four coasts with independent references — NOAA topobathy lidar (Ponce, Puerto Rico; 116,845 pixels), tide-corrected echo sounding (Kepulauan Seribu, Indonesia; 403), ICESat-2 ATL24 (Langebaan, South Africa; 475) and ICESat-2 ATL24 on the turbid Kerala shelf, India (819) — with 5-fold, 200 m spatial-block cross-validation and footprint buffers. Out-of-fold RMSE is 1.26 m (95 % CI 1.14–1.39; R² 0.94) at Ponce over 0–20 m, 0.82 m at Seribu and 0.97 m at Langebaan, against 3.03 / 1.10 / 1.40 m for a Stumpf model fitted on the same pixels. On the turbid Kerala shelf every method fails (v2 4.73 m, Stumpf 4.88 m; R² 0.28), but the intervals stay honest: cross-conformal 90 % intervals cover 88.8–89.9 % at all four sites, widening to 12.4 m at Kerala, whereas a single held-out calibration fold under-covers (80–84 %). Predictions are delivered as Cloud-Optimised GeoTIFF depth, uncertainty, interval and confidence-class layers through a PostGIS/FastAPI service and a planner dashboard.

**Keywords:** satellite-derived bathymetry; Sentinel-2; physics-guided deep learning; uncertainty quantification; conformal prediction; ICESat-2 ATL24; spatial cross-validation; coastal GIS

## 1. Introduction

The gap this paper closes is the step from an accurate SDB map to one a planner can act on: calibrated per-pixel intervals, an explicit flag where the seabed is optically invisible, and validation that does not leak space. Empirical log-ratio models (Stumpf et al., 2003; Lyzenga, 1978) need local soundings and degrade with bottom type and turbidity. CNNs exploit spatial context (Lumban-Gaol et al., 2022; Agrafiotis et al., 2024, MagicBathyNet) but are mostly reported per site, with random or weakly separated splits and without calibrated uncertainty. Physics-informed variants embed attenuation priors yet rarely test whether the resulting confidence is honest.

Our own v1 study on the Kerala coast exposed the same issues: GEBCO (~460 m, interpolated) was the reference, only 65 test patches existed, and nominal 68 / 90 % intervals covered 52 / 71 %. v2 is designed around those findings.

Contributions:

1. A physics-guided, scene-conditioned network that trains a differentiable shallow-water reflectance model jointly with depth, on labelled and unlabelled water, and outputs an optical detection limit per pixel.
2. A deep-ensemble + Mondrian cross-conformal pipeline that reaches near-nominal coverage at four sites with different reference technologies, and evidence that single-fold split conformal under-covers under spatial shift.
3. A leakage-safe multi-site benchmark (200 m blocks, footprint buffers, train-only fitting) on open lidar, sonar and ICESat-2 ATL24 references plus the turbid Kerala shelf, with baselines, ablations and leave-one-site-out transfer.
4. An open operational chain: COG products, a PostGIS schema, a FastAPI service and a planner dashboard with confidence classes, route profiles and draft screening.

## 2. Study sites and data

Four coasts with independent reference bathymetry were harmonised to Sentinel-2 B2/B3/B4/B8 surface reflectance on 10 m projected grids; water masks use imagery only (NDWI > 0, NIR < 0.08) and depths beyond 20 m are excluded from training and scoring.

| Site | Water type | Image (correction) | Reference | Labelled pixels ≤ 20 m | Depth range (m) |
| --- | --- | --- | --- | --- | --- |
| Ponce, Puerto Rico | Clear Caribbean, coral/seagrass | 2019-01-08 (Sen2Cor L2A) | NOAA topobathy lidar, tide-corrected, 10 m | 116,845 | 0.0–19.9 |
| Kepulauan Seribu, Indonesia | Reef lagoon, moderately turbid | L2A | 10,085 tide-corrected echo-sounding points, per-pixel median | 403 | 0.6–11.5 |
| Langebaan / Saldanha, South Africa | Temperate lagoon, sand/seagrass | 2024-11-28 (ACOLITE ρs) | ICESat-2 ATL24 class-40 photons, confidence ≥ 0.6, depth = −orthometric height | 475 | 0.7–10.8 |
| Kerala shelf (Alappuzha), India | Turbid monsoon shelf | Median of 5 Jan-2024 L2A tile-scenes (T43PFK/PFL, SCL-masked) | ICESat-2 ATL24 V002, 74 granules 2018–2024, per-pixel median | 819 | 0.6–20.0 |

Sources: Ponce from [Lumban-Gaol et al. (2022)](https://github.com/yustisiardhitasari/sdbcnn) (MIT); Seribu from the [SDB GUI sample data](https://github.com/rifqiharrys/sdb_gui) (MIT); Langebaan from [Payandeh et al. (2026)](https://github.com/nasa-jpl/Satellite-Derived-Bathymetry) with NASA ATL24 (doi:10.5067/ATLAS/ATL24.001). Files are SHA-256 pinned in `hbn2/download_open_data.py`.

Kerala is the v1 study area. Its ATL24 labels were screened once, before any training, by dropping returns shallower than 3 m where GEBCO is deeper than 10 m (offshore near-surface false bottoms; 99 of 1,637 removed); GEBCO is used for nothing else. Two-thirds of Kerala labels lie at 15–20 m, the images precede or follow the ATL24 passes by up to six years, and the two Sentinel-2 tiles join with a visible seam at the composite boundary.

## 3. Method

### 3.1 Image-only preprocessing

Sun glint is removed by Hedley et al. (2005) regression of each visible band on NIR over the darkest 10 % of water pixels. Deep-water reflectance R∞ is the 2nd percentile of low-glint water; a 16-value descriptor (5/25/50/95th percentiles of log reflectance per band) characterises each scene. No step reads reference depths.

### 3.2 Network

A 13 × 13 patch (130 m) predicts its centre depth. A parameter-free physics layer appends Lyzenga ln(Rᵢ − R∞ᵢ), deep-water-relative log reflectance, the Stumpf log-ratio and B/G, G/R log-ratios to log reflectance and the water mask (15 channels). Three residual blocks (width 32, GroupNorm, GELU, spatial dropout 0.1) are FiLM-modulated by the scene descriptor and R∞. The head returns mean depth μ, aleatoric variance σ²ₐ and four optical parameters (non-water attenuation K0, bottom albedo, sand/vegetation mix, radiometric gain); 78k parameters.

### 3.3 Physics-guided objective

The optical parameters and μ drive the Maritorena, Morel and Gentili (1994) model:

```latex
R(\lambda)=R_\infty(\lambda)+\big(A_b(\lambda)-R_\infty(\lambda)\big)\,e^{-2K_d(\lambda)z},\qquad K_d=K_w(\lambda)+K_0\,s(\lambda)
```

The loss is Huber(μ, y) + w(t)·β-NLL(β = 0.5) + 0.5·Huber(ln R̂, ln R), the last term on labelled pixels and an equal stream of unlabelled water. The detection limit is z_max = max over bands of ln(|A_b − R∞| / ε) / (2K_d), with ε the effective radiometric noise.

### 3.4 Uncertainty and calibration

Each fold trains a three-member deep ensemble; σ² = mean σ²ₐ + Var(μ). Intervals are normalised conformal, μ ± q̂σ, with Mondrian classes by site and predicted-depth bin (0–5, 5–10, > 10 m). The primary variant is cross-conformal: scores come from out-of-fold predictions of the other four spatial folds. Pixels are classed A (90 % half-width ≤ 1 m), B (≤ 2 m), C (wider) or D (μ > z_max or > 20 m).

### 3.5 Protocol

Sites are tiled into 200 m blocks assigned to five folds. For test fold f, fold f+1 is split into validation (early stopping after ≥ 60 % of a 25-epoch one-cycle schedule) and calibration halves. Training centres within a 6-pixel Chebyshev buffer of any held-out pixel are dropped, enforced by a unit test. Normalisation and the Stumpf / Lyzenga baselines are fitted on the same training pixels. Transfer uses leave-one-site-out training, zero-shot and with physics-only adaptation on unlabelled target imagery, plus 30-sounding linear recalibration versus Stumpf on the same soundings.

## 4. Results

On the three optically shallow sites HybridBathNet v2 cuts RMSE by 58 % (Ponce), 25 % (Seribu) and 31 % (Langebaan) against a Stumpf model fitted on identical training pixels, each paired difference excluding zero; on the turbid Kerala shelf no method is usable and v2's 0.14 m edge over Stumpf is not significant.

### 4.1 Accuracy (5-fold spatial-block CV, out-of-fold)

Out-of-fold RMSE (m) with 95 % spatial-block bootstrap intervals (chart in the live doc; values from `reports_v2/study_metrics.json`):

| Site | HybridBathNet v2 | Stumpf | Lyzenga |
| --- | --- | --- | --- |
| Ponce (lidar) | 1.26 (1.14–1.39) | 3.03 (2.87–3.20) | 3.91 (3.73–4.11) |
| Seribu (sonar) | 0.82 (0.40–1.15) | 1.10 (0.56–1.49) | 1.19 (0.51–1.73) |
| Langebaan (ATL24) | 0.97 (0.64–1.34) | 1.40 (1.16–1.66) | 1.34 (0.87–1.90) |
| Kerala (ATL24, turbid) | 4.73 (4.27–5.18) | 4.88 (4.50–5.29) | 5.15 (4.70–5.57) |

| Site | Model | MAE (m) | Bias (m) | R² | Within ±1 m (%) | Within IHO Order 2 TVU (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Ponce | HybridBathNet v2 | 0.86 | −0.14 | 0.94 | 70.8 | 71.9 |
| Ponce | Stumpf | 2.55 | 0.04 | 0.66 | 19.8 | 20.4 |
| Ponce | Lyzenga | 3.36 | −0.04 | 0.44 | 12.7 | 13.1 |
| Seribu | HybridBathNet v2 | 0.51 | −0.04 | 0.87 | 84.9 | 84.9 |
| Seribu | Stumpf | 0.79 | −0.20 | 0.76 | 75.2 | 75.2 |
| Seribu | Lyzenga | 0.74 | −0.33 | 0.72 | 80.6 | 80.6 |
| Langebaan | HybridBathNet v2 | 0.52 | −0.24 | 0.37 | 84.4 | 84.4 |
| Langebaan | Stumpf | 1.11 | 0.12 | −0.31 | 50.1 | 50.1 |
| Langebaan | Lyzenga | 0.80 | 0.19 | −0.20 | 75.2 | 75.4 |
| Kerala | HybridBathNet v2 | 3.43 | 0.14 | 0.28 | 21.1 | 22.2 |
| Kerala | Stumpf | 3.71 | 0.14 | 0.23 | 16.2 | 17.1 |
| Kerala | Lyzenga | 3.96 | 0.14 | 0.15 | 13.3 | 14.4 |

Paired RMSE difference v2 − Stumpf (95 % CI): Ponce −1.76 m (−1.97, −1.57), Seribu −0.28 m (−0.49, −0.08), Langebaan −0.43 m (−0.76, −0.11), Kerala −0.14 m (−0.50, +0.22). At Ponce error grows with depth: 0.32 / 0.59 / 1.11 / 1.46 / 1.60 m RMSE in the 0–2 / 2–5 / 5–10 / 10–15 / 15–20 m bins. At Kerala the pattern inverts: the 120 labels shallower than 10 m are over-predicted by about 7 m and the 549 labels at 15–20 m are under-predicted by 2.2 m, the signature of a model regressing to the mean where the seabed is not visible.

### 4.2 Uncertainty

| Metric | Ponce | Seribu | Langebaan | Kerala |
| --- | --- | --- | --- | --- |
| 90 % coverage, cross-conformal (%) | 89.9 | 88.8 | 88.8 | 89.4 |
| 95 % coverage, cross-conformal (%) | 94.8 | 93.3 | 94.3 | 94.4 |
| 90 % coverage, single-fold split conformal (%) | 83.8 | 83.9 | 82.1 | 79.9 |
| 90 % coverage, raw Gaussian ensemble σ (%) | 91.0 | 91.1 | 90.3 | 89.5 |
| Mean 90 % interval width (m) | 3.57 | 2.00 | 2.25 | 12.38 |
| Spearman σ vs \|error\| | 0.47 | 0.60 | 0.54 | 0.47 |
| RMSE of the 50 % lowest-σ cells (m) | 0.83 | 0.27 | 0.46 | 2.90 |
| NLL / CRPS | 1.47 / 0.63 | 0.72 / 0.37 | 0.94 / 0.39 | 2.81 / 2.41 |
| Mean aleatoric / epistemic SD (m) | 1.07 / 0.27 | 0.57 / 0.21 | 0.61 / 0.21 | 3.86 / 0.61 |

Cross-conformal coverage sits within 1.7 points of nominal at every site and level, against 52 / 71 % for the v1 model at 68 / 90 %. Calibrating on a single held-out fold under-covers by 6–10 points, because residual scale differs between spatial blocks.

The confidence classes separate usable from unusable depths. Held-out RMSE in class A is 0.52 m (Ponce), 0.32 m (Seribu) and 0.56 m (Langebaan), covering 18 %, 59 % and 54 % of cells. At Kerala no cell reaches class A or B: 97.7 % are class C (RMSE 4.77 m) and 2.3 % class D, so the map itself tells a planner not to rely on it.

### 4.3 Ablations (one member per fold, matched seeds and splits)

| Variant | Ponce RMSE (m) | Seribu RMSE (m) | Langebaan RMSE (m) | Kerala RMSE (m) | NLL Ponce / Seribu / Langebaan / Kerala |
| --- | --- | --- | --- | --- | --- |
| Full v2 (member 0) | 1.30 | 0.86 | 1.08 | 4.77 | 1.55 / 0.80 / 1.18 / 2.81 |
| No physics loss / head | 1.33 | 1.01 | 1.07 | 5.09 | 1.53 / 1.16 / 1.26 / 2.97 |
| Plain CNN (no physics features, FiLM or physics loss) | 1.38 | 1.04 | 1.18 | 4.94 | 1.61 / 1.10 / 1.56 / 3.19 |

With four sites the full model beats the plain CNN at every site, significantly at Ponce (−0.08 m, CI −0.15 to −0.02). The physics loss is significant at Seribu (−0.16 m, CI −0.31 to −0.01) and Kerala (−0.32 m, CI −0.63 to −0.01) and neutral at Langebaan. Gains range from 0.03 to 0.32 m; most of the improvement over empirical baselines comes from learned spatial context, while the physics terms mainly lower NLL.

### 4.4 Leave-one-site-out transfer

| Held-out site | Trained on | Zero-shot RMSE (m) | Physics-adapted RMSE (m) | 30 soundings: recalibrated v2 (m) | 30 soundings: Stumpf (m) |
| --- | --- | --- | --- | --- | --- |
| Ponce (0–20 m) | Seribu + Langebaan | 4.68 | 8.94 | 3.83 | 3.00 |
| Seribu | Ponce + Langebaan | 4.47 | 2.07 | 1.28 | 0.99 |
| Langebaan | Ponce + Seribu | 2.06 | 2.02 | 1.37 | 1.39 |
| Kerala | Ponce + Seribu + Langebaan | 6.05 | 11.54 | 4.99 | 5.05 |

Transfer without local labels is not solved. Physics-only adaptation on unlabelled target imagery halves Seribu error but collapses Ponce and Kerala toward the shallow depth range of the source sites. Zero-shot 90 % coverage falls to 2–26 %, so conformal quantiles must come from target-site soundings before maps are released. The first three rows come from the three-site study; the Kerala row from the four-site study.

## 5. Operational system

The deployed ensemble (one member per fold, five networks) maps every water pixel and writes per-site Cloud-Optimised GeoTIFFs: depth, σ, aleatoric and epistemic SD, 68/90/95 % conformal bounds, z_max and an 8-bit confidence class with an embedded colour table, plus a multi-band stack. Site-specific cross-conformal quantiles and model metadata are stored as GeoTIFF tags.

| Component | What it does | Verified |
| --- | --- | --- |
| PostGIS schema (`sdb_platform/db/schema.sql`) | site, model_run, product, depth_cell (1.68 M 10 m cells, GiST index), reference_obs, planner_aoi; `navigation_screen(site, draft, ukc)` SQL function | Nearest-cell lookup 0.18 ms; SQL and COG navigation screens agree cell-for-cell |
| Ingest (`sdb_platform/db/ingest.py`) | Idempotent COPY of COG layers and reference soundings | 4 sites, 1,679,257 cells (Kerala limited to water within 2 km of land) |
| FastAPI (`sdb_platform/api/main.py`) | XYZ PNG tiles from COGs (depth, σ, interval width, confidence, draft hazard), point query, route profile, polygon zonal + navigation statistics, downloads, metrics; works without a database | 15 automated tests pass |
| Dashboard (`sdb_platform/dashboard/index.html`) | MapLibre map with confidence overlays, chart-style sounding readout with intervals, route drawing with 90/95 % band profile, vessel draft + under-keel clearance screening | Rendered in headless Chrome; static single-file build for sharing |
| Deployment (`sdb_platform/docker-compose.yml`) | postgis/postgis:16-3.4 + ingest + API | Compose file; not run in this environment |

The hazard rule is conservative by construction: a cell is flagged when the shoal-side 95 % bound is shallower than draft plus clearance, and class-D cells are reported as unknown rather than clear.

## 6. Discussion and conclusion

The usable result is a within-site SDB model whose intervals can be trusted: 1.26 m RMSE over 0–20 m against lidar, under 1 m at the two lagoon sites, and 90 / 95 % intervals that cover 88.8–89.9 / 93.3–94.8 % of held-out cells at all four sites. That is what lets a planner treat class-A cells as a screening-grade sounding (0.3–0.6 m RMSE) and route around class C and D.

Kerala is the negative result the paper should keep. With 4.7 m RMSE and R² 0.28, neither v2 nor the empirical baselines map its turbid 15–20 m shelf from Sentinel-2; ATL24 dates span six years against a January-2024 composite. The calibrated intervals still cover 89 % and flag every cell as low confidence, which is the behaviour a planning tool needs.

What the evidence does not support, and the paper should say so:

- The physics terms are a modest accuracy contributor (0.03–0.32 m); the case for them is likelihood, interpretability (K0, albedo, z_max) and the Seribu adaptation gain.
- z_max is not trustworthy yet: conservative in clear water (Ponce cells beyond it had 0.71 m RMSE) and far too optimistic at Kerala (median 26 m on a turbid shelf). Its noise constants need per-sensor tuning or measured water clarity.
- Cross-site transfer needs local soundings, both for depth (30-sounding Stumpf still beats our recalibrated network at Ponce and Seribu) and for calibration.
- Seribu and Langebaan have 403 and 475 labelled cells, so their RMSE intervals are wide (0.40–1.15 and 0.64–1.34 m). ATL24 depths at Langebaan are referenced to the geoid, not the image-time water level.
- Each site is one image or composite; per-date training and tide-matched ATL24 pairs are the next additions.

Next experiments, in order: tide- and date-matched ATL24 pairs for Kerala restricted to the shallow (< 10 m) band, per-sensor tuning of the noise model behind z_max, a few-shot fine-tuning head rather than linear recalibration, and MagicBathyNet (Agia Napa, Puck Lagoon) as a fifth test site.

## 7. References and target journals

Suggested venues, in order of fit: IEEE JSTARS (operational remote sensing + uncertainty), International Journal of Applied Earth Observation and Geoinformation, Remote Sensing (MDPI, faster turnaround), Marine Geodesy.

- Agrafiotis, P., Janowski, Ł., Skarlatos, D., Demir, B. (2024). [MagicBathyNet: a multimodal remote sensing dataset for bathymetry prediction and pixel-based classification in shallow waters](https://arxiv.org/abs/2405.15477). IGARSS 2024.
- Angelopoulos, A. N., Bates, S. (2023). Conformal prediction: a gentle introduction. Foundations and Trends in Machine Learning 16(4).
- Hedley, J. D., Harborne, A. R., Mumby, P. J. (2005). Simple and robust removal of sun glint for mapping shallow-water benthos. International Journal of Remote Sensing 26(10).
- Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.
- [Lumban-Gaol, Y., Ohori, K. A., Peters, R. (2022)](https://doi.org/10.1080/01490419.2022.2091696). Extracting coastal water depths from multi-temporal Sentinel-2 images using convolutional neural networks. Marine Geodesy 45(6), 615–644.
- Lyzenga, D. R. (1978). Passive remote sensing techniques for mapping water depth and bottom features. Applied Optics 17(3).
- Magruder, L. A., Parrish, C. E., et al. (2025). [ATLAS/ICESat-2 L3A Along Track Coastal and Nearshore Bathymetry (ATL24), V1](https://www.earthdata.nasa.gov/fr/data/catalog/nsidc-cprd-atl24-001). NSIDC DAAC, doi:10.5067/ATLAS/ATL24.001.
- Maritorena, S., Morel, A., Gentili, B. (1994). Diffuse reflectance of oceanic shallow waters: influence of water depth and bottom albedo. Limnology and Oceanography 39(7).
- Parrish, C. E., et al. (2025). Analysis and accuracy assessment of a new global nearshore ICESat-2 bathymetric data product. Earth and Space Science.
- [Payandeh, A. R., Simard, M., et al. (2026)](https://doi.org/10.3389/frsen.2026.1751006). A fully satellite-driven workflow for hydrodynamic modeling in data-scarce coastal systems. Frontiers in Remote Sensing 7, 1751006.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A. (2018). FiLM: visual reasoning with a general conditioning layer. AAAI.
- Seitzer, M., Tavakoli, A., Antic, D., Martius, G. (2022). On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks. ICLR.
- Stumpf, R. P., Holderied, K., Sinclair, M. (2003). Determination of water depth with high-resolution satellite imagery over variable bottom types. Limnology and Oceanography 48(1).
- Vovk, V. (2015). Cross-conformal predictors. Annals of Mathematics and Artificial Intelligence 74.
- IHO (2020). S-44 Standards for Hydrographic Surveys, Edition 6.0.

Code and data: `hbn2/`, `sdb_platform/`, `reports_v2/` in this repository. Reference details marked without links are from memory; verify volume and page numbers before submission.
