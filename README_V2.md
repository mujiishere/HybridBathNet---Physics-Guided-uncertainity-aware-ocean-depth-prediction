# HybridBathNet v2 — physics-guided, uncertainty-aware SDB with a planner-facing GIS

HybridBathNet v2 is the final-year-project system that extends the v1 Kerala study (GEBCO/ATL24, see
`PROJECT_CONTEXT.md`) into a **multi-site, independently referenced** study with deployable products.
v1 code, checkpoints and results are untouched; everything new lives in `hbn2/`, `sdb_platform/`,
`scripts_v2/`, `tests/test_hbn2.py`, `tests/test_platform_api.py`, `runs_v2/`, `reports_v2/`, `products_v2/`.

## What changed versus v1 and why

| v1 limitation (from the v1 audits) | v2 answer |
|---|---|
| GEBCO (~460 m, interpolated) used as "truth"; only 65 test patches | Four sites with independent references: NOAA topobathy **lidar** (Ponce, Puerto Rico), tide-corrected **echo-sounding** (Kepulauan Seribu, Indonesia), **ICESat-2 ATL24** photons (Langebaan, South Africa) and recovered **ATL24 V002** on the turbid Kerala shelf (`hbn2/kerala_build.py`). 3-site results are preserved in `*_3site/`. |
| Uncertainty under-calibrated (68/90 % intervals covered 52/71 %) | Deep ensemble (aleatoric + epistemic) **+ Mondrian split-conformal calibration** on spatially held-out calibration blocks, giving finite-sample coverage guarantees. |
| Physics prior heuristic, reconstruction branch unused | Differentiable **Maritorena–Morel–Gentili (1994)** shallow-water reflectance model trained jointly with depth; yields a per-pixel **optical detection limit z_max** used to flag optically deep / extrapolated pixels. Physics loss also runs on **unlabelled** water pixels, enabling label-free adaptation to a new site. |
| Single scene, single region | Scene-conditioned network (FiLM on image-only water-type statistics), **leave-one-site-out** transfer, few-shot recalibration. |
| No products/platform | COG GeoTIFF depth + uncertainty + interval + confidence-class layers, **PostGIS** store, **FastAPI** service, **MapLibre dashboard** with confidence overlays, route profiles and vessel-draft screening. |

## Model (hbn2/model.py, hbn2/physics.py)

Input: 13×13 patch (130 m) of Hedley-deglinted B2/B3/B4/B8 reflectance + water mask.
Physics feature layer: Lyzenga log-transforms ln(R−R∞), deep-water-relative log reflectance, Stumpf log-ratio, B/G and G/R log-ratios.
Scene encoder: 20 image-only statistics (water reflectance percentiles, R∞) → FiLM modulation of 3 residual conv blocks (width 32).
Heads: mean depth μ (softplus), aleatoric variance, and optical parameters (K0, bottom albedo, sand/vegetation mix, gain) that drive

R(λ, z) = R∞(λ) + (A_b(λ) − R∞(λ)) · exp(−2 K_d(λ) z),  K_d = K_w(λ) + K0·s(λ)

Loss: Huber + β-NLL (β = 0.5) on labelled centres + 0.5 × log-reflectance reconstruction (labelled and unlabelled pixels).
Uncertainty: 3-member deep ensemble per fold → σ² = mean aleatoric + ensemble variance → normalised split conformal with depth-bin (Mondrian) quantiles.

Confidence classes for planners: **A** 90 % half-width ≤ 1 m, **B** ≤ 2 m, **C** wider, **D** μ > z_max or > 20 m (do not use).

## Evaluation protocol (hbn2/data.py, hbn2/run_study.py, hbn2/evaluate.py)

* 200 m spatial blocks, 5 folds per site, assigned before training; for each test fold the next fold is split block-wise into validation (early stopping) and calibration (conformal) halves.
* Training centres within a 6-pixel Chebyshev buffer (the patch footprint) of any held-out pixel are removed; a unit test enforces this.
* Normalisation and empirical baselines (Stumpf, Lyzenga) are fitted on training pixels only. Test folds are never used for any choice.
* Metrics: RMSE (spatial-block bootstrap 95 % CI), MAE, bias, NMAD, R², % within ±0.5/1/2 m, IHO S-44 Order-2 TVU share, per-depth-bin RMSE; NLL, CRPS, ECE, PICP/MPIW at 68/90/95 %, Spearman(σ,|e|), AUSE.

## Repository map

- `docs_v2/MANUSCRIPT_DRAFT.md` — journal manuscript draft (4-site results)
- `docs_v2/REPRODUCE.md` — what is committed, what is regenerated, and the exact commands
- `requirements_v2.txt` — dependencies for `hbn2/` and `sdb_platform/`
- `runs_v2/` 4-site study, `runs_v2_3site/` 3-site study (models + out-of-fold predictions)
- `reports_v2/` metrics, significance, figures, static dashboard
- `products_v2/` manifest (COG maps are regenerated locally, not committed)

## Run everything

```powershell
.\scripts_v2\run_full_study.ps1            # data -> CV -> LOSO -> ablations -> reports -> COGs -> tests
$env:HBN_PRODUCTS="products_v2"; python -m uvicorn sdb_platform.api.main:app --port 8000   # dashboard at http://localhost:8000
docker compose -f sdb_platform/docker-compose.yml up --build                              # with PostGIS
```

API: `GET /api/sites`, `GET /api/sites/{site}/stats`, `GET /tiles/{site}/{layer}/{z}/{x}/{y}.png`
(layers: depth, sigma, aleatoric_sd, epistemic_sd, width90, confidence, hazard?draft=), `GET /api/query/point`,
`POST /api/query/profile` (route), `POST /api/query/polygon` (zonal + navigation), `GET /api/navigation/{site}`,
`GET /api/sites/{site}/download/{layer}`, `GET /api/metrics`. Interactive docs at `/docs`.

## Data licences

Ponce: MIT (Lumban-Gaol et al. 2022). Seribu: MIT (sdb_gui sample). Langebaan: Caltech/JPL research example
(Payandeh et al. 2026) + NASA ATL24 (open). Cite each source; see `hbn2/download_open_data.py`.
