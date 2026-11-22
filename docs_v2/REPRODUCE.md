# Reproducing HybridBathNet v2

## What is in git and what is not

| Path | In git | How to regenerate if missing |
| --- | --- | --- |
| `hbn2/`, `sdb_platform/`, `scripts_v2/`, `tests/test_hbn2.py`, `tests/test_platform_api.py` | yes | — |
| `runs_v2/` (4-site study) and `runs_v2_3site/` (3-site study): trained ensemble members (`*.pt`, ~0.3 MB each), out-of-fold predictions (`pred_*.npz`), logs, `done.json` | yes | `python -m hbn2.run_study ...` (see `scripts_v2/run_full_study.ps1`) |
| `reports_v2/`, `reports_v2_3site/`: metrics JSON, significance JSON, figures, static dashboard | yes | `python -m hbn2.evaluate`, `python -m hbn2.significance`, `python sdb_platform/build_static_dashboard.py` |
| `products_v2/manifest.json`, `products_v2/study_metrics.json` | yes | — |
| `products_v2/*.tif`, `products_v2_3site/*.tif` (COG maps, ~250 MB) | **no** (gitignored) | `python -m hbn2.make_products --runs runs_v2/cv_full --metrics reports_v2/study_metrics.json --out products_v2 --sites ponce_pr seribu_id langebaan_za kerala_in` |
| `data/v2_sites/*.npz` (harmonised site files, up to 61 MB) | **no** | `python -m hbn2.download_open_data` (3 open sites); Kerala: `python -m hbn2.kerala_build --bands <_hbn2_extract> --atl24 runs/2026-11-06/recovery_v2/official_profiles.csv --gebco <gebco_subset.tif> --out data/v2_sites/kerala_in.npz` |
| Raw Sentinel-2 Kerala bands (`_hbn2_extract/*.jp2`) | **no** | local copy in `C:\Users\Akash A\final year project\HybridBathNet...\_hbn2_extract` |

## Commands (PowerShell, repository root)

```powershell
pip install -r requirements_v2.txt
$env:PYTHONPATH = (Get-Location).Path

# full pipeline (data -> CV -> LOSO -> ablations -> reports -> COGs -> tests)
.\scripts_v2\run_full_study.ps1

# only re-derive tables/figures from the committed predictions (no training)
python -m hbn2.evaluate --runs runs_v2 --data data/v2_sites --out reports_v2
python -m hbn2.significance

# maps + dashboard
python -m hbn2.make_products --runs runs_v2/cv_full --metrics reports_v2/study_metrics.json --out products_v2 --sites ponce_pr seribu_id langebaan_za kerala_in
$env:HBN_PRODUCTS = "products_v2"; python -m uvicorn sdb_platform.api.main:app --port 8000   # http://localhost:8000
docker compose -f sdb_platform/docker-compose.yml up --build                               # with PostGIS

# tests
python -m pytest -q tests/test_hbn2.py tests/test_platform_api.py
```

`hbn2.evaluate` and `hbn2.significance` need `data/v2_sites/*.npz` (for block ids and baselines); build them first.

## Reported numbers

Four-site out-of-fold results are in `reports_v2/study_metrics.json`; the manuscript draft is `docs_v2/MANUSCRIPT_DRAFT.md`. Training ran on CPU (2 threads); a 3-member ensemble per fold took about 4–5 minutes per member.
