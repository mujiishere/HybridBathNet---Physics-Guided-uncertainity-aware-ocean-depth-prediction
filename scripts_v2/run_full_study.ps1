# HybridBathNet v2 — full reproducible pipeline on the Windows workstation (existing venv).
# Run from the repository root in PowerShell:  .\scripts_v2\run_full_study.ps1
# Optional: add the original Kerala AOI as a 4th site (needs the local Sentinel stack + ATL24 profiles).
param(
  [string]$KeralaBands = "C:\Users\Akash A\final year project\HybridBathNet---Physics-Guided-uncertainity-aware-ocean-depth-prediction\_hbn2_extract",
  [string]$KeralaGebco = "C:\Users\Akash A\final year project\HybridBathNet---Physics-Guided-uncertainity-aware-ocean-depth-prediction\data\raw\gebco\gebco_subset.tif",
  [string]$KeralaAtl24 = "runs\2026-11-06\recovery_v2\official_profiles.csv",
  [int]$Members = 3
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = (Get-Location).Path
$py = if (Test-Path ".\venv\Scripts\python.exe") { ".\venv\Scripts\python.exe" } else { "python" }

& $py -m pip install -q rasterio pyproj geopandas xarray netCDF4 scipy scikit-learn matplotlib rio-cogeo fastapi "uvicorn[standard]" pillow "psycopg[binary]" shapely pytest

# 1. open multi-site data (SHA-256 verified) -> data/v2_sites/*.npz
& $py -m hbn2.download_open_data --raw data/v2_raw --out data/v2_sites
$sites = @("ponce_pr", "seribu_id", "langebaan_za")

# 2. Kerala site: Jan-2024 L2A band files (<date>_<tile>_<band>.jp2) + ATL24 V002 profiles; GEBCO only screens offshore false-bottom photons
if ((Test-Path $KeralaBands) -and (Test-Path $KeralaAtl24)) {
  & $py -m hbn2.kerala_build --bands $KeralaBands --atl24 $KeralaAtl24 --gebco $KeralaGebco --out data/v2_sites/kerala_in.npz
  $sites += "kerala_in"
}

# 3. pooled 5-fold spatial-block CV (3-member deep ensemble per fold), LOSO transfer and ablations
& $py -m hbn2.run_study cv --data data/v2_sites --sites $sites --config full --members $Members --out runs_v2/cv_full
foreach ($t in $sites) {
  & $py -m hbn2.run_study loso --data data/v2_sites --sites $sites --target $t --members 2 --out runs_v2/loso
  & $py -m hbn2.run_study loso --data data/v2_sites --sites $sites --target $t --adapt --members 2 --out runs_v2/loso
}
foreach ($c in @("cnn_only", "no_physics_loss", "no_physics_features", "no_film")) {
  & $py -m hbn2.run_study cv --data data/v2_sites --sites $sites --config $c --members 1 --out "runs_v2/abl_$c"
}

# 4. tables/figures, GeoTIFF products, static dashboard, tests
& $py -m hbn2.evaluate --runs runs_v2 --data data/v2_sites --out reports_v2
& $py -m hbn2.make_products --runs runs_v2/cv_full --data data/v2_sites --metrics reports_v2/study_metrics.json --out products_v2 --sites $sites
& $py sdb_platform/build_static_dashboard.py --products products_v2 --out reports_v2/dashboard_static.html
& $py -m pytest -q tests/test_hbn2.py tests/test_platform_api.py

# 5. serve: file mode (no DB) ...
Write-Host "Dashboard: set HBN_PRODUCTS=products_v2 then: $py -m uvicorn sdb_platform.api.main:app --port 8000"
# ... or full stack with PostGIS:  docker compose -f sdb_platform/docker-compose.yml up --build
