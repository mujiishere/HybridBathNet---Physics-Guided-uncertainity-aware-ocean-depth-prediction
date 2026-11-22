"""Build deployable GeoTIFF products from the cross-validated ensemble.

Deployment ensemble = every member of every CV fold (each saw ~60 % of labelled blocks).
Conformal quantiles = fitted on the pooled spatially held-out calibration predictions of all folds.

python -m hbn2.make_products --runs runs_v2/cv_full --data data/sites --metrics reports_v2/study_metrics.json --out products_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import prepare
from .data import R
from .products import load_members, nearshore_extent, scene_layers, write_cogs
from .sites import SiteData
from .train import ConformalCalibrator

SITES = ["ponce_pr", "seribu_id", "langebaan_za"]  # override with --sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_v2/cv_full"); ap.add_argument("--data", default="data/v2_sites")
    ap.add_argument("--metrics", default="reports_v2/study_metrics.json"); ap.add_argument("--out", default="products_v2")
    ap.add_argument("--sites", nargs="+", default=SITES)
    ap.add_argument("--members-per-fold", type=int, default=1)
    ap.add_argument("--max-water-cells", type=int, default=1_500_000,
                    help="larger scenes are limited to a near-shore band (image-only) to bound inference cost")
    ap.add_argument("--nearshore-km", type=float, default=2.0)
    ap.add_argument("--skip-existing", action="store_true", help="reuse COGs already written for a site")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    members = sorted(p for p in runs.glob("fold*/member*.pt") if int(p.stem[6:]) < a.members_per_fold)
    models = load_members(members)
    def scores(pattern):
        zs = [np.load(p) for p in runs.glob(pattern)]
        return [np.concatenate([z[k] for z in zs]) for k in ("y", "mu", "sigma")] if zs else None

    pooled = ConformalCalibrator().fit(*scores("fold*/pred_test_*.npz"))
    cals = {}
    metrics = json.loads(Path(a.metrics).read_text()) if Path(a.metrics).exists() else {}
    site_metrics, site_meta = {}, {}
    for s in a.sites:
        ps = prepare(SiteData.load(Path(a.data) / f"{s}.npz"))
        sc = scores(f"fold*/pred_test_{s}.npz")
        # Mondrian-by-site cross-conformal on out-of-fold predictions (falls back to pooled)
        cal = ConformalCalibrator().fit(*sc) if sc is not None and len(sc[0]) >= 100 else pooled
        cals[s] = cal.to_dict()
        ext = None
        if (ps.water[R:-R, R:-R] > 0).sum() > a.max_water_cells:
            ext = nearshore_extent(ps, a.nearshore_km)
            ps.site.meta["product_extent"] = f"water within {a.nearshore_km} km of land"
        if not (a.skip_existing and (out / f"{s}_stack.tif").exists()):
            lay = scene_layers(models, ps, cal, extent=ext)
            tags = {"model": "HybridBathNetV2", "version": "2.0.0", "n_members": len(models),
                    "conformal": cal.to_dict(), "site": s, "reference": ps.site.meta.get("reference", "")}
            write_cogs(lay, ps, out, tags)
        m = metrics.get("hybridbathnet_v2", {}).get(s, {})
        acc, unc = m.get("accuracy", {}), m.get("uncertainty", {})
        site_metrics[s] = {**{k: acc.get(k) for k in ("n", "rmse", "mae", "bias", "r2", "within_1m", "within_2m")},
                           "picp90": unc.get("picp90_conformal"), "picp95": unc.get("picp95_conformal")}
        site_meta[s] = ps.site.meta
        print(s, "written")
    man = {"version": "2.0.0", "n_members": len(models), "members": [str(m) for m in members],
           "calibration": {"method": "cross-conformal on out-of-fold test predictions, Mondrian by site and depth bin", "per_site": cals, "pooled": pooled.to_dict()}, "sites": a.sites, "metrics": site_metrics, "site_meta": site_meta,
           "config": __import__("torch").load(members[0], map_location="cpu", weights_only=False)["cfg"]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    if Path(a.metrics).exists():
        (out / "study_metrics.json").write_text(Path(a.metrics).read_text())


if __name__ == "__main__":
    main()
