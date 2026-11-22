"""Load HybridBathNet v2 COG products + reference observations into PostGIS.

  python -m platform_ingest --dsn postgresql://sdb:sdb@localhost:5432/sdb --products products_v2 --sites data/v2_sites
(or: python platform/db/ingest.py ...)

Idempotent: re-running replaces the cells/products of the same (site, run version).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import psycopg
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds

LAYER_COLS = {"depth": "depth_m", "sigma": "sigma_m", "aleatoric_sd": "aleatoric_sd", "epistemic_sd": "epistemic_sd",
              "lower90": "lower90", "upper90": "upper90", "lower95": "lower95", "upper95": "upper95",
              "zmax": "zmax_m", "confidence": "confidence"}


def ensure_schema(conn):
    conn.execute((Path(__file__).parent / "schema.sql").read_text())


def upsert_run(conn, manifest):
    r = conn.execute(
        """INSERT INTO sdb.model_run (model_name, version, n_members, config, calibration, metrics)
           VALUES ('HybridBathNetV2', %s, %s, %s, %s, %s)
           ON CONFLICT (model_name, version) DO UPDATE SET n_members=EXCLUDED.n_members, config=EXCLUDED.config,
             calibration=EXCLUDED.calibration, metrics=EXCLUDED.metrics RETURNING run_id""",
        (manifest["version"], manifest.get("n_members"), json.dumps(manifest.get("config", {})),
         json.dumps(manifest.get("calibration", {})), json.dumps(manifest.get("metrics", {})))).fetchone()
    return r[0]


def _first_date(v):
    import re
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(v or ""))
    return m.group(0) if m else None


def upsert_site(conn, name, meta, crs, bounds4326):
    w, s, e, n = bounds4326
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    r = conn.execute(
        """INSERT INTO sdb.site (name, region, native_crs, reference_type, water_type, image_date, footprint, meta)
           VALUES (%s,%s,%s,%s,%s,%s, ST_GeomFromText(%s,4326), %s)
           ON CONFLICT (name) DO UPDATE SET region=EXCLUDED.region, native_crs=EXCLUDED.native_crs,
             reference_type=EXCLUDED.reference_type, water_type=EXCLUDED.water_type, image_date=EXCLUDED.image_date,
             footprint=EXCLUDED.footprint, meta=EXCLUDED.meta RETURNING site_id""",
        (name, meta.get("region"), crs, meta.get("reference_type"), meta.get("water_type"), _first_date(meta.get("image_date")),
         poly, json.dumps(meta))).fetchone()
    return r[0]


def ingest_site(conn, run_id, name, prod_dir: Path, site_npz: Path | None, stride: int = 1):
    paths = {l: prod_dir / f"{name}_{l}.tif" for l in LAYER_COLS if (prod_dir / f"{name}_{l}.tif").exists()}
    with rasterio.open(paths["depth"]) as src:
        crs, tr, H, W = src.crs.to_string(), src.transform, src.height, src.width
        b4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        tags = src.tags()
    meta = {}
    if site_npz and site_npz.exists():
        meta = json.loads(str(np.load(site_npz)["meta"]))
    site_id = upsert_site(conn, name, meta, crs, b4326)
    arr = {}
    for l, p in paths.items():
        with rasterio.open(p) as src:
            arr[l] = src.read(1)
            nod = src.nodata
        conn.execute("""INSERT INTO sdb.product (site_id, run_id, layer, uri, units, stats) VALUES (%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (site_id, run_id, layer) DO UPDATE SET uri=EXCLUDED.uri, stats=EXCLUDED.stats""",
                     (site_id, run_id, l, str(p.resolve()), "class" if l == "confidence" else "m",
                      json.dumps({"min": float(arr[l][arr[l] != nod].min()), "max": float(arr[l][arr[l] != nod].max())})))
    valid = arr["depth"] != -9999
    if stride > 1:
        m = np.zeros_like(valid); m[::stride, ::stride] = True; valid &= m
    rr, cc = np.where(valid)
    xs, ys = rasterio.transform.xy(tr, rr, cc)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    conn.execute("DELETE FROM sdb.depth_cell WHERE site_id=%s AND run_id=%s", (site_id, run_id))
    cols = ["site_id", "run_id", "row_i", "col_j", "geom"] + [LAYER_COLS[l] for l in paths]
    with conn.cursor() as cur:
        with cur.copy(f"COPY sdb.depth_cell ({','.join(cols)}) FROM STDIN") as cp:
            for i in range(len(rr)):
                vals = [site_id, run_id, int(rr[i]), int(cc[i]), f"SRID=4326;POINT({lon[i]:.7f} {lat[i]:.7f})"]
                for l in paths:
                    v = arr[l][rr[i], cc[i]]
                    vals.append(int(v) if l == "confidence" else (None if v == -9999 else float(v)))
                cp.write_row(vals)
    # reference observations
    if site_npz and site_npz.exists():
        z = np.load(site_npz)
        d, cnt = z["depth"], z["count"]
        r2, c2 = np.where(np.isfinite(d))
        if len(r2):
            if len(r2) > 50000:
                k = np.random.default_rng(0).choice(len(r2), 50000, replace=False); r2, c2 = r2[k], c2[k]
            x2, y2 = rasterio.transform.xy(tr, r2, c2)
            lo2, la2 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.asarray(x2), np.asarray(y2))
            conn.execute("DELETE FROM sdb.reference_obs WHERE site_id=%s", (site_id,))
            with conn.cursor() as cur:
                with cur.copy("COPY sdb.reference_obs (site_id, geom, depth_m, n_raw, source) FROM STDIN") as cp:
                    for i in range(len(r2)):
                        cp.write_row([site_id, f"SRID=4326;POINT({lo2[i]:.7f} {la2[i]:.7f})", float(d[r2[i], c2[i]]),
                                      int(cnt[r2[i], c2[i]]), meta.get("reference_type")])
    conn.commit()
    return {"site": name, "site_id": site_id, "cells": int(len(rr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://sdb:sdb@localhost:5432/sdb")
    ap.add_argument("--products", default="products_v2")
    ap.add_argument("--sites", default="data/v2_sites")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    prod = Path(a.products)
    manifest = json.loads((prod / "manifest.json").read_text())
    with psycopg.connect(a.dsn) as conn:
        ensure_schema(conn)
        run_id = upsert_run(conn, manifest)
        for name in manifest["sites"]:
            print(ingest_site(conn, run_id, name, prod, Path(a.sites) / f"{name}.npz", a.stride))


if __name__ == "__main__":
    main()
