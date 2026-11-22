"""HybridBathNet v2 FastAPI service.

Serves the GIS dashboard, XYZ map tiles rendered from Cloud-Optimised GeoTIFFs, and planner
queries (point, polygon, route profile, navigation screen) backed by PostGIS. If no database
is configured (HBN_DSN unset) the point/profile queries fall back to reading the COGs directly,
so the dashboard also works in a laptop "file mode".

Environment:
  HBN_PRODUCTS  directory with <site>_<layer>.tif COGs + manifest.json   (default products_v2)
  HBN_DSN       postgresql://user:pass@host:5432/db                       (optional)
Run:
  uvicorn sdb_platform.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

PRODUCTS = Path(os.environ.get("HBN_PRODUCTS", "products_v2"))
DSN = os.environ.get("HBN_DSN")
DASH = Path(__file__).resolve().parents[1] / "dashboard"
NODATA = -9999.0
LAYERS = ["depth", "sigma", "aleatoric_sd", "epistemic_sd", "lower90", "upper90", "lower95", "upper95",
          "zmax", "confidence"]
CONF = {1: "A high", 2: "B medium", 3: "C low", 4: "D optically deep / extrapolated"}

app = FastAPI(title="HybridBathNet v2 SDB service", version="2.0.0",
              description="Physics-guided, uncertainty-aware satellite-derived bathymetry")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def manifest() -> dict:
    p = PRODUCTS / "manifest.json"
    if not p.exists():
        raise HTTPException(503, f"no products found at {PRODUCTS}")
    return json.loads(p.read_text())


def _path(site: str, layer: str) -> Path:
    if site not in manifest()["sites"] or layer not in LAYERS:
        raise HTTPException(404, "unknown site/layer")
    p = PRODUCTS / f"{site}_{layer}.tif"
    if not p.exists():
        raise HTTPException(404, f"{p.name} missing")
    return p


@lru_cache(maxsize=64)
def _array(site: str, layer: str):
    with rasterio.open(_path(site, layer)) as src:
        return src.read(1), src.transform, src.crs.to_string(), src.nodata


def _sample(site: str, lon: float, lat: float) -> dict | None:
    _, tr, crs, _ = _array(site, "depth")
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    col, row = ~tr @ (x, y)
    r, c = int(math.floor(row)), int(math.floor(col))
    a = _array(site, "depth")[0]
    if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) or a[r, c] == NODATA:
        return None
    out = {}
    for l in LAYERS:
        try:
            v = _array(site, l)[0][r, c]
        except HTTPException:
            continue
        out[l] = int(v) if l == "confidence" else (None if v == NODATA else round(float(v), 3))
    out["confidence_label"] = CONF.get(out.get("confidence"), "n/a")
    out["row"], out["col"] = r, c
    return out


def _db():
    import psycopg
    return psycopg.connect(DSN)


# --------------------------------------------------------------------------- colour maps
def _ramp(stops):
    xs = np.array([s[0] for s in stops]); cs = np.array([s[1] for s in stops], float)
    def f(v):
        v = np.clip(v, xs[0], xs[-1])
        return np.stack([np.interp(v, xs, cs[:, i]) for i in range(3)], -1)
    return f

RAMPS = {
    "depth": (_ramp([(0, (222, 245, 229)), (2, (161, 218, 180)), (5, (65, 182, 196)), (10, (44, 127, 184)),
                     (15, (37, 52, 148)), (20, (8, 29, 88))]), 0, 20),
    "sigma": (_ramp([(0, (255, 255, 204)), (0.75, (254, 217, 118)), (1.5, (253, 141, 60)), (3, (227, 26, 28)),
                     (5, (128, 0, 38))]), 0, 5),
    "width": (_ramp([(0, (247, 252, 245)), (2, (161, 217, 155)), (4, (65, 171, 93)), (8, (0, 68, 27))]), 0, 8),
}
CONF_RGBA = {1: (26, 150, 65, 200), 2: (253, 174, 97, 200), 3: (215, 25, 28, 200), 4: (90, 90, 90, 170)}


def _tile_bounds(z, x, y):
    n = 2 ** z
    def merc(xt, yt):
        lon = xt / n * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))
        return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)
    x0, y1 = merc(x, y); x1, y0 = merc(x + 1, y + 1)
    return x0, y0, x1, y1


def _render(site, layer, z, x, y, draft=None):
    from PIL import Image
    x0, y0, x1, y1 = _tile_bounds(z, x, y)
    dst_t = rasterio.transform.from_bounds(x0, y0, x1, y1, 256, 256)
    src_layer = {"width90": "upper90", "hazard": "lower95"}.get(layer, layer)
    arr, tr, crs, nod = _array(site, src_layer)
    out = np.full((256, 256), nod, arr.dtype)
    reproject(arr, out, src_transform=tr, src_crs=crs, dst_transform=dst_t, dst_crs="EPSG:3857",
              src_nodata=nod, dst_nodata=nod,
              resampling=Resampling.nearest if layer in ("confidence", "hazard") else Resampling.bilinear)
    rgba = np.zeros((256, 256, 4), np.uint8)
    valid = out != nod
    if layer == "confidence":
        for k, c in CONF_RGBA.items():
            rgba[out == k] = c
    elif layer == "hazard":
        d = float(draft or 2.0)
        conf = np.full((256, 256), 0, np.uint8)
        reproject(_array(site, "confidence")[0], conf, src_transform=tr, src_crs=crs, dst_transform=dst_t,
                  dst_crs="EPSG:3857", src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
        mu = np.full((256, 256), NODATA, np.float32)
        reproject(_array(site, "depth")[0], mu, src_transform=tr, src_crs=crs, dst_transform=dst_t,
                  dst_crs="EPSG:3857", src_nodata=NODATA, dst_nodata=NODATA, resampling=Resampling.nearest)
        rgba[valid & (out < d + 0.5)] = (215, 25, 28, 190)                         # hazard
        rgba[valid & (out >= d + 0.5) & (mu < d + 2.5)] = (253, 174, 97, 170)     # caution
        rgba[valid & (out >= d + 0.5) & (mu >= d + 2.5)] = (26, 150, 65, 120)     # clear
        rgba[conf == 4] = (90, 90, 90, 150)                                       # unknown
    else:
        if layer == "width90":
            lo = np.full((256, 256), nod, np.float32)
            reproject(_array(site, "lower90")[0], lo, src_transform=tr, src_crs=crs, dst_transform=dst_t,
                      dst_crs="EPSG:3857", src_nodata=nod, dst_nodata=nod, resampling=Resampling.bilinear)
            vals = out - lo; valid &= lo != nod; ramp = RAMPS["width"]
        else:
            vals = out; ramp = RAMPS["sigma" if layer in ("sigma", "aleatoric_sd", "epistemic_sd") else "depth"]
        rgba[..., :3] = ramp[0](vals).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 230, 0)
    buf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (DASH / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    ok_db = None
    if DSN:
        try:
            with _db() as c:
                c.execute("select 1"); ok_db = True
        except Exception as e:  # pragma: no cover
            ok_db = f"error: {e.__class__.__name__}"
    return {"status": "ok", "products": str(PRODUCTS), "database": ok_db}


@app.get("/api/sites")
def sites():
    m = manifest(); out = []
    for s in m["sites"]:
        with rasterio.open(_path(s, "depth")) as src:
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        out.append({"name": s, "bounds": b, "meta": m.get("site_meta", {}).get(s, {}),
                    "metrics": m.get("metrics", {}).get(s, {}), "layers": [l for l in LAYERS if (PRODUCTS / f"{s}_{l}.tif").exists()]})
    return {"model": {k: m.get(k) for k in ("version", "n_members", "calibration")}, "sites": out}


@app.get("/api/sites/{site}/stats")
def site_stats(site: str):
    a = _array(site, "depth")[0]; c = _array(site, "confidence")[0]; s = _array(site, "sigma")[0]
    v = a != NODATA
    return {"site": site, "water_cells": int(v.sum()), "area_km2": float(v.sum() * 1e-4),
            "depth": {"mean": float(a[v].mean()), "p5": float(np.percentile(a[v], 5)), "p95": float(np.percentile(a[v], 95))},
            "sigma_mean": float(s[v].mean()),
            "confidence_share_pct": {CONF[k]: float((c[v] == k).mean() * 100) for k in CONF},
            "depth_histogram": np.histogram(a[v], bins=np.arange(0, 32, 2))[0].tolist()}


@app.get("/tiles/{site}/{layer}/{z}/{x}/{y}.png")
def tile(site: str, layer: str, z: int, x: int, y: int, draft: float | None = None):
    if layer not in LAYERS + ["width90", "hazard"]:
        raise HTTPException(404)
    return Response(_render(site, layer, z, x, y, draft), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/query/point")
def query_point(lon: float, lat: float, site: str | None = None):
    for s in ([site] if site else manifest()["sites"]):
        r = _sample(s, lon, lat)
        if r:
            return {"site": s, "lon": lon, "lat": lat, **r}
    return JSONResponse({"detail": "no prediction at this location"}, 404)


class Geo(BaseModel):
    site: str
    geometry: dict                 # GeoJSON Polygon or LineString (EPSG:4326)
    draft_m: float | None = None
    ukc_m: float = 0.5
    n_samples: int = 200


@app.post("/api/query/polygon")
def query_polygon(q: Geo):
    """Zonal statistics + navigation screen inside a polygon."""
    from rasterio.features import geometry_mask
    from shapely.geometry import shape
    from shapely.ops import transform as stransform

    a, tr, crs, _ = _array(q.site, "depth")
    g = stransform(Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform, shape(q.geometry))
    inside = ~geometry_mask([g.__geo_interface__], a.shape, tr) & (a != NODATA)
    n = int(inside.sum())
    if n == 0:
        return {"cells": 0}
    lo95 = _array(q.site, "lower95")[0][inside]; conf = _array(q.site, "confidence")[0][inside]
    sig = _array(q.site, "sigma")[0][inside]; d = a[inside]
    res = {"site": q.site, "cells": n, "area_km2": n * 1e-4, "depth_mean": float(d.mean()),
           "depth_min": float(d.min()), "depth_max": float(d.max()), "sigma_mean": float(sig.mean()),
           "shoalest_lower95": float(lo95.min()),
           "confidence_share_pct": {CONF[k]: float((conf == k).mean() * 100) for k in CONF}}
    if q.draft_m is not None:
        lim = q.draft_m + q.ukc_m
        res["navigation"] = {"draft_m": q.draft_m, "ukc_m": q.ukc_m,
                             "hazard_pct": float(((lo95 < lim) & (conf != 4)).mean() * 100),
                             "caution_pct": float(((lo95 >= lim) & (d < lim + 2) & (conf != 4)).mean() * 100),
                             "unknown_pct": float((conf == 4).mean() * 100)}
    if DSN:
        with _db() as c:
            c.execute("INSERT INTO sdb.planner_aoi (name, geom, draft_m, summary) VALUES (%s, ST_SetSRID(ST_GeomFromGeoJSON(%s),4326), %s, %s)",
                      (f"{q.site} query", json.dumps(q.geometry), q.draft_m, json.dumps(res)))
    return res


@app.post("/api/query/profile")
def query_profile(q: Geo):
    """Depth profile with 90/95 % conformal bands along a route (LineString)."""
    from shapely.geometry import shape
    from shapely.ops import transform as stransform

    _, tr, crs, _ = _array(q.site, "depth")
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    line = stransform(fwd.transform, shape(q.geometry))
    L = line.length
    pts = []
    for i in range(q.n_samples):
        p = line.interpolate(L * i / max(1, q.n_samples - 1))
        col, row = ~tr @ (p.x, p.y)
        r, c = int(row), int(col)
        rec = {"distance_m": round(L * i / max(1, q.n_samples - 1), 1)}
        a = _array(q.site, "depth")[0]
        if 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and a[r, c] != NODATA:
            for l in ("depth", "lower90", "upper90", "lower95", "upper95", "sigma", "confidence"):
                v = _array(q.site, l)[0][r, c]; rec[l] = int(v) if l == "confidence" else float(v)
        pts.append(rec)
    ok = [p for p in pts if "depth" in p]
    summ = {"length_m": L, "samples": len(pts), "covered": len(ok)}
    if ok and q.draft_m is not None:
        lim = q.draft_m + q.ukc_m
        summ["min_lower95"] = min(p["lower95"] for p in ok)
        summ["hazard_segments_m"] = sum(L / max(1, q.n_samples - 1) for p in ok if p["lower95"] < lim and p["confidence"] != 4)
        summ["route_status"] = "HAZARD" if summ["hazard_segments_m"] > 0 else "CLEAR (95 % conformal bound)"
    return {"summary": summ, "profile": pts}


@app.get("/api/navigation/{site}")
def navigation(site: str, draft: float = Query(2.0), ukc: float = Query(0.5)):
    if DSN:
        with _db() as c:
            rows = c.execute("SELECT * FROM sdb.navigation_screen(%s::text, %s::float8, %s::float8)", (site, draft, ukc)).fetchall()
        return {"site": site, "draft": draft, "ukc": ukc, "source": "postgis",
                "classes": {r[0]: {"cells": r[1], "area_km2": r[2]} for r in rows}}
    lo = _array(site, "lower95")[0]; d = _array(site, "depth")[0]; c = _array(site, "confidence")[0]
    v = d != NODATA; lim = draft + ukc
    cls = {"unknown_optically_deep": v & (c == 4), "hazard": v & (c != 4) & (lo < lim),
           "caution": v & (c != 4) & (lo >= lim) & (d < lim + 2), "clear": v & (c != 4) & (lo >= lim) & (d >= lim + 2)}
    return {"site": site, "draft": draft, "ukc": ukc, "source": "cog",
            "classes": {k: {"cells": int(m.sum()), "area_km2": float(m.sum() * 1e-4)} for k, m in cls.items()}}


@app.get("/api/sites/{site}/download/{layer}")
def download(site: str, layer: str):
    p = _path(site, layer)
    return FileResponse(p, media_type="image/tiff", filename=p.name)


@app.get("/api/metrics")
def metrics():
    p = PRODUCTS / "study_metrics.json"
    return json.loads(p.read_text()) if p.exists() else manifest().get("metrics", {})
