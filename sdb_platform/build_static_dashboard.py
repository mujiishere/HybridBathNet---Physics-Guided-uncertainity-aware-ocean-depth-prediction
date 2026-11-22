"""Bundle the dashboard + products into ONE self-contained HTML file (no server needed).

Overlays are pre-rendered PNGs warped to EPSG:4326; point/route queries use an embedded
down-sampled grid. Area statistics need the full API.

python sdb_platform/build_static_dashboard.py --products products_v2 --out dashboard_static.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdb_platform.api.main import CONF_RGBA, NODATA, RAMPS  # noqa: E402

FIELDS = ["depth", "sigma", "lower90", "upper90", "lower95", "upper95", "zmax", "confidence", "aleatoric_sd", "epistemic_sd"]


def warp(path, max_px=900, nearest=False):
    with rasterio.open(path) as src:
        t, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        s = min(1.0, max_px / max(w, h))
        w2, h2 = int(w * s), int(h * s)
        t2 = t * rasterio.Affine.scale(w / w2, h / h2)
        out = np.full((h2, w2), src.nodata, src.dtypes[0])
        reproject(rasterio.band(src, 1), out, dst_transform=t2, dst_crs="EPSG:4326", src_nodata=src.nodata,
                  dst_nodata=src.nodata, resampling=Resampling.nearest if nearest else Resampling.bilinear)
        b = (t2.c, t2.f + t2.e * h2, t2.c + t2.a * w2, t2.f)
        return out, src.nodata, b


def png(rgba):
    buf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ramp_img(a, nod, ramp):
    rgba = np.zeros(a.shape + (4,), np.uint8)
    rgba[..., :3] = ramp[0](a).astype(np.uint8); rgba[..., 3] = np.where(a != nod, 230, 0)
    return rgba


def site_bundle(prod: Path, s: str, draft_lim=2.5, target_cells=22000):
    imgs, grids = {}, {}
    for f in FIELDS:
        p = prod / f"{s}_{f}.tif"
        if p.exists():
            grids[f] = warp(p, nearest=True)
    d, nod, b = grids["depth"]
    imgs["depth"] = png(ramp_img(d, nod, RAMPS["depth"]))
    imgs["sigma"] = png(ramp_img(grids["sigma"][0], nod, RAMPS["sigma"]))
    lo, hi = grids["lower90"][0], grids["upper90"][0]
    imgs["width90"] = png(ramp_img(np.where(d != nod, hi - lo, nod), nod, RAMPS["width"]))
    c = grids["confidence"][0]
    rgba = np.zeros(c.shape + (4,), np.uint8)
    for k, col in CONF_RGBA.items():
        rgba[c == k] = col
    imgs["confidence"] = png(rgba)
    lo95 = grids["lower95"][0]
    hz = np.zeros(c.shape + (4,), np.uint8); v = d != nod
    hz[v & (lo95 < draft_lim)] = (215, 25, 28, 190)
    hz[v & (lo95 >= draft_lim) & (d < draft_lim + 2)] = (253, 174, 97, 170)
    hz[v & (lo95 >= draft_lim) & (d >= draft_lim + 2)] = (26, 150, 65, 120)
    hz[c == 4] = (90, 90, 90, 150)
    imgs["hazard"] = png(hz)
    # query grid
    H, W = d.shape
    st = max(1, int(np.ceil(np.sqrt(H * W / target_cells))))
    g = {"nx": len(range(0, W, st)), "ny": len(range(0, H, st)), "fields": [f for f in FIELDS if f in grids]}
    for f in g["fields"]:
        a, n_, _ = grids[f]
        a = a[::st, ::st]
        g[f] = [None if (x == n_ or dd == nod) else (int(x) if f == "confidence" else round(float(x), 2))
                for x, dd in zip(a.ravel(), d[::st, ::st].ravel())]
    return {"bounds": list(b), "images": imgs, "grid": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="products_v2"); ap.add_argument("--out", default="dashboard_static.html")
    a = ap.parse_args()
    prod = Path(a.products)
    man = json.loads((prod / "manifest.json").read_text())
    bundle = {"sites": {}, "index": {"sites": []}}
    for s in man["sites"]:
        sb = site_bundle(prod, s)
        bundle["sites"][s] = sb
        bundle["index"]["sites"].append({"name": s, "bounds": sb["bounds"], "meta": man["site_meta"].get(s, {}),
                                         "metrics": man["metrics"].get(s, {})})
    html = (Path(__file__).parent / "dashboard" / "index.html").read_text(encoding="utf-8")
    inject = "<script>window.HBN_STATIC=" + json.dumps(bundle, separators=(",", ":")) + ";</script>\n"
    html = html.replace("<script>\nconst STATIC", inject + "<script>\nconst STATIC", 1)
    # static/offline build: no remote basemap tiles (blocked in sandboxed viewers); plain sea-chart ground instead
    html = html.replace("layers:[{id:'img',type:'raster',source:'img',paint:{'raster-saturation':-0.35}}]",
                        "layers:[{id:'bg',type:'background',paint:{'background-color':'#E6D8A6'}}]")
    css = (Path(__file__).parent / "dashboard" / "vendor" / "maplibre-gl-4.7.1.css").read_text(encoding="utf-8")
    html = html.replace('<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">', "<style>" + css + "</style>")
    html = html.replace("Draft check</button>", "Draft check (2 m + 0.5 m)</button>")
    Path(a.out).write_text(html, encoding="utf-8")
    print(a.out, round(Path(a.out).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
