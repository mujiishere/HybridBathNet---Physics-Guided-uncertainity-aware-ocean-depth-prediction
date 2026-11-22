"""Build the Kerala (v1 AOI) site from raw Sentinel-2 L2A band files + official ATL24 V002 profiles.

Median composite of SCL-screened Jan 2024 L2A observations (baseline N0510: BOA_ADD_OFFSET = -1000),
4 bands at 10 m on an EPSG:32643 grid covering lon 76.30-76.55, lat 8.90-9.20 (the v1 AOI).
Reference: recovered official ATL24 V002 bathymetry (runs/2026-11-06/recovery_v2/official_profiles.csv),
depth_m relative to the ATL24 water surface, per-pixel median over passes (2018-2024).

python -m hbn2.kerala_build --bands <dir with YYYYMMDD_TILE_Bxx.jp2> --atl24 <official_profiles.csv> --out data/v2_sites/kerala_in.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from .sites import SiteData, _rasterize_points, water_mask

BAD_SCL = {0, 1, 2, 3, 8, 9, 10, 11}  # nodata, saturated, dark/shadow, cloud shadow, clouds, cirrus, snow


def grid(bbox=(76.30, 8.90, 76.55, 9.20), crs="EPSG:32643", res=10.0):
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    x0, x1 = np.floor(min(xs) / res) * res, np.ceil(max(xs) / res) * res
    y0, y1 = np.floor(min(ys) / res) * res, np.ceil(max(ys) / res) * res
    return from_origin(x0, y1, res, res), (int((y1 - y0) / res), int((x1 - x0) / res)), crs


def build(band_dir, atl24_csv, out, gebco_tif=None):
    tr, shape, crs = grid()
    obs = defaultdict(dict)
    for f in glob.glob(os.path.join(band_dir, "*.jp2")):
        m = re.match(r"(\d{8})_(T\w+)_(B02|B03|B04|B08|SCL)\.jp2", os.path.basename(f))
        if m:
            obs[(m.group(1), m.group(2))][m.group(3)] = f
    stacks = []
    for key, d in sorted(obs.items()):
        if len(d) < 5:
            continue
        arr = []
        for b in ("B02", "B03", "B04", "B08", "SCL"):
            dst = np.zeros(shape, np.float32)
            with rasterio.open(d[b]) as src:
                reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs, dst_transform=tr,
                          dst_crs=crs, src_nodata=0, dst_nodata=0,
                          resampling=Resampling.nearest if b == "SCL" else Resampling.bilinear)
            arr.append(dst)
        a = np.stack(arr)
        refl = (a[:4] - 1000.0) / 1e4
        bad = np.isin(a[4].astype(int), list(BAD_SCL)) | (a[:4] == 0).any(0)
        refl[:, bad] = np.nan
        print(key, "clear fraction", round(1 - bad.mean(), 3))
        stacks.append(refl)
    cube = np.stack(stacks)
    with np.errstate(all="ignore"):
        comp = np.nanmedian(cube, 0).astype(np.float32)
    nobs = np.isfinite(cube[:, 1]).sum(0)
    p = pd.read_csv(atl24_csv)
    p = p[(p.depth_m > 0) & (p.depth_m <= 25)]
    n0 = len(p)
    if gebco_tif:
        # QC (applied to every split before any training): drop ATL24 "seabed" photons shallower than 3 m
        # where the GEBCO grid is deeper than 10 m -- these are near-surface false bottoms offshore.
        with rasterio.open(gebco_tif) as g:
            gx, gy = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True).transform(p.lon.values, p.lat.values)
            gd = -np.array([v[0] for v in g.sample(zip(gx, gy))])
        p = p[~((p.depth_m.values < 3) & (gd > 10))]
    print("ATL24 QC kept", len(p), "of", n0)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(p.lon.values, p.lat.values)
    depth, count = _rasterize_points(x, y, p.depth_m.values, tr, shape)
    s = SiteData("kerala_in", comp, depth, count, water_mask(np.nan_to_num(comp)) & (nobs > 0), tr, crs,
                 meta={"region": "Kerala coast (Alappuzha), India - Arabian Sea", "reference": "ICESat-2 ATL24 V002 official profiles 2018-2024, per-pixel median",
                       "reference_type": "icesat2_atl24", "image_date": "2024-01-13/2024-01-28 median", "atm_correction": "Sen2Cor L2A median of 5 scene-tiles",
                       "water_type": "turbid monsoon-influenced shelf", "n_points": int(len(p)), "qc": "dropped ATL24 depth<3 m where GEBCO>10 m (offshore near-surface false bottoms)"})
    s.save(out)
    return s.summary()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", required=True); ap.add_argument("--atl24", required=True)
    ap.add_argument("--out", default="data/v2_sites/kerala_in.npz"); ap.add_argument("--gebco", default=None)
    a = ap.parse_args()
    import json
    print(json.dumps(build(a.bands, a.atl24, a.out, a.gebco), indent=1))
