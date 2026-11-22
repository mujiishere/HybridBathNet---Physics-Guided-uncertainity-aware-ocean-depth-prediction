"""Site adapters: harmonise heterogeneous Sentinel-2 + reference-depth sources
into one common representation (``SiteData``).

Every site is converted to
  * ``refl``   float32 (4, H, W) surface reflectance [0-1] in band order
               Blue (B2, ~492 nm), Green (B3, ~560 nm), Red (B4, ~665 nm), NIR (B8, ~833 nm)
  * ``depth``  float32 (H, W) positive-down reference depth in metres, NaN where unlabelled
  * ``count``  int16  (H, W) number of reference observations aggregated into the pixel
  * ``water``  bool   (H, W) optical water mask (derived from imagery only, never labels)
  * ``transform`` / ``crs`` of a projected 10 m grid.

No information from reference depths is used to build inputs or masks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

BANDS = ("blue", "green", "red", "nir")


@dataclass
class SiteData:
    name: str
    refl: np.ndarray
    depth: np.ndarray
    count: np.ndarray
    water: np.ndarray
    transform: rasterio.Affine
    crs: str
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.depth.shape

    def summary(self) -> dict:
        lab = np.isfinite(self.depth) & self.water
        d = self.depth[lab]
        return {
            "site": self.name,
            "grid": list(self.shape),
            "crs": self.crs,
            "water_pixels": int(self.water.sum()),
            "labelled_water_pixels": int(lab.sum()),
            "depth_min": float(np.nanmin(d)) if d.size else None,
            "depth_median": float(np.nanmedian(d)) if d.size else None,
            "depth_max": float(np.nanmax(d)) if d.size else None,
            **{k: v for k, v in self.meta.items() if isinstance(v, (str, int, float))},
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, refl=self.refl, depth=self.depth, count=self.count, water=self.water,
            transform=np.array(self.transform)[:6], crs=self.crs, name=self.name,
            meta=json.dumps(self.meta),
        )

    @staticmethod
    def load(path: str | Path) -> "SiteData":
        z = np.load(path, allow_pickle=False)
        t = z["transform"]
        return SiteData(
            name=str(z["name"]), refl=z["refl"], depth=z["depth"], count=z["count"],
            water=z["water"], transform=rasterio.Affine(*t), crs=str(z["crs"]),
            meta=json.loads(str(z["meta"])),
        )


# ----------------------------------------------------------------------------
# Image-only water mask
# ----------------------------------------------------------------------------
def water_mask(refl: np.ndarray, ndwi_thr: float = 0.0, nir_max: float = 0.08) -> np.ndarray:
    """McFeeters NDWI water mask with a NIR brightness guard (cloud / land / glint).
    Uses imagery only; reference depths are never consulted."""
    g, n = refl[1], refl[3]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (g - n) / (g + n)
    ok = np.all(np.isfinite(refl), axis=0) & np.all(refl > 0, axis=0)
    return ok & (ndwi > ndwi_thr) & (n < nir_max)


def _rasterize_points(xs, ys, zs, transform, shape, agg="median"):
    """Aggregate point depths into grid cells (median), returning depth & count."""
    inv = ~transform
    cols, rows = inv @ (np.asarray(xs), np.asarray(ys))
    rows = np.floor(rows).astype(int)
    cols = np.floor(cols).astype(int)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1]) & np.isfinite(zs)
    rows, cols, zs = rows[ok], cols[ok], np.asarray(zs)[ok]
    depth = np.full(shape, np.nan, np.float32)
    count = np.zeros(shape, np.int16)
    key = rows * shape[1] + cols
    order = np.argsort(key)
    key, zs = key[order], zs[order]
    uniq, start = np.unique(key, return_index=True)
    ends = np.r_[start[1:], key.size]
    for k, s, e in zip(uniq, start, ends):
        r, c = divmod(int(k), shape[1])
        depth[r, c] = np.median(zs[s:e]) if agg == "median" else np.mean(zs[s:e])
        count[r, c] = e - s
    return depth, count


# ----------------------------------------------------------------------------
# Site 1: Ponce, Puerto Rico — Sentinel-2 L2A + NOAA topobathy lidar (10 m)
# Source: Lumban-Gaol et al. (2022) Marine Geodesy, github.com/yustisiardhitasari/sdbcnn
# ----------------------------------------------------------------------------
def load_ponce(root: str | Path) -> SiteData:
    root = Path(root)
    with rasterio.open(root / "s2.tif") as r:
        a = r.read().astype(np.float32)
        nod = r.nodata
        transform, crs = r.transform, r.crs.to_string()
    a[a == nod] = np.nan
    a[a < -1e30] = np.nan
    # band order R,G,B,NIR,SWIR1,SWIR2 ("RGBNSS"); L2A DN scaled by 1e4 (2019 product: no offset)
    refl = np.stack([a[2], a[1], a[0], a[3]]) / 1e4
    with rasterio.open(root / "depth.tif") as r:
        z = r.read(1).astype(np.float32)
    depth = np.where(np.isfinite(z) & (z < 0), -z, np.nan).astype(np.float32)
    count = np.isfinite(depth).astype(np.int16)
    return SiteData(
        "ponce_pr", refl, depth, count, water_mask(refl), transform, crs,
        meta={"region": "Ponce, Puerto Rico (Caribbean)", "reference": "NOAA topobathymetric lidar, tide-corrected, 10 m",
              "reference_type": "lidar", "image_date": "2019-01-08", "atm_correction": "Sen2Cor L2A",
              "water_type": "clear oceanic / coral-seagrass"},
    )


# ----------------------------------------------------------------------------
# Site 2: Kepulauan Seribu (Thousand Islands), Java Sea, Indonesia — S2 + tide-corrected
# single-beam echo-sounding points. Source: github.com/rifqiharrys/sdb_gui (sample data)
# ----------------------------------------------------------------------------
def load_seribu(root: str | Path) -> SiteData:
    import geopandas as gpd

    root = Path(root)
    with rasterio.open(root / "image.tif") as r:
        a = r.read().astype(np.float32)
        nod = r.nodata
        transform, crs = r.transform, r.crs.to_string()
    a[a == nod] = np.nan
    refl = a[:4] / 1e4  # B2,B3,B4,B8 (blue brightest over water)
    pts = gpd.read_file(root / "depth_sample.shp")
    depth, count = _rasterize_points(pts.X.values, pts.Y.values, pts.Z_Koreksi.values.astype(float),
                                     transform, refl.shape[1:])
    return SiteData(
        "seribu_id", refl, depth, count, water_mask(refl), transform, crs,
        meta={"region": "Kepulauan Seribu, Java Sea, Indonesia", "reference": "single-beam echo sounding, tide-corrected (Z_Koreksi)",
              "reference_type": "sonar", "atm_correction": "L2A", "water_type": "reef lagoon / moderately turbid",
              "n_points": int(len(pts))},
    )


# ----------------------------------------------------------------------------
# Site 3: Langebaan / Saldanha, South Africa — ACOLITE S2 rhos + ICESat-2 ATL24 bathymetry photons
# Source: Payandeh et al. (2026) Front. Remote Sens.; github.com/nasa-jpl/Satellite-Derived-Bathymetry
# ----------------------------------------------------------------------------
def load_langebaan(root: str | Path, res: float = 10.0, min_conf: float = 0.6) -> SiteData:
    import pandas as pd
    import xarray as xr
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    root = Path(root)
    ds = xr.open_dataset(root / "s2_l2r.nc")
    lon, lat = ds.lon.values.astype(np.float64), ds.lat.values.astype(np.float64)
    bands = [ds[k].values.astype(np.float32) for k in ("rhos_492", "rhos_559", "rhos_665", "rhos_833")]
    crs = "EPSG:32734"
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    X, Y = tr.transform(lon, lat)
    ok = np.isfinite(X) & np.isfinite(Y)
    x0, x1 = np.floor(X[ok].min() / res) * res, np.ceil(X[ok].max() / res) * res
    y0, y1 = np.floor(Y[ok].min() / res) * res, np.ceil(Y[ok].max() / res) * res
    W, H = int((x1 - x0) / res), int((y1 - y0) / res)
    transform = from_origin(x0, y1, res, res)
    gx = x0 + (np.arange(W) + 0.5) * res
    gy = y1 - (np.arange(H) + 0.5) * res
    GX, GY = np.meshgrid(gx, gy)
    tree = cKDTree(np.c_[X[ok], Y[ok]])
    dist, idx = tree.query(np.c_[GX.ravel(), GY.ravel()], k=1)
    refl = np.full((4, H * W), np.nan, np.float32)
    good = dist < res * 0.75
    for i, b in enumerate(bands):
        refl[i, good] = b[ok][idx[good]]
    refl = refl.reshape(4, H, W)

    a = pd.read_csv(root / "atl24.csv")
    a = a[(a.class_ph == 40) & (a.confidence >= min_conf)]
    ax, ay = tr.transform(a.lon.values, a.lat.values)
    # ATL24 ortho_h is seabed height w.r.t. EGM2008 geoid; depth ~ -ortho_h relative to mean sea level.
    depth, count = _rasterize_points(ax, ay, -a.ortho_h.values, transform, (H, W))
    depth[depth < 0] = np.nan
    return SiteData(
        "langebaan_za", refl, depth, count, water_mask(refl), transform, crs,
        meta={"region": "Langebaan Lagoon / Saldanha Bay, South Africa", "reference": "ICESat-2 ATL24 seabed photons (conf>=0.6), depth=-ortho_h",
              "reference_type": "icesat2_atl24", "image_date": "2024-11-28", "atm_correction": "ACOLITE rhos",
              "water_type": "temperate lagoon, sand/seagrass", "n_points": int(len(a))},
    )


# ----------------------------------------------------------------------------
# Site 4 (user-local): Kerala, India (HybridBathNet v1 AOI) — S2 median composite + GEBCO / ATL24
# Runs on the user's workstation where the original stack.tif is stored.
# ----------------------------------------------------------------------------
def load_kerala(stack_tif: str | Path, atl24_csv: str | Path | None = None, use_gebco: bool = False) -> SiteData:
    """Adapter for runs/2026-11-04/processed/stack.tif produced by src/preprocess.py
    (bands: B02,B03,B04,B08 medians, 4 std, gebco_depth, valid_mask).

    Reference: by default ICESat-2 ATL24 points if ``atl24_csv`` (columns lon,lat,depth or
    lon,lat,ortho_h) is given; GEBCO only if ``use_gebco`` (coarse, not independent survey)."""
    import pandas as pd
    from pyproj import Transformer

    with rasterio.open(stack_tif) as r:
        a = r.read().astype(np.float32)
        transform, crs = r.transform, r.crs.to_string()
    dn = a[:4]
    # Processing baseline >= 04.00 (2022+) carries BOA_ADD_OFFSET=-1000
    offset = 1000.0 if np.nanmedian(dn[0][dn[0] > 0]) > 1000 else 0.0
    refl = np.clip((dn - offset) / 1e4, 1e-5, None)
    refl[:, ~np.isfinite(dn).all(0)] = np.nan
    H, W = refl.shape[1:]
    depth = np.full((H, W), np.nan, np.float32)
    count = np.zeros((H, W), np.int16)
    ref = "none"
    if atl24_csv:
        p = pd.read_csv(atl24_csv)
        if "depth_m" in p:          # HybridBathNet v1 recovery_v2 official_profiles.csv (ATL24 V002, surface-referenced)
            p = p[(p.depth_m > 0) & (p.depth_m <= 25)]
            z = p["depth_m"].values
        else:
            z = p["depth"].values if "depth" in p else -p["ortho_h"].values
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(p.lon.values, p.lat.values)
        depth, count = _rasterize_points(x, y, z, transform, (H, W))
        ref = "icesat2_atl24"
    elif use_gebco:
        g = a[8]
        depth = np.where(g < 0, -g, np.nan).astype(np.float32)
        count = np.isfinite(depth).astype(np.int16)
        ref = "gebco_2020_resampled"
    return SiteData("kerala_in", refl, depth, count, water_mask(refl), transform, crs,
                    meta={"region": "Kerala coast, India (Arabian Sea)", "reference_type": ref,
                          "atm_correction": "Sen2Cor L2A median composite", "water_type": "turbid monsoon coast"})


LOADERS = {"ponce_pr": load_ponce, "seribu_id": load_seribu, "langebaan_za": load_langebaan}


def build_kerala(stack_tif, atl24_csv, out_dir="data/v2_sites"):
    """CLI helper: python -c "from hbn2.sites import build_kerala; build_kerala(r'<stack.tif>', r'<official_profiles.csv>')" """
    s = load_kerala(stack_tif, atl24_csv)
    s.save(Path(out_dir) / "kerala_in.npz")
    return s.summary()
DEFAULT_ROOTS = {"ponce_pr": "ponce", "seribu_id": "sdbgui", "langebaan_za": "jpl"}


def build_all(raw_root: str | Path, out_dir: str | Path) -> list[dict]:
    out = []
    for name, fn in LOADERS.items():
        s = fn(Path(raw_root) / DEFAULT_ROOTS[name])
        s.save(Path(out_dir) / f"{name}.npz")
        out.append(s.summary())
    return out

