"""Full-scene inference -> Cloud-Optimised GeoTIFF depth & uncertainty products.

Layers (float32, metres unless stated, nodata = -9999):
  depth        ensemble mean depth (positive down)
  sigma        total predictive SD  sqrt(aleatoric + epistemic)
  aleatoric_sd / epistemic_sd
  lower68/upper68, lower90/upper90, lower95/upper95   conformal prediction intervals
  zmax         optical detection limit from the Maritorena model head
  confidence   uint8 class: 1=A high, 2=B medium, 3=C low, 4=D optically-deep/extrapolated, 0=nodata
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling

from .data import MAX_DEPTH, R, PreparedSite, all_water_index
from .model import HybridBathNetV2
from .train import ConformalCalibrator, predict

NODATA = -9999.0
CLASS_NAMES = {1: "A_high", 2: "B_medium", 3: "C_low", 4: "D_optically_deep_or_extrapolated"}
CLASS_COLORS = {0: (0, 0, 0, 0), 1: (26, 150, 65, 255), 2: (253, 174, 97, 255), 3: (215, 25, 28, 255), 4: (90, 90, 90, 255)}


def load_members(paths):
    ms = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        c = ck["cfg"]
        m = HybridBathNetV2(width=c["width"], n_blocks=c["blocks"], p_drop=c["dropout"],
                            use_physics_features=c["phys_feat"], use_film=c["film"], use_physics_head=c["phys_head"])
        m.load_state_dict(ck["state_dict"]); m.eval(); ms.append(m)
    return ms


def confidence_class(mu, half90, zmax):
    cls = np.full(mu.shape, 3, np.uint8)
    cls[half90 <= 2.0] = 2
    cls[half90 <= 1.0] = 1
    deep = (mu > MAX_DEPTH)
    if zmax is not None:
        deep |= mu > zmax
    cls[deep] = 4
    return cls


def nearshore_extent(ps: PreparedSite, km: float) -> np.ndarray:
    """Image-only product extent: water cells within `km` of non-water (land) cells."""
    from scipy.ndimage import distance_transform_edt, label
    w = ps.water[R:-R, R:-R] > 0
    lab, n = label(~w)
    sizes = np.bincount(lab.ravel())
    land = np.isin(lab, np.where(sizes >= 5000)[0][1:])  # non-water bodies >= 0.5 km^2 (ignores cloud holes)
    d = distance_transform_edt(~land) * abs(ps.site.transform.a)
    return w & (d <= km * 1000)


def scene_layers(models, ps: PreparedSite, cal: ConformalCalibrator, chunk=8192, extent=None):
    rr, cc = all_water_index(ps, exclude=None if extent is None else ~extent)
    o = predict(models, ps, rr, cc, chunk=chunk)
    H, W = ps.site.shape
    lay = {}

    def put(v, dtype=np.float32, fill=NODATA):
        a = np.full((H, W), fill, dtype)
        a[rr, cc] = v
        return a

    mu, sg = o["mu"], o["sigma"]
    lay["depth"] = put(mu)
    lay["sigma"] = put(sg)
    lay["aleatoric_sd"] = put(np.sqrt(o["alea"]))
    lay["epistemic_sd"] = put(np.sqrt(o["epi"]))
    for a, tag in ((0.32, "68"), (0.10, "90"), (0.05, "95")):
        lo, hi = cal.interval(mu, sg, a)
        lay[f"lower{tag}"] = put(lo)
        lay[f"upper{tag}"] = put(hi)
    zmax = o.get("zmax")
    if zmax is not None:
        lay["zmax"] = put(zmax)
    half90 = (lay["upper90"][rr, cc] - lay["lower90"][rr, cc]) / 2
    lay["confidence"] = put(confidence_class(mu, half90, zmax), np.uint8, 0)
    return lay


def write_cogs(layers: dict, ps: PreparedSite, out_dir: str | Path, tags: dict):
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    H, W = ps.site.shape
    written = {}
    for name, arr in layers.items():
        tmp = out_dir / f".{name}.tif"
        prof = dict(driver="GTiff", height=H, width=W, count=1, dtype=arr.dtype.name, crs=ps.site.crs,
                    transform=ps.site.transform, nodata=0 if name == "confidence" else NODATA)
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(arr, 1)
            dst.update_tags(**{k: (json.dumps(v) if not isinstance(v, str) else v) for k, v in tags.items()},
                            layer=name, units="class" if name == "confidence" else "m",
                            depth_convention="positive down, metres below reference water level")
            if name == "confidence":
                dst.write_colormap(1, CLASS_COLORS)
                dst.update_tags(classes=json.dumps(CLASS_NAMES))
        dst_path = out_dir / f"{ps.name}_{name}.tif"
        cog_translate(tmp, dst_path, cog_profiles.get("deflate"), quiet=True,
                      overview_resampling="nearest" if name == "confidence" else "average")
        tmp.unlink()
        written[name] = str(dst_path)
    # multi-band convenience stack
    names = [n for n in layers if n != "confidence"]
    stack = out_dir / f"{ps.name}_stack.tif"
    with rasterio.open(stack, "w", driver="GTiff", height=H, width=W, count=len(names), dtype="float32",
                       crs=ps.site.crs, transform=ps.site.transform, nodata=NODATA, compress="deflate",
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        for i, n in enumerate(names, 1):
            dst.write(layers[n].astype(np.float32), i)
            dst.set_band_description(i, n)
        dst.update_tags(**{k: (json.dumps(v) if not isinstance(v, str) else v) for k, v in tags.items()})
        dst.build_overviews([2, 4, 8], Resampling.average)
    written["stack"] = str(stack)
    return written
