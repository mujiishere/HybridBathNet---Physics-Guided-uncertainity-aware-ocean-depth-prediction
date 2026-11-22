"""Patch sampling and leakage-safe spatial splits for HybridBathNet v2."""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from .physics import hedley_deglint, scene_stats
from .sites import SiteData

PATCH = 13
R = PATCH // 2
MAX_DEPTH = 20.0  # study domain (m); labels deeper than this are excluded from training & scoring


@dataclass
class PreparedSite:
    site: SiteData
    refl: np.ndarray       # padded deglinted reflectance (4, H+2R, W+2R)
    water: np.ndarray      # padded water mask
    rinf: np.ndarray       # (4,)
    desc: np.ndarray       # (16,)
    glint: dict
    blocks: np.ndarray     # (H,W) block id
    labelled: np.ndarray   # (H,W) bool labelled & water & within domain

    @property
    def name(self):
        return self.site.name


def prepare(site: SiteData, block_m: float = 200.0) -> PreparedSite:
    refl = np.nan_to_num(site.refl, nan=0.0)
    w = site.water.copy()
    dg, info = hedley_deglint(refl, w)
    st = scene_stats(dg, w)
    pad = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(R, R), (R, R)], mode="constant")
    H, W = site.shape
    res = abs(site.transform.a)
    bs = max(4, int(round(block_m / res)))
    rows, cols = np.mgrid[0:H, 0:W]
    blocks = (rows // bs) * (W // bs + 1) + cols // bs
    lab = np.isfinite(site.depth) & w & (site.depth <= MAX_DEPTH) & (site.depth > 0)
    return PreparedSite(site, pad(dg).astype(np.float32), pad(w.astype(np.float32)),
                        np.array(st["rinf"], np.float32), np.array(st["desc"], np.float32), info, blocks, lab)


def assign_folds(ps: PreparedSite, k: int = 5, seed: int = 0) -> np.ndarray:
    """Random assignment of labelled spatial blocks to k folds (balanced by label count)."""
    rng = np.random.default_rng(seed + zlib.crc32(ps.name.encode()) % 1000)
    b = ps.blocks[ps.labelled]
    ids, cnt = np.unique(b, return_counts=True)
    order = rng.permutation(len(ids))
    load = np.zeros(k)
    fold_of = {}
    for i in order[np.argsort(-cnt[order], kind="stable")]:
        f = int(np.argmin(load + rng.random(k) * 1e-3))
        fold_of[ids[i]] = f
        load[f] += cnt[i]
    folds = np.full(ps.labelled.shape, -1, np.int8)
    for bid, f in fold_of.items():
        folds[(ps.blocks == bid)] = f
    return folds


def split_masks(ps: PreparedSite, folds: np.ndarray, test_fold: int, k: int = 5, buffer_px: int = R):
    """Return boolean masks (train, val, cal, test) over labelled pixels. The fold after the
    test fold is split block-wise into validation (early stopping) and calibration (conformal).
    Training pixels within ``buffer_px`` of any held-out pixel are removed (patch footprint)."""
    lab = ps.labelled
    test = lab & (folds == test_fold)
    hold = (folds == (test_fold + 1) % k)
    hb = np.unique(ps.blocks[hold & lab])
    rng = np.random.default_rng(test_fold)
    vb = set(rng.permutation(hb)[: len(hb) // 2].tolist())
    isval = np.isin(ps.blocks, list(vb))
    val = lab & hold & isval
    cal = lab & hold & ~isval
    held = (folds == test_fold) | hold
    near = binary_dilation(held, structure=np.ones((3, 3), bool), iterations=buffer_px) if buffer_px else held
    train = lab & ~near & (folds >= 0)
    return train, val, cal, test


def all_water_index(ps: PreparedSite, exclude: np.ndarray | None = None, stride: int = 1):
    w = ps.water[R:-R, R:-R] > 0
    if exclude is not None:
        w &= ~exclude
    rr, cc = np.where(w)
    if stride > 1:
        keep = (rr % stride == 0) & (cc % stride == 0)
        rr, cc = rr[keep], cc[keep]
    return rr, cc


def gather(ps: PreparedSite, rr: np.ndarray, cc: np.ndarray, augment: bool = False, rng=None):
    """Extract patches centred at (rr,cc) (unpadded coords)."""
    oy = np.arange(PATCH)[None, :, None] + rr[:, None, None]
    ox = np.arange(PATCH)[None, None, :] + cc[:, None, None]
    r = ps.refl[:, oy, ox].transpose(1, 0, 2, 3)  # (N,4,P,P)
    m = ps.water[oy, ox][:, None]
    if augment:
        k = rng.integers(0, 4)
        r = np.rot90(r, k, axes=(2, 3))
        m = np.rot90(m, k, axes=(2, 3))
        if rng.random() < 0.5:
            r, m = r[..., ::-1], m[..., ::-1]
        r = r * (1 + rng.normal(0, 0.01, size=(r.shape[0], 4, 1, 1)))
    n = len(rr)
    return (torch.from_numpy(np.ascontiguousarray(r, np.float32)),
            torch.from_numpy(np.ascontiguousarray(m, np.float32)),
            torch.from_numpy(np.repeat(ps.rinf[None], n, 0)),
            torch.from_numpy(np.repeat(ps.desc[None], n, 0)))


def labels(ps: PreparedSite, rr, cc):
    return torch.from_numpy(ps.site.depth[rr, cc].astype(np.float32))
