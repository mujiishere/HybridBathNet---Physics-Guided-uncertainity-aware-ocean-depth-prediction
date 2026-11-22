"""Optical physics used by HybridBathNet v2.

* Scene-level, image-only preprocessing: Hedley et al. (2005) sun-glint correction and
  Lyzenga deep-water reflectance R_inf estimation.
* Physics feature layer: Lyzenga (1978/1985) log-linear transforms and Stumpf et al. (2003)
  log-ratio.
* Differentiable shallow-water reflectance model of Maritorena, Morel & Gentili (1994):
      R(lambda, z) = R_inf + (A_b - R_inf) * exp(-2 K_d z)
  with K_d = K_w (pure water) + K0 * s(lambda) and a two-endmember bottom albedo
  (sand / vegetated), used as a consistency constraint and to derive the per-pixel
  optical detection limit z_max (depth beyond which the bottom signal falls below
  sensor+atmospheric noise).

All coefficient choices are approximate literature values, documented here, not a
calibrated inherent-optical-property retrieval.
"""
from __future__ import annotations

import numpy as np
import torch

# Pure-water diffuse attenuation (m^-1), approx. Pope & Fry 1997 / Morel 2007 at S2 B2,B3,B4
KW = (0.0166, 0.0640, 0.4245)
# spectral shape of non-water (CDOM + particle) attenuation, normalised to blue
KSHAPE = (1.0, 0.72, 0.55)
# bottom albedo spectral shapes (relative), sand and seagrass/algae, B2,B3,B4
SAND = (0.20, 0.27, 0.30)
VEG = (0.04, 0.09, 0.05)
# effective radiometric noise (sensor NE-delta-rho + residual atmospheric correction), reflectance units
NOISE = (0.0030, 0.0025, 0.0020)


def hedley_deglint(refl: np.ndarray, water: np.ndarray) -> tuple[np.ndarray, dict]:
    """Hedley et al. (2005) deglinting. Regression of each visible band on NIR over
    optically-deep water candidates (darkest 10 % green reflectance of water pixels)."""
    out = refl.copy()
    g = refl[1][water]
    if g.size < 100:
        return out, {"slopes": [0, 0, 0], "nir_min": 0.0}
    deep = water & (refl[1] <= np.nanpercentile(g, 10))
    nir = refl[3][deep]
    nir_min = float(np.nanpercentile(refl[3][water], 1))
    slopes = []
    for i in range(3):
        v = refl[i][deep]
        ok = np.isfinite(v) & np.isfinite(nir)
        if ok.sum() > 50 and np.std(nir[ok]) > 1e-5:
            b = float(np.polyfit(nir[ok], v[ok], 1)[0])
            b = float(np.clip(b, 0.0, 1.5))
        else:
            b = 0.0
        slopes.append(b)
        out[i] = refl[i] - b * (refl[3] - nir_min)
    out[:3] = np.clip(out[:3], 1e-4, None)
    return out, {"slopes": slopes, "nir_min": nir_min}


def scene_stats(refl: np.ndarray, water: np.ndarray) -> dict:
    """Image-only scene descriptors: deep-water reflectance R_inf (2nd percentile of
    low-glint water pixels) and a 16-d water-type descriptor for FiLM conditioning."""
    wm = water & np.isfinite(refl).all(0)
    nir = refl[3][wm]
    low = wm & (refl[3] <= np.nanpercentile(nir, 50))
    rinf = [float(np.nanpercentile(refl[i][low], 2)) for i in range(4)]
    desc = []
    for i in range(4):
        v = np.log(np.clip(refl[i][wm], 1e-4, None))
        desc += [float(np.nanpercentile(v, q)) for q in (5, 25, 50, 95)]
    return {"rinf": rinf, "desc": desc}


# ---------------------------------------------------------------------------
# torch ops
# ---------------------------------------------------------------------------
def physics_features(r: torch.Tensor, rinf: torch.Tensor) -> torch.Tensor:
    """r: (N,4,h,w) deglinted reflectance; rinf: (N,4). Returns (N,10,h,w) physics features:
    Lyzenga X_i=ln(R_i-R_inf_i) for B,G,R; ln(R) for B,G,R,NIR relative to deep water;
    Stumpf ln(1000B)/ln(1000G); ln(B/G); ln(G/R)."""
    eps = 1e-4
    ri = rinf[:, :, None, None]
    lyz = torch.log(torch.clamp(r[:, :3] - ri[:, :3], min=eps))
    lnr = torch.log(torch.clamp(r, min=eps)) - torch.log(torch.clamp(ri, min=eps))
    lb, lg, lr = (torch.log(torch.clamp(1000 * r[:, i], min=1.0001)) for i in range(3))
    stumpf = lb / lg
    rbg = torch.log(torch.clamp(r[:, 0], min=eps)) - torch.log(torch.clamp(r[:, 1], min=eps))
    rgr = torch.log(torch.clamp(r[:, 1], min=eps)) - torch.log(torch.clamp(r[:, 2], min=eps))
    return torch.cat([lyz, lnr, stumpf[:, None], rbg[:, None], rgr[:, None]], 1)


def forward_reflectance(z, k0, albedo, mix, rinf, gain):
    """Maritorena et al. (1994) model for B,G,R. z,k0,albedo,mix,gain: (N,), rinf: (N,4).
    Returns predicted R (N,3), Kd (N,3), bottom term A (N,3)."""
    kw = z.new_tensor(KW)
    ks = z.new_tensor(KSHAPE)
    kd = kw[None] + k0[:, None] * ks[None]
    a = albedo[:, None] * (mix[:, None] * z.new_tensor(SAND)[None] + (1 - mix[:, None]) * z.new_tensor(VEG)[None]) / 0.3
    a = gain[:, None] * a
    rin = rinf[:, :3]
    r = rin + (a - rin) * torch.exp(-2.0 * kd * z[:, None])
    return r, kd, a


def detection_limit(kd: torch.Tensor, a: torch.Tensor, rinf: torch.Tensor) -> torch.Tensor:
    """Optical detection limit (m): deepest z where |A-R_inf| exp(-2 Kd z) > noise, max over bands."""
    noise = kd.new_tensor(NOISE)[None]
    contrast = torch.clamp((a - rinf[:, :3]).abs(), min=1e-6)
    zmax = torch.log(torch.clamp(contrast / noise, min=1.0)) / (2.0 * kd)
    return zmax.max(1).values
