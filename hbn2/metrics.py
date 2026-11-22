"""Accuracy and uncertainty metrics for SDB (all inputs in metres, positive down)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, spearmanr

BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20)]


def depth_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    if len(y) == 0:
        return {"n": 0}
    ss = ((y - y.mean()) ** 2).sum()
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "nmad": float(1.4826 * np.median(np.abs(e - np.median(e)))),
        "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else None,
        "rel_rmse_pct": float(100 * np.sqrt(np.mean(e ** 2)) / max(np.mean(y), 1e-6)),
        "within_0.5m": float(np.mean(np.abs(e) <= 0.5) * 100),
        "within_1m": float(np.mean(np.abs(e) <= 1) * 100),
        "within_2m": float(np.mean(np.abs(e) <= 2) * 100),
        # IHO S-44 (2020) Order 2 TVU at 95%: sqrt(1^2 + (0.023 d)^2); share of points within the limit
        "iho_order2_pct": float(np.mean(np.abs(e) <= np.sqrt(1.0 + (0.023 * y) ** 2) * 1.0) * 100),
    }


def binned(y, p):
    out = {}
    for lo, hi in BINS:
        k = (y >= lo) & (y < hi) if hi < 20 else (y >= lo) & (y <= hi)
        if k.sum() >= 10:
            d = depth_metrics(y[k], p[k])
            out[f"{lo}-{hi}"] = {"n": d["n"], "rmse": d["rmse"], "mae": d["mae"], "bias": d["bias"]}
        else:
            out[f"{lo}-{hi}"] = {"n": int(k.sum())}
    return out


def crps_gaussian(y, mu, s):
    z = (y - mu) / s
    return float(np.mean(s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def uncertainty_metrics(y, mu, sigma, intervals: dict | None = None):
    """intervals: {nominal_coverage: (lo, hi)} (e.g. conformal). Gaussian intervals are always reported."""
    y, mu, sigma = map(lambda a: np.asarray(a, float), (y, mu, sigma))
    e = np.abs(y - mu)
    out = {
        "nll": float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2))),
        "crps": crps_gaussian(y, mu, sigma),
        "mean_sigma": float(sigma.mean()),
        "spearman_sigma_abs_err": float(spearmanr(sigma, e).correlation) if len(y) > 3 else None,
    }
    # Gaussian coverage + calibration error over 19 levels (Kuleshov et al. 2018)
    levels = np.linspace(0.05, 0.95, 19)
    obs = [np.mean(e <= norm.ppf(0.5 + c / 2) * sigma) for c in levels]
    out["ece_gauss"] = float(np.mean(np.abs(np.array(obs) - levels)))
    for c in (0.68, 0.90, 0.95):
        out[f"picp{int(c*100)}_gauss"] = float(np.mean(e <= norm.ppf(0.5 + c / 2) * sigma) * 100)
    if intervals:
        for c, (lo, hi) in intervals.items():
            out[f"picp{int(round(c*100))}_conformal"] = float(np.mean((y >= lo) & (y <= hi)) * 100)
            out[f"mpiw{int(round(c*100))}_conformal"] = float(np.mean(hi - lo))
    # sparsification: RMSE of retained fraction when removing most-uncertain first, vs oracle
    fr = np.linspace(0, 0.9, 10)
    o_s, o_e = np.argsort(-sigma), np.argsort(-e)
    curve, oracle = [], []
    for f in fr:
        k = int(len(y) * f)
        curve.append(np.sqrt(np.mean(e[o_s[k:]] ** 2)))
        oracle.append(np.sqrt(np.mean(e[o_e[k:]] ** 2)))
    out["ause"] = float(np.trapezoid(np.array(curve) - np.array(oracle), fr))
    out["rmse_keep50_by_sigma"] = float(curve[5])
    out["sparsification_curve"] = [float(c) for c in curve]
    return out
