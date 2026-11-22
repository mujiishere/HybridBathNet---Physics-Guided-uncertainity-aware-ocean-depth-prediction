"""Aggregate study outputs into paper tables, figures and machine-readable metrics.

python -m hbn2.evaluate --runs runs_v2 --data data/sites --out reports_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import Lyzenga, Stumpf
from .data import assign_folds, prepare, split_masks
from .metrics import binned, depth_metrics, uncertainty_metrics
from .sites import SiteData
from .train import ConformalCalibrator

SITES = ["ponce_pr", "seribu_id", "langebaan_za"]
LABEL = {"ponce_pr": "Ponce (PR)", "seribu_id": "Kep. Seribu (ID)", "langebaan_za": "Langebaan (ZA)", "kerala_in": "Kerala (IN)"}
ALPHAS = {0.68: 0.32, 0.90: 0.10, 0.95: 0.05}


def load_fold(fdir: Path, split: str, site: str):
    p = fdir / f"pred_{split}_{site}.npz"
    return dict(np.load(p)) if p.exists() else None


def block_bootstrap_ci(y, p, blocks, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    vals = []
    for _ in range(n):
        s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        vals.append(np.sqrt(np.mean((p[s] - y[s]) ** 2)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def collect_cv(run_dir: Path, P: dict, k=5):
    """Out-of-fold predictions for all sites.

    Two calibrations are reported:
      * split conformal: quantiles from the fold's own held-out calibration blocks (pooled over sites);
      * cross-conformal (primary): for test fold f and site s, normalised scores |y-mu|/sigma of the
        out-of-fold TEST predictions of the other k-1 folds of the same site (Mondrian by site and
        predicted-depth bin; falls back to all sites if a class has < 30 scores). This spreads
        calibration over every spatial region instead of one, which matters when residual scale
        varies spatially (non-exchangeable blocks)."""
    per = []   # list of (fold, site, dict)
    cal_q = []
    for f in range(k):
        fd = run_dir / f"fold{f}"
        if not (fd / "done.json").exists():
            continue
        cal = [c for c in (load_fold(fd, "cal", s) for s in SITES) if c is not None]
        cc = ConformalCalibrator(alphas=tuple(ALPHAS.values())).fit(
            np.concatenate([c["y"] for c in cal]), np.concatenate([c["mu"] for c in cal]),
            np.concatenate([c["sigma"] for c in cal]))
        cal_q.append(cc.to_dict())
        for s in SITES:
            t = load_fold(fd, "test", s)
            if t is None:
                continue
            t = dict(t)
            if "zmax" not in t:
                t["zmax"] = np.full(len(t["y"]), np.nan)
            for c, a in ALPHAS.items():
                t[f"lo_split{c}"], t[f"hi_split{c}"] = cc.interval(t["mu"], t["sigma"], a)
            per.append((f, s, t))
    for f, s, t in per:
        own = [u for g, ss, u in per if g != f and ss == s]
        allo = [u for g, ss, u in per if g != f]
        src = own if own and sum(len(u["y"]) for u in own) >= 30 else allo
        if not src:
            for c in ALPHAS:
                t[f"lo{c}"], t[f"hi{c}"] = t[f"lo_split{c}"], t[f"hi_split{c}"]
            continue
        xc = ConformalCalibrator(alphas=tuple(ALPHAS.values())).fit(
            np.concatenate([u["y"] for u in src]), np.concatenate([u["mu"] for u in src]),
            np.concatenate([u["sigma"] for u in src]))
        for c, a in ALPHAS.items():
            t[f"lo{c}"], t[f"hi{c}"] = xc.interval(t["mu"], t["sigma"], a)
    out = {}
    keys = ["y", "mu", "sigma", "alea", "epi", "zmax", "rr", "cc"] + [f"{p}{c}" for c in ALPHAS for p in ("lo", "hi", "lo_split", "hi_split")]
    for s in SITES:
        ts = [t for _, ss, t in per if ss == s]
        if ts:
            out[s] = {kk: np.concatenate([t[kk] for t in ts]) for kk in keys}
            out[s]["blocks"] = P[s].blocks[out[s]["rr"], out[s]["cc"]]
    return out, cal_q


def site_report(d, with_unc=True):
    y, mu = d["y"], d["mu"]
    r = {"accuracy": depth_metrics(y, mu), "bins": binned(y, mu),
         "rmse_ci95_block_bootstrap": block_bootstrap_ci(y, mu, d["blocks"])}
    if with_unc:
        ints = {c: (d[f"lo{c}"], d[f"hi{c}"]) for c in ALPHAS}
        r["uncertainty"] = uncertainty_metrics(y, mu, d["sigma"], ints)
        for c in ALPHAS:
            lo, hi = d[f"lo_split{c}"], d[f"hi_split{c}"]
            r["uncertainty"][f"picp{int(round(c*100))}_split_conformal"] = float(np.mean((y >= lo) & (y <= hi)) * 100)
            r["uncertainty"][f"mpiw{int(round(c*100))}_split_conformal"] = float(np.mean(hi - lo))
        r["uncertainty"]["mean_aleatoric_sd"] = float(np.sqrt(d["alea"]).mean())
        r["uncertainty"]["mean_epistemic_sd"] = float(np.sqrt(d["epi"]).mean())
        z = d["zmax"]
        if np.isfinite(z).any():
            beyond = mu > z
            r["optical_limit"] = {"median_zmax_m": float(np.nanmedian(z)), "share_beyond_zmax_pct": float(beyond.mean() * 100),
                                  "rmse_within": float(np.sqrt(np.mean((mu[~beyond] - y[~beyond]) ** 2))) if (~beyond).any() else None,
                                  "rmse_beyond": float(np.sqrt(np.mean((mu[beyond] - y[beyond]) ** 2))) if beyond.any() else None}
    return r


def baselines_cv(P: dict, k=5):
    """Stumpf & Lyzenga fitted per site on exactly the same training pixels per fold."""
    res = {}
    for name, ctor in (("stumpf", Stumpf), ("lyzenga", Lyzenga)):
        res[name] = {}
        for s, ps in P.items():
            folds = assign_folds(ps, k)
            ys, ps_, bl = [], [], []
            for f in range(k):
                a, b, c, d = split_masks(ps, folds, f, k)
                tr = np.where(a); te = np.where(d)
                if len(te[0]) == 0 or len(tr[0]) < 5:
                    continue
                m = ctor().fit([ps], [tr], [ps.site.depth[tr]])
                ys.append(ps.site.depth[te]); ps_.append(m.predict(ps, *te)); bl.append(ps.blocks[te])
            y, p, b = map(np.concatenate, (ys, ps_, bl))
            res[name][s] = {"accuracy": depth_metrics(y, p), "bins": binned(y, p),
                            "rmse_ci95_block_bootstrap": block_bootstrap_ci(y, p, b)}
    return res


def loso(run_dir: Path, P: dict, n_shot=30, reps=20):
    out = {}
    for s in SITES:
        out[s] = {}
        for mode in ("zeroshot", "adapt"):
            p = run_dir / f"{s}_{mode}" / f"pred_target_{s}.npz"
            if not p.exists():
                continue
            d = dict(np.load(p))
            y, mu = d["y"], d["mu"]
            r = {"accuracy": depth_metrics(y, mu), "bins": binned(y, mu),
                 "uncertainty": uncertainty_metrics(y, mu, d["sigma"])}
            # few-shot: linear recalibration of the network output with n target soundings vs Stumpf on same soundings
            rng = np.random.default_rng(0)
            fs_net, fs_st = [], []
            ps = P[s]
            for _ in range(reps):
                k = rng.choice(len(y), min(n_shot, len(y) // 2), replace=False)
                rest = np.setdiff1d(np.arange(len(y)), k)
                A = np.c_[np.ones(len(k)), mu[k]]
                w = np.linalg.lstsq(A, y[k], rcond=None)[0]
                fs_net.append(np.sqrt(np.mean((w[0] + w[1] * mu[rest] - y[rest]) ** 2)))
                st = Stumpf().fit([ps], [(d["rr"][k], d["cc"][k])], [y[k]])
                fs_st.append(np.sqrt(np.mean((st.predict(ps, d["rr"][rest], d["cc"][rest]) - y[rest]) ** 2)))
            r[f"fewshot{n_shot}_recal_rmse"] = [float(np.mean(fs_net)), float(np.std(fs_net))]
            r[f"fewshot{n_shot}_stumpf_rmse"] = [float(np.mean(fs_st)), float(np.std(fs_st))]
            out[s][mode] = r
    return out


def full_single_member(run_dir: Path, P: dict, k=5):
    """Matched comparator for 1-member ablations: member0 of the full model, same folds/seeds."""
    from .products import load_members
    from .train import predict
    acc = {s: {"y": [], "mu": [], "sigma": []} for s in SITES}
    for f in range(k):
        m = run_dir / f"fold{f}" / "member0.pt"
        if not m.exists():
            continue
        net = load_members([m])
        for s in SITES:
            t = load_fold(run_dir / f"fold{f}", "test", s)
            if t is None:
                continue
            o = predict(net, P[s], t["rr"], t["cc"])
            acc[s]["y"].append(t["y"]); acc[s]["mu"].append(o["mu"]); acc[s]["sigma"].append(o["sigma"])
    out = {}
    for s, v in acc.items():
        if v["y"]:
            y, mu, sg = (np.concatenate(v[k_]) for k_ in ("y", "mu", "sigma"))
            out[s] = {"accuracy": depth_metrics(y, mu),
                      "uncertainty": {k_: x for k_, x in uncertainty_metrics(y, mu, sg).items() if k_ != "sparsification_curve"}}
    return out


def figures(cv, base, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    # 1. scatter per site
    fig, ax = plt.subplots(1, len(SITES), figsize=(3.7 * len(SITES), 3.6), squeeze=False); ax = ax[0]
    for a, s in zip(ax, SITES):
        if s not in cv:
            continue
        d = cv[s]; y, mu = d["y"], d["mu"]
        lim = max(y.max(), mu.max()) * 1.05
        if len(y) > 5000:
            a.hexbin(y, mu, gridsize=60, cmap="Blues", bins="log", extent=(0, lim, 0, lim), mincnt=1)
        else:
            a.errorbar(y, mu, yerr=d["sigma"], fmt="o", ms=2.5, alpha=.5, lw=.5, color="#0E3A5B")
        a.plot([0, lim], [0, lim], "--", color="#B8237A", lw=1)
        m = depth_metrics(y, mu)
        a.set_title(f"{LABEL[s]}\nRMSE {m['rmse']:.2f} m · R² {m['r2']:.2f} · n={m['n']}")
        a.set_xlabel("reference depth (m)"); a.set_ylabel("predicted depth (m)"); a.set_xlim(0, lim); a.set_ylim(0, lim)
    fig.tight_layout(); fig.savefig(out / "fig_scatter.png", dpi=200); plt.close(fig)
    # 2. reliability diagram
    from scipy.stats import norm
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    lv = np.linspace(0.05, 0.95, 19)
    for s, c in zip(SITES, ("#0E3A5B", "#5E9CC0", "#B8237A", "#C9A227")):
        if s not in cv:
            continue
        d = cv[s]; e = np.abs(d["y"] - d["mu"])
        ax[0].plot(lv, [np.mean(e <= norm.ppf(0.5 + q / 2) * d["sigma"]) for q in lv], "-o", ms=3, color=c, label=LABEL[s])
        cov = [np.mean((d["y"] >= d[f"lo{q}"]) & (d["y"] <= d[f"hi{q}"])) for q in ALPHAS]
        ax[1].plot(list(ALPHAS), cov, "-s", color=c, label=LABEL[s])
    for a, t in zip(ax, ("Gaussian (ensemble σ, uncalibrated)", "Cross-conformal intervals")):
        a.plot([0, 1], [0, 1], "--", color="grey", lw=1); a.set_xlabel("nominal coverage"); a.set_ylabel("empirical coverage")
        a.set_title(t); a.legend(fontsize=8); a.set_xlim(0, 1); a.set_ylim(0, 1)
    ax[1].set_xlim(0.6, 1); ax[1].set_ylim(0.6, 1)
    fig.tight_layout(); fig.savefig(out / "fig_calibration.png", dpi=200); plt.close(fig)
    # 3. sparsification
    fig, a = plt.subplots(figsize=(4.4, 3.4))
    for s, c in zip(SITES, ("#0E3A5B", "#5E9CC0", "#B8237A", "#C9A227")):
        if s not in cv:
            continue
        um = uncertainty_metrics(cv[s]["y"], cv[s]["mu"], cv[s]["sigma"])
        a.plot(np.linspace(0, 90, 10), um["sparsification_curve"], "-o", ms=3, color=c, label=LABEL[s])
    a.set_xlabel("% most-uncertain cells removed"); a.set_ylabel("RMSE of retained cells (m)"); a.legend(fontsize=8)
    a.set_title("Uncertainty ranks error"); fig.tight_layout(); fig.savefig(out / "fig_sparsification.png", dpi=200); plt.close(fig)
    # 4. depth-bin RMSE vs baselines
    fig, ax = plt.subplots(1, len(SITES), figsize=(3.7 * len(SITES), 3.3), squeeze=False); ax = ax[0]
    for a, s in zip(ax, SITES):
        if s not in cv:
            continue
        bins = list(binned(cv[s]["y"], cv[s]["mu"]).keys())
        xs = np.arange(len(bins))
        for j, (nm, src, col) in enumerate((("HybridBathNet v2", binned(cv[s]["y"], cv[s]["mu"]), "#0E3A5B"),
                                            ("Lyzenga", base["lyzenga"][s]["bins"], "#5E9CC0"),
                                            ("Stumpf", base["stumpf"][s]["bins"], "#C9A227"))):
            a.bar(xs + (j - 1) * 0.27, [src[b].get("rmse", np.nan) for b in bins], 0.27, label=nm, color=col)
        a.set_xticks(xs, [f"{b} m" for b in bins], rotation=0); a.set_ylabel("RMSE (m)"); a.set_title(LABEL[s])
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig_depth_bins.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_v2"); ap.add_argument("--data", default="data/v2_sites")
    ap.add_argument("--out", default="reports_v2")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sj = runs / "cv_full" / "study.json"
    if sj.exists():
        SITES[:] = json.loads(sj.read_text())["sites"]
    P = {s: prepare(SiteData.load(Path(a.data) / f"{s}.npz")) for s in SITES}
    res = {"sites": {s: P[s].site.summary() for s in SITES}}
    cv, calq = collect_cv(runs / "cv_full", P)
    res["hybridbathnet_v2"] = {s: site_report(d) for s, d in cv.items()}
    res["conformal_quantiles_per_fold"] = calq
    res["baselines"] = baselines_cv(P)
    res["ablations"] = {}
    for c in ("cnn_only", "no_physics_loss", "no_physics_features", "no_film"):
        d = runs / f"abl_{c}"
        if d.exists():
            ab, _ = collect_cv(d, P)
            if ab:
                res["ablations"][c] = {s: {"accuracy": depth_metrics(v["y"], v["mu"]),
                                           "uncertainty": {k: x for k, x in uncertainty_metrics(v["y"], v["mu"], v["sigma"]).items()
                                                           if k != "sparsification_curve"}} for s, v in ab.items()}
    # matched single-member comparator for ablations: member-0-only is not saved separately, so we
    # report the 3-member ensemble and note it; ablations use 1 member (see report text).
    if res["ablations"]:
        res["ablations"]["full_single_member"] = full_single_member(runs / "cv_full", P)
    res["loso"] = loso(runs / "loso", P)
    json.dump(res, open(out / "study_metrics.json", "w"), indent=1, default=float)
    if cv:
        figures(cv, res["baselines"], out / "figures")
    print(json.dumps({s: res["hybridbathnet_v2"][s]["accuracy"] for s in res["hybridbathnet_v2"]}, indent=1))
    return res


if __name__ == "__main__":
    main()
