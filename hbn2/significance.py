"""Paired spatial-block bootstrap of RMSE differences (model A - model B) on identical held-out pixels."""
import json
from pathlib import Path

import numpy as np

from .baselines import Stumpf
from .data import assign_folds, prepare, split_masks
from .evaluate import SITES, collect_cv
from .products import load_members
from .sites import SiteData
from .train import predict


def paired(y, a, b, blocks, n=2000, seed=0):
    rng = np.random.default_rng(seed); ub = np.unique(blocks); idx = {k: np.where(blocks == k)[0] for k in ub}
    d = []
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(ub, len(ub))])
        d.append(np.sqrt(np.mean((a[s] - y[s]) ** 2)) - np.sqrt(np.mean((b[s] - y[s]) ** 2)))
    d = np.array(d)
    return {"diff": float(np.sqrt(np.mean((a - y) ** 2)) - np.sqrt(np.mean((b - y) ** 2))),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_a_not_better": float(np.mean(d >= 0))}


def key(rr, cc):
    return rr.astype(np.int64) * 100000 + cc


def main(runs="runs_v2", data="data/v2_sites", out="reports_v2/significance.json"):
    runs = Path(runs)
    sj = runs / "cv_full" / "study.json"
    if sj.exists():
        SITES[:] = json.loads(sj.read_text())["sites"]
    P = {s: prepare(SiteData.load(Path(data) / f"{s}.npz")) for s in SITES}
    full, _ = collect_cv(runs / "cv_full", P)
    res = {}
    # single-member full predictions (same as evaluate.full_single_member) keyed by pixel
    single = {s: {} for s in SITES}
    for f in range(5):
        net = load_members([runs / "cv_full" / f"fold{f}" / "member0.pt"])
        for s in SITES:
            t = np.load(runs / "cv_full" / f"fold{f}" / f"pred_test_{s}.npz")
            o = predict(net, P[s], t["rr"], t["cc"])
            single[s].update(dict(zip(key(t["rr"], t["cc"]), o["mu"])))
    for s in SITES:
        d = full[s]; k = key(d["rr"], d["cc"]); y = d["y"]; b = d["blocks"]
        r = {}
        # Stumpf on identical folds
        ps = P[s]; folds = assign_folds(ps, 5); st = {}
        for f in range(5):
            a_, _, _, te = split_masks(ps, folds, f, 5); tr = np.where(a_); te = np.where(te)
            if len(te[0]) and len(tr[0]) > 5:
                m = Stumpf().fit([ps], [tr], [ps.site.depth[tr]])
                st.update(dict(zip(key(*te), m.predict(ps, *te))))
        sp = np.array([st[x] for x in k])
        r["ensemble_vs_stumpf"] = paired(y, d["mu"], sp, b)
        sm = np.array([single[s][x] for x in k])
        for c in ("cnn_only", "no_physics_loss"):
            ab, _ = collect_cv(runs / f"abl_{c}", P)
            if s in ab:
                m = dict(zip(key(ab[s]["rr"], ab[s]["cc"]), ab[s]["mu"]))
                r[f"full1_vs_{c}"] = paired(y, sm, np.array([m[x] for x in k]), b)
        res[s] = r
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
