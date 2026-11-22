"""Experiment runner for the HybridBathNet v2 multi-site study.

Modes
-----
cv    : pooled multi-site 5-fold spatial-block cross-validation (one configuration).
loso  : leave-one-site-out transfer (zero-shot, or physics-adapted with unlabelled target imagery).

Each trained member saves its state_dict; per-fold raw predictions for test and calibration
pixels are written as .npz so that all metrics/conformal steps are reproducible offline.
Usage:
  python -m hbn2.run_study cv --config full --members 3 --out runs_v2/cv_full
  python -m hbn2.run_study loso --target ponce_pr --adapt --members 2 --out runs_v2/loso
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import all_water_index, assign_folds, prepare, split_masks
from .sites import SiteData
from .train import predict, train_member

SITES = ["ponce_pr", "seribu_id", "langebaan_za"]
CONFIGS = {
    "full": {},
    "no_physics_loss": {"w_phys": 0.0, "phys_head": False},
    "no_physics_features": {"phys_feat": False},
    "no_film": {"film": False},
    "no_uncertainty": {"uncertainty": False},
    "cnn_only": {"w_phys": 0.0, "phys_head": False, "phys_feat": False, "film": False},
}


def load_sites(data_dir, names):
    return [prepare(SiteData.load(Path(data_dir) / f"{n}.npz")) for n in names]


def _save_pred(path, ps, rr, cc, o):
    np.savez_compressed(path, site=ps.name, rr=rr, cc=cc, y=ps.site.depth[rr, cc], **{k: v for k, v in o.items()})


def run_cv(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    P = load_sites(args.data, args.sites)
    folds = {p.name: assign_folds(p, args.k) for p in P}
    cfg = {**CONFIGS[args.config], "epochs": args.epochs}
    log = open(out / "log.txt", "a")
    say = lambda s: (print(s, flush=True), log.write(s + "\n"), log.flush())
    for f in args.folds:
        fdir = out / f"fold{f}"
        if (fdir / "done.json").exists():
            continue
        fdir.mkdir(exist_ok=True)
        tr, va, ca, te, unl = [], [], [], [], []
        for p in P:
            a, b, c, d = split_masks(p, folds[p.name], f, args.k)
            tr.append((p, *np.where(a))); va.append((p, *np.where(b)))
            ca.append((p, *np.where(c))); te.append((p, *np.where(d)))
            held = (folds[p.name] == f) | (folds[p.name] == (f + 1) % args.k)
            from scipy.ndimage import binary_dilation
            unl.append((p, *all_water_index(p, exclude=binary_dilation(held, structure=np.ones((3, 3), bool), iterations=6), stride=2)))
        models, infos = [], []
        for m in range(args.members):
            mp = fdir / f"member{m}.pt"
            say(f"[{args.config}] fold {f} member {m}")
            t0 = time.time()
            mdl, info = train_member(tr, va, unl, cfg=cfg, seed=1000 * f + m, log=say)
            torch.save({"state_dict": mdl.state_dict(), "cfg": info["cfg"]}, mp)
            info["seconds"] = time.time() - t0
            infos.append(info); models.append(mdl)
        for split, sets in (("test", te), ("cal", ca), ("val", va)):
            for p, rr, cc in sets:
                if len(rr):
                    _save_pred(fdir / f"pred_{split}_{p.name}.npz", p, rr, cc, predict(models, p, rr, cc))
        json.dump({"members": [{k: v for k, v in i.items() if k != "history"} for i in infos],
                   "histories": [i["history"] for i in infos],
                   "split_sizes": {p.name: {s: int(len(x[1])) for s, x in zip(("train", "val", "cal", "test"), (t, v, c, e))}
                                   for p, t, v, c, e in zip(P, tr, va, ca, te)}},
                  open(fdir / "done.json", "w"), indent=1)
    json.dump({"config": args.config, "cfg": cfg, "sites": args.sites, "k": args.k, "members": args.members},
              open(out / "study.json", "w"), indent=1)


def run_loso(args):
    out = Path(args.out) / f"{args.target}_{'adapt' if args.adapt else 'zeroshot'}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        return
    P = load_sites(args.data, args.sites)
    src = [p for p in P if p.name != args.target]
    tgt = [p for p in P if p.name == args.target][0]
    log = open(out / "log.txt", "a")
    say = lambda s: (print(s, flush=True), log.write(s + "\n"), log.flush())
    tr, va, unl = [], [], []
    for p in src:
        fo = assign_folds(p, args.k)
        a, b, c, d = split_masks(p, fo, 0, args.k)
        tr.append((p, *np.where(a | d))); va.append((p, *np.where(b | c)))
        unl.append((p, *all_water_index(p, stride=2)))
    if args.adapt:  # transductive physics adaptation: target imagery only, never target depths
        unl.append((tgt, *all_water_index(tgt, stride=2)))
    models = []
    for m in range(args.members):
        say(f"[loso {args.target} adapt={args.adapt}] member {m}")
        mdl, info = train_member(tr, va, unl, cfg={"epochs": args.epochs}, seed=77 + m, log=say)
        torch.save({"state_dict": mdl.state_dict(), "cfg": info["cfg"]}, out / f"member{m}.pt")
        models.append(mdl)
    rr, cc = np.where(tgt.labelled)
    _save_pred(out / f"pred_target_{tgt.name}.npz", tgt, rr, cc, predict(models, tgt, rr, cc))
    for p, r2, c2 in va:
        _save_pred(out / f"pred_srcval_{p.name}.npz", p, r2, c2, predict(models, p, r2, c2))
    json.dump({"target": args.target, "adapt": args.adapt, "members": args.members}, open(out / "done.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "loso"])
    ap.add_argument("--data", default="data/v2_sites")
    ap.add_argument("--sites", nargs="+", default=SITES)
    ap.add_argument("--config", default="full", choices=list(CONFIGS))
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--target", default="ponce_pr")
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    (run_cv if args.mode == "cv" else run_loso)(args)


if __name__ == "__main__":
    main()
