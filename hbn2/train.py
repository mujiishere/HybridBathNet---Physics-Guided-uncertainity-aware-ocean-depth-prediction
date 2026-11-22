"""Training, ensemble prediction and conformal calibration for HybridBathNet v2."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import PreparedSite, all_water_index, gather, labels
from .model import HybridBathNetV2, beta_nll, physics_loss

torch.set_num_threads(max(1, torch.get_num_threads()))


def _sample(sets, n, rng, weights):
    """sets: list of (ps, rr, cc). Site-weighted random draw."""
    p = np.asarray(weights, float)
    p = p / p.sum()
    which = rng.choice(len(sets), size=n, p=p)
    out = []
    for i, (ps, rr, cc) in enumerate(sets):
        k = int((which == i).sum())
        if k == 0 or len(rr) == 0:
            continue
        j = rng.integers(0, len(rr), k)
        out.append((ps, rr[j], cc[j]))
    return out


def _batch(parts, rng, augment, with_y=True):
    rs, ms, ris, ds, ys = [], [], [], [], []
    for ps, rr, cc in parts:
        r, m, ri, d = gather(ps, rr, cc, augment, rng)
        rs.append(r); ms.append(m); ris.append(ri); ds.append(d)
        if with_y:
            ys.append(labels(ps, rr, cc))
    cat = lambda x: torch.cat(x, 0)
    return cat(rs), cat(ms), cat(ris), cat(ds), (cat(ys) if with_y else None)


def train_member(train_sets, val_sets, unl_sets=None, cfg=None, seed=0, log=None):
    """train_sets/val_sets: list of (PreparedSite, rr, cc) labelled; unl_sets: list of
    (PreparedSite, rr, cc) unlabelled water pixels for physics consistency."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HybridBathNetV2(width=cfg["width"], n_blocks=cfg["blocks"], p_drop=cfg["dropout"],
                            use_physics_features=cfg["phys_feat"], use_film=cfg["film"],
                            use_physics_head=cfg["phys_head"])
    # normalisation fitted on training patches only
    w_sites = [len(s[1]) ** cfg["site_power"] for s in train_sets]
    r, m, ri, d, _ = _batch(_sample(train_sets, 4096, rng, w_sites), rng, False)
    with torch.no_grad():
        x = model.features(r, m, ri)
    model.set_norm(x, torch.cat([d, torch.log(ri.clamp(min=1e-4))], 1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    total = cfg["epochs"] * cfg["steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total, pct_start=0.1)
    vb = [_batch([s], rng, False) for s in val_sets if len(s[1])]
    best, best_state, bad, hist = np.inf, None, 0, []
    use_unl = cfg["phys_head"] and cfg["w_phys"] > 0 and unl_sets
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        agg = np.zeros(4)
        for _ in range(cfg["steps"]):
            r, m, ri, d, y = _batch(_sample(train_sets, cfg["batch"], rng, w_sites), rng, True)
            nl = len(y)
            if use_unl:
                ru, mu_, riu, du, _ = _batch(_sample(unl_sets, cfg["batch_unl"], rng, [1.0] * len(unl_sets)),
                                             rng, True, with_y=False)
                r, m, ri, d = (torch.cat([a, b], 0) for a, b in ((r, ru), (m, mu_), (ri, riu), (d, du)))
            out = model(r, m, ri, d)
            mu_l, var_l = out["mu"][:nl], out["var"][:nl]
            warm = min(1.0, ep / max(1, cfg["nll_warmup"]))
            l_nll = beta_nll(mu_l, var_l, y, cfg["beta"]) if cfg["uncertainty"] else torch.zeros(())
            l_hub = F.smooth_l1_loss(mu_l, y, beta=1.0)
            loss = cfg["w_huber"] * l_hub + warm * l_nll
            l_ph = torch.zeros(())
            if cfg["phys_head"] and cfg["w_phys"] > 0:
                c = r.shape[2] // 2
                l_ph = physics_loss(out, r[:, :, c, c])
                loss = loss + cfg["w_phys"] * l_ph
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg += [loss.item(), l_hub.item(), float(l_nll.detach()), float(l_ph.detach())]
        model.eval()
        with torch.no_grad():
            se, n = 0.0, 0
            for r, m, ri, d, y in vb:
                mu = model(r, m, ri, d)["mu"]
                se += float(((mu - y) ** 2).sum()); n += len(y)
        vr = float(np.sqrt(se / max(n, 1)))
        hist.append({"epoch": ep, "loss": agg[0] / cfg["steps"], "huber": agg[1] / cfg["steps"],
                     "nll": agg[2] / cfg["steps"], "phys": agg[3] / cfg["steps"], "val_rmse": vr})
        if log:
            log(f"  ep{ep:02d} loss {hist[-1]['loss']:.3f} phys {hist[-1]['phys']:.4f} val_rmse {vr:.3f} ({time.time()-t0:.0f}s)")
        if vr < best - 1e-4:
            best, best_state, bad = vr, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"] and ep >= int(cfg["min_epoch_frac"] * cfg["epochs"]):
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_rmse": best, "epochs_run": len(hist), "history": hist, "cfg": cfg}


DEFAULT_CFG = dict(width=32, blocks=3, dropout=0.1, phys_feat=True, film=True, phys_head=True,
                   uncertainty=True, lr=2e-3, wd=1e-4, epochs=25, steps=60, batch=96, batch_unl=64,
                   patience=5, min_epoch_frac=0.6, nll_warmup=4, beta=0.5, w_huber=1.0, w_phys=0.5, site_power=0.5)


@torch.no_grad()
def predict(models, ps: PreparedSite, rr, cc, chunk=4096, mc=0):
    """Ensemble prediction. Returns dict of numpy arrays: mu, aleatoric var, epistemic var,
    zmax, k0, albedo. ``mc``>0 adds MC-dropout passes per member."""
    mus, vars_, zmx, k0s, albs = [], [], [], [], []
    for mdl in models:
        mdl.eval()
        if mc:
            for mod in mdl.modules():
                if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout2d)):
                    mod.train()
        m_mu, m_var, m_z, m_k, m_a = [], [], [], [], []
        for s in range(0, len(rr), chunk):
            r, m, ri, d = gather(ps, rr[s:s + chunk], cc[s:s + chunk])
            passes = [mdl(r, m, ri, d) for _ in range(max(1, mc))]
            m_mu.append(torch.stack([p["mu"] for p in passes]).numpy())
            m_var.append(torch.stack([p["var"] for p in passes]).mean(0).numpy())
            if "zmax" in passes[0]:
                m_z.append(passes[0]["zmax"].numpy()); m_k.append(passes[0]["k0"].numpy()); m_a.append(passes[0]["albedo"].numpy())
        mus.append(np.concatenate(m_mu, 1)); vars_.append(np.concatenate(m_var))
        if m_z:
            zmx.append(np.concatenate(m_z)); k0s.append(np.concatenate(m_k)); albs.append(np.concatenate(m_a))
    allmu = np.concatenate(mus, 0)  # (members*passes, N)
    out = {"mu": allmu.mean(0), "alea": np.mean(vars_, 0), "epi": allmu.var(0)}
    out["sigma"] = np.sqrt(out["alea"] + out["epi"])
    if zmx:
        out["zmax"] = np.median(zmx, 0); out["k0"] = np.median(k0s, 0); out["albedo"] = np.median(albs, 0)
    return out


class ConformalCalibrator:
    """Normalised split-conformal (Papadopoulos 2002; Angelopoulos & Bates 2023) with optional
    Mondrian classes by predicted depth. Scores s=|y-mu|/sigma on spatially held-out
    calibration blocks; interval mu +/- q_{alpha} sigma."""

    def __init__(self, alphas=(0.32, 0.10, 0.05), bins=(0, 5, 10, 1e9)):
        self.alphas, self.bins, self.q = alphas, bins, {}

    def fit(self, y, mu, sigma):
        s = np.abs(y - mu) / sigma
        b = np.digitize(mu, self.bins[1:-1])
        for a in self.alphas:
            qs = []
            for k in range(len(self.bins) - 1):
                sk = s[b == k]
                if len(sk) < 30:
                    sk = s
                n = len(sk)
                lvl = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
                qs.append(float(np.quantile(sk, lvl, method="higher")))
            self.q[a] = qs
        return self

    def scale(self, mu, alpha):
        b = np.digitize(mu, self.bins[1:-1])
        return np.asarray(self.q[alpha])[b]

    def interval(self, mu, sigma, alpha):
        h = self.scale(mu, alpha) * sigma
        return np.clip(mu - h, 0, None), mu + h

    def to_dict(self):
        return {"alphas": list(self.alphas), "bins": list(self.bins), "q": {str(k): v for k, v in self.q.items()}}
