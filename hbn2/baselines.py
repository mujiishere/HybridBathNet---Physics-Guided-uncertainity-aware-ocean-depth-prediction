"""Classical empirical SDB baselines fitted on the same training pixels:
Stumpf et al. (2003) log-ratio and Lyzenga (1985) / Lyzenga et al. (2006) multi-band log-linear.
Both use deglinted reflectance and are fitted by ordinary least squares."""
from __future__ import annotations

import numpy as np

from .data import R, PreparedSite


def _centre(ps: PreparedSite, rr, cc):
    return ps.refl[:, rr + R, cc + R]


def stumpf_x(ps, rr, cc, n=1000.0):
    r = _centre(ps, rr, cc)
    return (np.log(np.clip(n * r[0], 1.0001, None)) / np.log(np.clip(n * r[1], 1.0001, None)))[:, None]


def lyzenga_x(ps, rr, cc):
    r = _centre(ps, rr, cc)
    return np.stack([np.log(np.clip(r[i] - ps.rinf[i], 1e-4, None)) for i in range(3)], 1)


class LinearSDB:
    def __init__(self, feat):
        self.feat = feat

    def fit(self, ps_list, idx_list, ys):
        X = np.concatenate([self.feat(p, *ix) for p, ix in zip(ps_list, idx_list)])
        X = np.c_[np.ones(len(X)), X]
        self.w = np.linalg.lstsq(X, np.concatenate(ys), rcond=None)[0]
        res = np.concatenate(ys) - X @ self.w
        self.sigma = float(np.std(res))
        return self

    def predict(self, ps, rr, cc):
        X = np.c_[np.ones(len(rr)), self.feat(ps, rr, cc)]
        return np.clip(X @ self.w, 0, None)


Stumpf = lambda: LinearSDB(stumpf_x)
Lyzenga = lambda: LinearSDB(lyzenga_x)
