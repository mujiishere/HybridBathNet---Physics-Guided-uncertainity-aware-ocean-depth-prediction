"""Unit tests for HybridBathNet v2 physics, splits, calibration, metrics and the model."""
import numpy as np
import pytest
import rasterio
import torch

from hbn2.data import R, PreparedSite, assign_folds, gather, prepare, split_masks
from hbn2.metrics import depth_metrics, uncertainty_metrics
from hbn2.model import HybridBathNetV2, beta_nll, physics_loss
from hbn2.physics import detection_limit, forward_reflectance, hedley_deglint
from hbn2.sites import SiteData, _rasterize_points, water_mask
from hbn2.train import ConformalCalibrator, predict


def synthetic_site(H=60, W=80, seed=0):
    rng = np.random.default_rng(seed)
    z = np.tile(np.linspace(0.5, 18, W), (H, 1)).astype(np.float32)
    z += rng.normal(0, 0.2, z.shape).astype(np.float32)
    kd = np.array([0.08, 0.12, 0.5]); rinf = np.array([0.02, 0.015, 0.003]); A = np.array([0.15, 0.2, 0.22])
    refl = np.zeros((4, H, W), np.float32)
    for i in range(3):
        refl[i] = rinf[i] + (A[i] - rinf[i]) * np.exp(-2 * kd[i] * z) + rng.normal(0, 0.0005, z.shape)
    refl[3] = 0.002 + rng.normal(0, 0.0003, z.shape)
    tr = rasterio.transform.from_origin(500000, 1000000, 10, 10)
    return SiteData("synth", refl, z, np.ones_like(z, np.int16), water_mask(refl), tr, "EPSG:32619")


def test_forward_model_limits():
    z = torch.tensor([0.0, 100.0])
    one = torch.ones(2)
    rinf = torch.tensor([[0.02, 0.015, 0.003, 0.001]] * 2)
    r, kd, a = forward_reflectance(z, 0.1 * one, 0.3 * one, 0.5 * one, rinf, one)
    assert torch.allclose(r[0], a[0], atol=1e-6)            # zero depth -> bottom albedo
    assert torch.allclose(r[1], rinf[1, :3], atol=1e-6)     # optically deep -> R_inf
    assert (kd > 0).all()


def test_detection_limit_decreases_with_turbidity():
    rinf = torch.tensor([[0.02, 0.015, 0.003, 0.001]] * 2)
    z = torch.ones(2); one = torch.ones(2)
    _, kd, a = forward_reflectance(z, torch.tensor([0.05, 0.8]), 0.3 * one, 0.5 * one, rinf, one)
    zm = detection_limit(kd, a, rinf)
    assert zm[0] > zm[1] > 0


def test_deglint_removes_nir_correlated_glint():
    s = synthetic_site()
    g = np.random.default_rng(1).uniform(0, 0.01, s.refl.shape[1:]).astype(np.float32)
    r = s.refl.copy(); r[:3] += g; r[3] += g
    out, info = hedley_deglint(r, s.water)
    assert np.abs(out[1] - s.refl[1]).mean() < np.abs(r[1] - s.refl[1]).mean()


def test_rasterize_points_median_and_count():
    tr = rasterio.transform.from_origin(0, 100, 10, 10)
    d, c = _rasterize_points([5, 6, 7, 55], [95, 95, 95, 45], [1, 2, 9, 4], tr, (10, 10))
    assert d[0, 0] == 2 and c[0, 0] == 3 and d[5, 5] == 4 and np.isnan(d[1, 1])


def test_water_mask_ignores_labels():
    s = synthetic_site()
    m1 = water_mask(s.refl)
    s.depth[:] = np.nan
    assert np.array_equal(m1, water_mask(s.refl))


def test_split_has_no_footprint_leakage():
    ps = prepare(synthetic_site(), block_m=100)
    folds = assign_folds(ps, 5)
    tr, va, ca, te = split_masks(ps, folds, 0)
    assert tr.sum() > 0 and te.sum() > 0
    assert not (tr & (te | va | ca)).any()
    ti, tj = np.where(tr); ei, ej = np.where(te | va | ca)
    dmin = np.min(np.maximum(np.abs(ti[:, None] - ei[None]), np.abs(tj[:, None] - ej[None])))
    assert dmin > R  # patch footprints of train never touch held-out centre pixels


def test_gather_shapes_and_centre():
    ps = prepare(synthetic_site())
    r, m, ri, d = gather(ps, np.array([10]), np.array([20]))
    assert r.shape == (1, 4, 13, 13) and m.shape == (1, 1, 13, 13) and d.shape == (1, 16)
    assert np.allclose(r[0, :, 6, 6].numpy(), ps.refl[:, 10 + R, 20 + R])


def test_model_forward_backward_finite():
    ps = prepare(synthetic_site())
    rr, cc = np.where(ps.labelled)
    r, m, ri, d = gather(ps, rr[:32], cc[:32])
    y = torch.from_numpy(ps.site.depth[rr[:32], cc[:32]])
    net = HybridBathNetV2(width=16)
    net.set_norm(net.features(r, m, ri), torch.cat([d, torch.log(ri)], 1))
    out = net(r, m, ri, d)
    loss = beta_nll(out["mu"], out["var"], y) + physics_loss(out, r[:, :, 6, 6])
    loss.backward()
    assert torch.isfinite(loss) and (out["mu"] > 0).all() and (out["var"] > 0).all() and (out["zmax"] > 0).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_predict_ensemble_variance_decomposition():
    ps = prepare(synthetic_site())
    rr, cc = np.where(ps.labelled)
    nets = []
    for s in range(2):
        torch.manual_seed(s); n = HybridBathNetV2(width=16)
        r, m, ri, d = gather(ps, rr[:64], cc[:64]); n.set_norm(n.features(r, m, ri), torch.cat([d, torch.log(ri)], 1))
        nets.append(n.eval())
    o = predict(nets, ps, rr[:50], cc[:50])
    assert np.allclose(o["sigma"] ** 2, o["alea"] + o["epi"], rtol=1e-5)
    assert (o["epi"] > 0).any()


def test_conformal_coverage_guarantee():
    rng = np.random.default_rng(0)
    n = 4000
    mu = rng.uniform(0, 20, n); sig = rng.uniform(0.3, 2, n)
    y = mu + rng.standard_t(4, n) * sig * 1.7  # miscalibrated, heavy-tailed
    cal = ConformalCalibrator().fit(y[:2000], mu[:2000], sig[:2000])
    for a in (0.32, 0.10, 0.05):
        lo, hi = cal.interval(mu[2000:], sig[2000:], a)
        cov = np.mean((y[2000:] >= lo) & (y[2000:] <= hi))
        assert cov >= 1 - a - 0.03


def test_metrics_identities():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 20, 500); p = y + rng.normal(0.3, 1, 500)
    m = depth_metrics(y, p)
    assert np.isclose(m["rmse"] ** 2, m["bias"] ** 2 + np.var(p - y), rtol=1e-9)
    u = uncertainty_metrics(y, p, np.ones(500))
    assert 60 < u["picp68_gauss"] < 76 and u["crps"] > 0
