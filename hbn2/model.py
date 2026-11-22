"""HybridBathNet v2: physics-guided, scene-conditioned, uncertainty-aware patch network.

Input  : deglinted reflectance patch (N,4,P,P), water mask patch (N,1,P,P),
         scene deep-water reflectance R_inf (N,4) and scene descriptor (N,16).
Outputs: mean depth mu (m), aleatoric variance (m^2), optical-model parameters
         (K0, bottom albedo, sand/vegetation mix, gain), reconstructed centre reflectance,
         and the optical detection limit z_max (m).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import detection_limit, forward_reflectance, physics_features


class FiLMBlock(nn.Module):
    def __init__(self, ch: int, cond: int, p_drop: float):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.film = nn.Linear(cond, 2 * ch) if cond else None
        self.drop = nn.Dropout2d(p_drop)

    def forward(self, x, c):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        if self.film is not None:
            g, b = self.film(c).chunk(2, 1)
            h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return F.gelu(x + self.drop(h))


class HybridBathNetV2(nn.Module):
    def __init__(self, width: int = 48, n_blocks: int = 3, p_drop: float = 0.1,
                 use_physics_features: bool = True, use_film: bool = True,
                 use_physics_head: bool = True, zmax_prior: float = 30.0):
        super().__init__()
        self.use_pf = use_physics_features
        self.use_film = use_film
        self.use_ph = use_physics_head
        in_ch = 4 + 1 + (10 if use_physics_features else 0)
        cond = 32 if use_film else 0
        self.scene = nn.Sequential(nn.Linear(16 + 4, 64), nn.GELU(), nn.Linear(64, cond)) if use_film else None
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU())
        self.blocks = nn.ModuleList([FiLMBlock(width, cond, p_drop) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(2 * width + in_ch, 96), nn.GELU(), nn.Dropout(p_drop), nn.Linear(96, 6))
        self.zmax_prior = zmax_prior
        # input standardisation constants (set from training data via set_norm)
        self.register_buffer("in_mean", torch.zeros(in_ch))
        self.register_buffer("in_std", torch.ones(in_ch))
        self.register_buffer("desc_mean", torch.zeros(20))
        self.register_buffer("desc_std", torch.ones(20))

    def features(self, r, m, rinf):
        x = [torch.log(torch.clamp(r, min=1e-4)), m]
        if self.use_pf:
            x.append(physics_features(r, rinf))
        return torch.cat(x, 1)

    def set_norm(self, x, desc):
        with torch.no_grad():
            self.in_mean.copy_(x.mean((0, 2, 3)))
            self.in_std.copy_(x.std((0, 2, 3)).clamp(min=1e-3))
            self.desc_mean.copy_(desc.mean(0))
            self.desc_std.copy_(desc.std(0).clamp(min=1e-2))

    def forward(self, r, m, rinf, desc):
        x = self.features(r, m, rinf)
        x = (x - self.in_mean[None, :, None, None]) / self.in_std[None, :, None, None]
        x = x * m  # zero non-water context
        c = None
        if self.use_film:
            d = torch.cat([desc, torch.log(torch.clamp(rinf, min=1e-4))], 1)
            c = self.scene((d - self.desc_mean) / self.desc_std)
        h = self.stem(x)
        for b in self.blocks:
            h = b(h, c)
        P = h.shape[-1] // 2
        centre = h[:, :, P - 1:P + 2, P - 1:P + 2].mean((2, 3))
        ctx = (h * m).sum((2, 3)) / m.sum((2, 3)).clamp(min=1)
        o = self.head(torch.cat([centre, ctx, x[:, :, P, P]], 1))
        mu = F.softplus(o[:, 0]) + 1e-3
        var = F.softplus(o[:, 1]) + 1e-3
        out = {"mu": mu, "var": var}
        if self.use_ph:
            k0 = 0.02 + 1.5 * torch.sigmoid(o[:, 2])          # non-water attenuation (1/m)
            albedo = 0.05 + 0.45 * torch.sigmoid(o[:, 3])      # bottom brightness
            mix = torch.sigmoid(o[:, 4])                       # sand fraction
            gain = 0.3 + 0.7 * torch.sigmoid(o[:, 5])          # water-air / radiometric gain
            rhat, kd, a = forward_reflectance(mu, k0, albedo, mix, rinf, gain)
            out.update(k0=k0, albedo=albedo, mix=mix, rhat=rhat, kd=kd,
                       zmax=detection_limit(kd, a, rinf))
        return out


def beta_nll(mu, var, y, beta=0.5):
    """Seitzer et al. (2022) beta-NLL."""
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    if beta > 0:
        nll = nll * var.detach() ** beta
    return nll.mean()


def physics_loss(out, r_centre):
    """Log-space reflectance reconstruction error for B,G,R at the centre pixel."""
    return F.smooth_l1_loss(torch.log(out["rhat"].clamp(min=1e-4)), torch.log(r_centre[:, :3].clamp(min=1e-4)),
                            beta=0.1)
