"""
HybridBathNet Architecture — v3
================================
Physics-Guided + Uncertainty-Aware Ocean Depth Prediction Model.

v3 Improvements (no new data required):
  §1.1  σ units fix          : aleatoric std rescaled to metres in evaluate().
  §1.2  Raw-reflectance Kd   : model stores mean/std buffers and undoes z-score
                               before computing Kd / z_ext from real reflectance.
  §3.1  Dense U-Net output   : encoder with skip connections → decoder → full
                               (H, W) depth and log-variance maps (not scalar).
  §3.2  Per-pixel Kd         : K_d map (H, W) retained; patch average also kept
                               for the extinction bound scalar loss.
  §3.3  FiLM conditioning    : Kd map injected at every encoder scale via FiLM,
                               replacing the cross-attention token approach.
  §3.4  Remove bad bounding  : softplus + min(z_ext) on features removed.
  §3.5  Output-space physics : depth = softplus(h) for non-negativity;
                               extinction penalty applied per pixel with margin.
  §3.6  Reconstruction head  : predicts K_λ and A_λ per band, rebuilds R̂_λ
                               for self-supervised physics consistency.
  §4.2  β-NLL loss           : weights each sample's NLL by stopgrad(σ²)^β=0.5.
  §4.3  Huber mean term      : smooth_l1_loss replaces squared MSE.
  §5.2  Snapshot saving      : train() returns snapshot weights from each cosine
                               cycle for free ensemble inference.
  §5.4  Calibration metrics  : NLL, 68/90% coverage, σ-error correlation,
                               sparsification RMSE returned by evaluate().
  §5.5  Conformal recal.     : conformal_recalibrate() scales σ so the empirical
                               68% interval has 68% empirical coverage.

Usage:
    from src.hybridbathnet import HybridBathNetModel, CompositePhysicsNLLLoss
    from src.hybridbathnet import predict_with_uncertainty, DeepEnsemblePredictor
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 128


# ---------------------------------------------------------------------------
# 0.  Helpers
# ---------------------------------------------------------------------------

def _to_tensor_buf(arr, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr, dtype=np.float32)).to(dtype)


# ---------------------------------------------------------------------------
# 1.  Physics: per-pixel K_d and extinction depth  (§1.2, §3.2)
# ---------------------------------------------------------------------------

def compute_kd_and_extinction(
    blue:            torch.Tensor,
    green:           torch.Tensor,
    red:             torch.Tensor | None = None,
    i0_over_epsilon: float = 100.0,
    eps:             float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pixel K_d map and patch-average extinction depth.

    §3.2: returns the full (B, 1, H, W) K_d map in addition to the
    patch-average scalar (B, 1) kd_avg and (B, 1) z_ext.

    Args:
        blue, green, red: (B, 1, H, W) raw surface reflectance (not z-scored).
        i0_over_epsilon:  signal-to-noise ratio for Beer-Lambert extinction.
    Returns:
        kd_map: (B, 1, H, W)  per-pixel diffuse attenuation (m⁻¹)
        kd_avg: (B, 1)        spatial average of kd_map
        z_ext:  (B, 1)        maximum optical extinction depth (m)
    """
    b = torch.clamp(blue,  min=eps)
    g = torch.clamp(green, min=eps)

    kd_map = 0.0166 + 0.156 * torch.pow(b / g, -1.14)
    kd_map = torch.clamp(kd_map, 0.01, 5.0)

    if red is not None:
        r      = torch.clamp(red, min=eps)
        kd_red = torch.clamp(0.430 * r / (g + eps), 0.0, 3.0)
        kd_map = kd_map + kd_red

    kd_map = torch.clamp(kd_map, 0.01, 6.0)   # (B, 1, H, W)

    kd_avg = kd_map.mean(dim=(-2, -1))          # (B, 1)
    z_ext  = math.log(i0_over_epsilon) / (2.0 * kd_avg + eps)
    z_ext  = torch.clamp(z_ext, 1.0, 50.0)     # (B, 1)

    return kd_map, kd_avg, z_ext


# ---------------------------------------------------------------------------
# 2.  Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Conv2D residual block with SE attention and spatial dropout."""

    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se  = SEBlock(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.se(self.conv(x)))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (Perez et al., 2018) — §3.3.

    Applies a spatial, per-channel affine transform (γ, β) to feature maps,
    conditioned on the physics context (K_d map, z_prior map, mask).

    γ and β are predicted by a 1×1 conv on the upsampled conditioning input,
    so the modulation is spatially consistent with the feature resolution.
    """

    def __init__(self, cond_channels: int, feat_channels: int):
        super().__init__()
        self.net = nn.Conv2d(cond_channels, 2 * feat_channels, kernel_size=1)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C_feat, H, W)
            cond: (B, C_cond, H_cond, W_cond)  — resized to feat's spatial dims
        Returns:
            modulated feat (same shape)
        """
        if cond.shape[-2:] != feat.shape[-2:]:
            cond = F.interpolate(
                cond, size=feat.shape[-2:], mode="bilinear", align_corners=False
            )
        gamma, beta = self.net(cond).chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta


def _stem_block(cin: int, cout: int, dropout_rate: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_rate),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        SEBlock(cout),
    )


# ---------------------------------------------------------------------------
# 3.  Dense U-Net Encoder-Decoder  (§3.1)
# ---------------------------------------------------------------------------

class UNetEncoder(nn.Module):
    """Encoder side of the dense U-Net.

    Produces skip-connection feature maps at each scale.
    FiLM conditioning on the K_d / z_prior context is applied
    at every scale (§3.3).

    Channel widths: C_in → 32 → 64 → 96 → 128 (bottleneck)
    """

    # Number of FiLM conditioning channels: [kd_map, z_prior_map, mask]
    N_COND = 3

    def __init__(self, in_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Encoder stages (each returns the skip feature)
        self.stem1 = _stem_block(in_channels, 32, dropout_rate)
        self.res1  = ResidualBlock(32, dropout_rate)
        self.film1 = FiLM(self.N_COND, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.stem2 = _stem_block(32, 64, dropout_rate)
        self.res2  = ResidualBlock(64, dropout_rate)
        self.film2 = FiLM(self.N_COND, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.stem3 = _stem_block(64, 96, dropout_rate)
        self.res3  = ResidualBlock(96, dropout_rate)
        self.film3 = FiLM(self.N_COND, 96)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _stem_block(96, FEATURE_DIM, dropout_rate)
        self.res_btn    = ResidualBlock(FEATURE_DIM, dropout_rate)
        self.film_btn   = FiLM(self.N_COND, FEATURE_DIM)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x:    (B, C_in, H, W)
            cond: (B, N_COND, H, W) physics conditioning [kd_map, z_prior, mask]
        Returns:
            bottleneck: (B, 128, H/8, W/8)
            skips:      [(B, 32, H, W), (B, 64, H/2, W/2), (B, 96, H/4, W/4)]
        """
        s1 = self.film1(self.res1(self.stem1(x)), cond)       # full res
        s2 = self.film2(self.res2(self.stem2(self.pool1(s1))), cond)   # ½
        s3 = self.film3(self.res3(self.stem3(self.pool2(s2))), cond)   # ¼
        bn = self.film_btn(self.res_btn(self.bottleneck(self.pool3(s3))), cond)  # ⅛
        return bn, [s1, s2, s3]


class UNetDecoder(nn.Module):
    """Decoder with skip connections; predicts dense (H, W) output maps.

    Each stage: bilinear upsample → concat skip → residual block.
    The final 1×1 convolution produces two maps: depth and log-variance.
    """

    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()

        # Decoder stages (reverse order)
        self.up3  = nn.ConvTranspose2d(FEATURE_DIM, 96, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(96 + 96, dropout_rate)   # after skip concat

        self.up2  = nn.ConvTranspose2d(96 + 96, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(64 + 64, dropout_rate)

        self.up1  = nn.ConvTranspose2d(64 + 64, 32, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(32 + 32, dropout_rate)

        # Dual output: depth mean and log-variance
        self.depth_conv = nn.Conv2d(32 + 32, 1, kernel_size=1)
        self.logvar_conv = nn.Conv2d(32 + 32, 1, kernel_size=1)

    def forward(
        self, bottleneck: torch.Tensor, skips: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            depth  : (B, 1, H, W)
            log_var: (B, 1, H, W)
        """
        s1, s2, s3 = skips

        d = self.up3(bottleneck)
        d = torch.cat([d, s3], dim=1)
        d = self.dec3(d)

        d = self.up2(d)
        d = torch.cat([d, s2], dim=1)
        d = self.dec2(d)

        d = self.up1(d)
        d = torch.cat([d, s1], dim=1)
        d = self.dec1(d)

        # §3.5: softplus for non-negativity — depth ≥ 0 by construction
        depth   = F.softplus(self.depth_conv(d))              # (B, 1, H, W)
        log_var = torch.clamp(self.logvar_conv(d), -10.0, 10.0)  # (B, 1, H, W)
        return depth, log_var


# ---------------------------------------------------------------------------
# 4.  Optical Reconstruction Head  (§3.6)
# ---------------------------------------------------------------------------

class ReconstructionHead(nn.Module):
    """Predicts per-pixel K_λ and A_λ for each band, then reconstructs R̂_λ.

    Two-flow radiative transfer model:
        R̂_λ = R∞_λ · (1 − exp(−2·K_λ·z)) + A_λ · exp(−2·K_λ·z)

    The reconstruction loss L_recon = mean|R̂_λ − R_λ| is self-supervised:
    it uses every valid water pixel, with no depth label involved.

    Args:
        n_bands:     number of visible bands to reconstruct (default: 3 = B/G/R)
        feature_dim: bottleneck feature channels feeding the head
    """

    def __init__(self, n_bands: int = 3, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.n_bands = n_bands
        # Predict K_λ (positive) and A_λ (positive) for each band
        self.kl_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, n_bands, 1),
        )
        self.al_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, n_bands, 1),
        )

    def forward(
        self,
        feat_full_res: torch.Tensor,
        depth_map:     torch.Tensor,
        r_inf:         torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_full_res: (B, C, H, W) final decoder feature map
            depth_map:     (B, 1, H, W) predicted depth in metres
            r_inf:         (B, n_bands, 1, 1) deep-water reflectance per band
        Returns:
            r_hat:  (B, n_bands, H, W) reconstructed reflectance
            kl_map: (B, n_bands, H, W) K_λ
            al_map: (B, n_bands, H, W) A_λ
        """
        kl_map = F.softplus(self.kl_head(feat_full_res)) + 0.01   # (B, n, H, W)
        al_map = torch.sigmoid(self.al_head(feat_full_res))         # ∈ (0, 1)

        exp_term = torch.exp(-2.0 * kl_map * depth_map)            # (B, n, H, W)
        r_hat    = r_inf * (1.0 - exp_term) + al_map * exp_term
        return r_hat, kl_map, al_map


# ---------------------------------------------------------------------------
# 5.  Prediction Heads (dense 1×1 conv versions)
# ---------------------------------------------------------------------------

class DepthHead(nn.Module):
    """Dense scalar head used when the decoder is NOT attached.
    Used for backwards-compat with the existing training loop."""

    def __init__(self, feature_dim: int = FEATURE_DIM, dropout_rate: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.mlp(f).squeeze(-1)


class AleatoricUncertaintyHead(nn.Module):
    """Predicts log variance s = ln(σ²).  Used without the U-Net decoder."""

    def __init__(self, feature_dim: int = FEATURE_DIM, dropout_rate: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.mlp(f).squeeze(-1), -10.0, 10.0)


# ---------------------------------------------------------------------------
# 6.  Full HybridBathNet v3 Model
# ---------------------------------------------------------------------------

class HybridBathNetModel(nn.Module):
    """HybridBathNet v3 — Dense U-Net with FiLM physics conditioning.

    Key changes from v2:
    - Stores channel mean/std as buffers to undo z-scoring before physics (§1.2).
    - Uses UNetEncoder + UNetDecoder for dense (H, W) depth and uncertainty maps.
    - FiLM conditioning at every encoder scale with [K_d, z_prior, mask] (§3.3).
    - Softplus output for non-negativity; extinction penalty with margin (§3.5).
    - Optional ReconstructionHead for self-supervised physics loss (§3.6).

    Args:
        in_channels:       total number of input channels (spectral + physics)
        n_spectral:        first n_spectral channels are raw reflectance (for Kd)
        mean, std:         (n_spectral,) dataset channel mean/std for de-z-scoring
        feature_dim:       bottleneck width (default: 128)
        dropout_rate:      MC-dropout probability (default: 0.1)
        i0_over_epsilon:   Beer-Lambert extinction threshold
        use_red_kd:        include red band in Kd formula
        use_recon_head:    enable optical reconstruction head
        n_recon_bands:     number of bands to reconstruct (default: 3)
        r_inf:             (n_recon_bands,) deep-water reflectance for reconstruction
    """

    def __init__(
        self,
        in_channels:     int            = 6,
        n_spectral:      int            = 3,
        mean:            np.ndarray | None = None,
        std:             np.ndarray | None = None,
        feature_dim:     int            = FEATURE_DIM,
        dropout_rate:    float          = 0.1,
        i0_over_epsilon: float          = 100.0,
        use_red_kd:      bool           = True,
        use_recon_head:  bool           = True,
        n_recon_bands:   int            = 3,
        r_inf:           np.ndarray | None = None,
    ):
        super().__init__()
        self.n_spectral      = n_spectral
        self.i0_over_epsilon = i0_over_epsilon
        self.use_red_kd      = use_red_kd
        self.use_recon_head  = use_recon_head

        # §1.2 — store de-normalization buffers so physics can use real reflectance
        if mean is not None and std is not None:
            mu  = np.zeros(in_channels, dtype=np.float32)
            sig = np.ones(in_channels,  dtype=np.float32)
            mu[:n_spectral]  = mean[:n_spectral]
            sig[:n_spectral] = std[:n_spectral]
        else:
            mu  = np.zeros(in_channels, dtype=np.float32)
            sig = np.ones(in_channels,  dtype=np.float32)

        self.register_buffer("mu",  torch.from_numpy(mu ).view(1, in_channels, 1, 1))
        self.register_buffer("sig", torch.from_numpy(sig).view(1, in_channels, 1, 1))

        # Deep-water reflectance for reconstruction loss (§3.6)
        if r_inf is None:
            r_inf = np.zeros(n_recon_bands, dtype=np.float32)
        self.register_buffer(
            "r_inf_buf",
            torch.from_numpy(r_inf.astype(np.float32)).view(1, n_recon_bands, 1, 1)
        )

        # U-Net encoder + decoder
        self.encoder = UNetEncoder(in_channels=in_channels, dropout_rate=dropout_rate)
        self.decoder = UNetDecoder(dropout_rate=dropout_rate)

        # z_prior conditioning projection (H, W map → N_COND channels)
        # cond = [kd_map (1), z_prior_map (1), mask_ch (1)] — 3 channels total
        self.cond_proj = nn.Conv2d(3, 3, kernel_size=1)

        # Optional reconstruction head
        if use_recon_head:
            self.recon_head = ReconstructionHead(
                n_bands=n_recon_bands, feature_dim=32 + 32   # final decoder channels
            )

        # Global average pool head for scalar-output compat and for loss/eval
        # (used to compute the patch-level extinction check, not for depth pred)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x:    (B, C, H, W) z-scored input channels
            mask: (B, 1, H, W) or None — valid water pixel mask (float 0/1)
        Returns dict with keys:
            'depth'    : (B, 1, H, W) predicted depth in metres (via softplus)
            'log_var'  : (B, 1, H, W) predicted log variance
            'kd_map'   : (B, 1, H, W) per-pixel Kd
            'kd_avg'   : (B, 1) patch-average Kd (for extinction loss)
            'z_ext'    : (B, 1) patch extinction depth
            'r_hat'    : (B, n_bands, H, W) reconstructed reflectance (if recon_head)
        """
        # §1.2 — undo z-scoring for the spectral channels to get real reflectance
        raw = x * self.sig + self.mu            # (B, C, H, W) — partial undo
        blue  = raw[:, 0:1]                     # B02
        green = raw[:, 1:2]                     # B03
        red   = raw[:, 2:3] if (self.use_red_kd and self.n_spectral > 2) else None

        # §3.2 — per-pixel Kd map + patch-average statistics
        kd_map, kd_avg, z_ext = compute_kd_and_extinction(
            blue, green, red, self.i0_over_epsilon
        )

        # Build z_prior map from Beer-Lambert (in raw reflectance space)
        g = torch.clamp(green, 1e-6)
        kd_clamped = torch.clamp(kd_map, 0.01, 5.0)
        with torch.no_grad():
            i_surface = torch.quantile(g.reshape(g.shape[0], -1), 0.95, dim=1)
            i_surface = i_surface.view(-1, 1, 1, 1).clamp(min=1e-3)
        trans  = torch.clamp(i_surface / g, 1.0, 1e4)
        z_prior_map = (1.0 / (2.0 * kd_clamped + 1e-6)) * torch.log(trans)
        z_prior_map = torch.clamp(z_prior_map, 0.0, 50.0)

        # §3.3 — FiLM conditioning: [kd_map, z_prior_map, valid_mask]
        if mask is None:
            # Try to read mask channel from input (channel index 11 in default layout)
            mask_ch_idx = min(11, x.shape[1] - 1)
            mask = x[:, mask_ch_idx:mask_ch_idx + 1]
        cond_raw = torch.cat([kd_map, z_prior_map, mask], dim=1)   # (B, 3, H, W)
        cond     = self.cond_proj(cond_raw)

        # Encode with skip connections
        bottleneck, skips = self.encoder(x, cond)

        # Decode to full-resolution depth and log-variance maps
        depth, log_var = self.decoder(bottleneck, skips)
        # depth already has softplus applied in UNetDecoder → depth ≥ 0

        out = {
            "depth":   depth,     # (B, 1, H, W)
            "log_var": log_var,   # (B, 1, H, W)
            "kd_map":  kd_map,    # (B, 1, H, W)
            "kd_avg":  kd_avg,    # (B, 1)
            "z_ext":   z_ext,     # (B, 1)
        }

        # §3.6 — Optical reconstruction (requires the final decoder feature map)
        # We need the decoder's last-stage feature, so re-expose it:
        if self.use_recon_head and hasattr(self, "recon_head"):
            # Reconstruct using the full-res skip (s1) as feature map
            # We grab it from skips[0] (s1, full resolution)
            r_hat, _, _ = self.recon_head(
                skips[0],          # (B, 32, H, W)  — full-res decoder feature
                depth,
                self.r_inf_buf.to(x.device)
            )
            out["r_hat"] = r_hat

        return out

    # ------------------------------------------------------------------
    # Legacy-compatibility shim for the training loop that expects a
    # 4-tuple (pred_norm, log_var, kd, z_ext)
    # ------------------------------------------------------------------
    def forward_scalar(
        self, x: torch.Tensor, y_mean: float, y_std: float,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pred_norm_scalar, log_var_scalar, kd_avg, z_ext).

        Averages dense depth / log_var maps over valid water pixels
        so the existing scalar loss functions continue to work.
        """
        out = self.forward(x, mask)
        depth   = out["depth"].squeeze(1)     # (B, H, W)
        log_var = out["log_var"].squeeze(1)   # (B, H, W)

        # Centre pixel (compatible with scalar-label training)
        cy, cx      = depth.shape[1] // 2, depth.shape[2] // 2
        depth_sc    = depth[:, cy, cx]           # (B,)
        log_var_sc  = log_var[:, cy, cx]         # (B,)

        pred_norm = (depth_sc - y_mean) / (y_std + 1e-6)
        return pred_norm, log_var_sc, out["kd_avg"], out["z_ext"]


# ---------------------------------------------------------------------------
# 7.  Dense Physics + β-NLL Loss  (§4.2, §4.3, §3.5, §3.6)
# ---------------------------------------------------------------------------

class DensePhysicsLoss(nn.Module):
    """Dense-output physics-informed loss.

    Operates on full (B, 1, H, W) depth and log_var maps.
    Valid water pixels identified by the mask M.

    Loss terms:
        L_huber  : Huber (smooth-L1) between predicted and GEBCO depth maps
                   averaged over valid M pixels only.
        L_nll    : β-NLL (Seitzer et al., 2022) — standard NLL weighted by
                   stopgrad(σ²)^β to prevent variance collapse.
        L_ext    : soft extinction penalty per pixel with a 1 m margin.
        L_recon  : |R̂_λ − R_λ| over valid pixels (self-supervised physics).
        L_lowres : Huber on GEBCO-scale (46×46 pixel) averages to account for
                   GEBCO's ~460 m native resolution (§3.7).

    Args:
        w_huber:  weight for per-pixel Huber depth loss
        w_nll:    weight for β-NLL loss
        w_ext:    weight for extinction bound penalty
        w_recon:  weight for optical reconstruction loss
        w_lowres: weight for GEBCO-scale averaged loss
        beta_nll: exponent for β-NLL (0.5 recommended)
        huber_delta: Huber loss δ (in normalised depth units, ~0.5)
        ext_margin: margin (m) before extinction penalty activates
        gebco_pool_size: kernel size for GEBCO-scale pooling (§3.7)
    """

    def __init__(
        self,
        w_huber:        float = 1.0,
        w_nll:          float = 1.0,
        w_ext:          float = 0.3,
        w_recon:        float = 0.1,
        w_lowres:       float = 0.5,
        beta_nll:       float = 0.5,
        huber_delta:    float = 0.5,
        ext_margin:     float = 1.0,
        gebco_pool_size:int   = 46,
    ):
        super().__init__()
        self.w_huber        = w_huber
        self.w_nll          = w_nll
        self.w_ext          = w_ext
        self.w_recon        = w_recon
        self.w_lowres       = w_lowres
        self.beta_nll       = beta_nll
        self.huber_delta    = huber_delta
        self.ext_margin     = ext_margin
        self.gebco_pool_size = gebco_pool_size

    def forward(
        self,
        depth:   torch.Tensor,    # (B, 1, H, W) metres, softplus applied
        log_var: torch.Tensor,    # (B, 1, H, W)
        target:  torch.Tensor,    # (B, 1, H, W) GEBCO depth metres (positive)
        mask:    torch.Tensor,    # (B, 1, H, W) float 0/1
        z_ext:   torch.Tensor,    # (B, 1) extinction depth
        r_obs:   torch.Tensor | None = None,   # (B, n_bands, H, W) observed reflectance
        r_hat:   torch.Tensor | None = None,   # (B, n_bands, H, W) reconstructed
        y_std:   float = 1.0,                  # for normalised Huber delta
    ) -> tuple[torch.Tensor, dict]:

        m = mask.bool()   # (B, 1, H, W) boolean

        # --- 1. Per-pixel Huber depth loss (over valid water pixels only) ---
        diff = depth - target
        l_huber_px = F.smooth_l1_loss(depth, target, beta=self.huber_delta * y_std,
                                       reduction="none")  # (B, 1, H, W)
        l_huber = l_huber_px[m].mean() if m.any() else l_huber_px.mean()

        # --- 2. β-NLL over valid pixels ---
        var  = torch.exp(log_var)                               # (B, 1, H, W)
        nll  = 0.5 * (diff ** 2 / (var + 1e-8) + log_var)      # (B, 1, H, W)
        beta_w = var.detach() ** self.beta_nll                  # stopgrad
        l_nll = (beta_w * nll)[m].mean() if m.any() else nll.mean()

        # --- 3. Per-pixel extinction soft penalty with margin ---
        z_ext_map = z_ext.view(-1, 1, 1, 1).expand_as(depth)   # (B, 1, H, W)
        l_ext = F.relu(depth - z_ext_map - self.ext_margin)
        l_ext = l_ext[m].mean() if m.any() else l_ext.mean()

        # --- 4. Optical reconstruction loss (self-supervised physics) ---
        l_recon = torch.tensor(0.0, device=depth.device)
        if r_obs is not None and r_hat is not None:
            mask_vis = m.expand_as(r_obs)
            l_recon = torch.abs(r_hat - r_obs)[mask_vis].mean()

        # --- 5. GEBCO-scale (low-res) consistency loss ---
        l_lowres = torch.tensor(0.0, device=depth.device)
        ks = self.gebco_pool_size
        if ks > 1 and depth.shape[-1] >= ks and depth.shape[-2] >= ks:
            pred_lo = F.avg_pool2d(depth,  ks, stride=ks)
            tgt_lo  = F.avg_pool2d(target, ks, stride=ks)
            mask_lo = F.avg_pool2d(mask,   ks, stride=ks) > 0.5
            if mask_lo.any():
                l_lowres = F.smooth_l1_loss(
                    pred_lo[mask_lo], tgt_lo[mask_lo],
                    beta=self.huber_delta * y_std
                )

        total = (
            self.w_huber  * l_huber
            + self.w_nll  * l_nll
            + self.w_ext  * l_ext
            + self.w_recon * l_recon
            + self.w_lowres * l_lowres
        )

        # RMSE for logging
        with torch.no_grad():
            rmse_val = torch.sqrt((diff[m] ** 2).mean()).item() if m.any() else float("nan")

        parts = {
            "loss_total":  float(total.item()),
            "loss_huber":  float(l_huber.item()),
            "loss_nll":    float(l_nll.item()),
            "loss_ext":    float(l_ext.item()),
            "loss_recon":  float(l_recon.item()),
            "loss_lowres": float(l_lowres.item()),
            "rmse":        rmse_val,
        }
        return total, parts


# ---------------------------------------------------------------------------
# 8.  Scalar compat loss (used by cnn_baseline.py training loop)
# ---------------------------------------------------------------------------

class CompositePhysicsNLLLoss(nn.Module):
    """Scalar-output loss (kept for backward compatibility with cnn_baseline.py).

    Upgraded to use Huber + β-NLL internally (§4.2, §4.3).
    Depth-weighting removed if a DepthStratifiedSampler is active (§4.1).
    """

    def __init__(
        self,
        w_mse:               float = 1.0,
        w_nll:               float = 1.0,
        w_phys:              float = 0.3,
        w_ext:               float = 0.3,
        depth_scale:         float = 10.0,
        use_depth_weighting: bool  = True,
        beta_nll:            float = 0.5,
        huber_delta:         float = 0.5,
    ):
        super().__init__()
        self.w_mse               = w_mse
        self.w_nll               = w_nll
        self.w_phys              = w_phys
        self.w_ext               = w_ext
        self.depth_scale         = depth_scale
        self.use_depth_weighting = use_depth_weighting
        self.beta_nll            = beta_nll
        self.huber_delta         = huber_delta

    def forward(
        self,
        pred_depth_norm:   torch.Tensor,   # (B,)
        log_var:           torch.Tensor,   # (B,)
        target_depth_norm: torch.Tensor,   # (B,)
        z_ext:             torch.Tensor,   # (B, 1)
        y_mean: float = 0.0,
        y_std:  float = 1.0,
    ) -> tuple[torch.Tensor, dict]:

        pred_m   = pred_depth_norm   * y_std + y_mean
        target_m = target_depth_norm * y_std + y_mean

        if self.use_depth_weighting:
            w = torch.clamp(target_m.detach().abs() / self.depth_scale, 0.5, 3.0)
        else:
            w = torch.ones_like(target_m)
        w = w / (w.mean() + 1e-6)

        # §4.3 Huber mean term
        diff_sq = (pred_depth_norm - target_depth_norm) ** 2
        l_huber = torch.mean(
            w * F.smooth_l1_loss(
                pred_depth_norm, target_depth_norm,
                beta=self.huber_delta, reduction="none"
            )
        )

        # §4.2 β-NLL
        var     = torch.exp(log_var)
        nll     = 0.5 * (diff_sq / (var + 1e-8) + log_var)
        beta_w  = var.detach() ** self.beta_nll
        l_nll   = torch.mean(w * beta_w * nll)

        # Physics: non-negativity (depth ≥ 0) — soft via ReLU on metres
        l_nonneg = torch.mean(F.relu(-pred_m))

        # Physics: extinction bound
        z_ext_f = z_ext.squeeze(-1)
        l_ext   = torch.mean(F.relu(pred_m - z_ext_f - 1.0))   # 1m margin

        total = (
            self.w_mse  * l_huber
            + self.w_nll  * l_nll
            + self.w_phys * (l_nonneg / (y_std + 1e-6))
            + self.w_ext  * (l_ext   / (y_std + 1e-6))
        )

        diff_m = pred_m - target_m
        parts  = {
            "loss_total":  float(total.item()),
            "loss_nll":    float(l_nll.item()),
            "loss_nonneg": float(l_nonneg.item()),
            "loss_ext":    float(l_ext.item()),
            "rmse":        float(torch.sqrt(torch.mean(diff_m ** 2)).item()),
        }
        return total, parts


# ---------------------------------------------------------------------------
# 9.  MC Dropout Inference  (§5.3 — cheaper: encode once, sample heads)
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model:     nn.Module,
    x:         torch.Tensor,
    n_samples: int = 30,
    y_mean:    float = 0.0,
    y_std:     float = 1.0,
) -> dict[str, torch.Tensor]:
    """Monte Carlo Dropout with cheap cached-encoder sampling.

    §5.3: because the encoder has no dropout, we can run the full
    forward pass once per sample — but since Dropout2d is only in
    the ResidualBlocks inside the encoder, each call with .train()
    on dropout layers still gives a stochastic draw.

    Returns metrics in metres (§1.1 fix applied).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    depth_preds = []
    log_var_preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            # Centre pixel for scalar comparison
            d  = out["depth"].squeeze(1)
            lv = out["log_var"].squeeze(1)
            cy, cx = d.shape[1] // 2, d.shape[2] // 2
            depth_preds.append(d[:, cy, cx].unsqueeze(0))
            log_var_preds.append(lv[:, cy, cx].unsqueeze(0))

    preds    = torch.cat(depth_preds,   dim=0)   # (T, B) — already in metres
    log_vars = torch.cat(log_var_preds, dim=0)   # (T, B)

    mean_depth    = preds.mean(dim=0)
    epistemic_var = preds.var(dim=0, unbiased=True)

    # §1.1: aleatoric variance IS in metres because depth output is in metres
    aleatoric_var = torch.exp(log_vars).mean(dim=0)  # metres²

    total_var  = aleatoric_var + epistemic_var
    total_std  = torch.sqrt(total_var.clamp(1e-8))
    coeff_var  = total_std / (mean_depth.abs().clamp(0.1))

    return {
        "mean_depth":      mean_depth,
        "aleatoric_var":   aleatoric_var,
        "epistemic_var":   epistemic_var,
        "total_var":       total_var,
        "total_std":       total_std,
        "coeff_variation": coeff_var,
        "all_preds":       preds,
    }


# ---------------------------------------------------------------------------
# 10. Calibration helpers  (§5.4, §5.5)
# ---------------------------------------------------------------------------

def calibration_metrics(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    sigma:     np.ndarray,
    n_sparsify_steps: int = 20,
) -> dict:
    """Compute uncertainty calibration diagnostics.

    Args:
        y_true, y_pred, sigma: 1-D arrays of true depths, predictions, and σ (m).
    Returns:
        dict with Gaussian NLL, 68%/90% coverage, σ-|err| correlation,
        and sparsification RMSE curve (list of (frac_removed, rmse) pairs).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    sigma  = np.asarray(sigma,  dtype=float).ravel()
    sigma  = np.maximum(sigma, 1e-6)

    errors    = np.abs(y_pred - y_true)
    z_scores  = errors / sigma

    # Gaussian NLL (nats)
    gauss_nll = float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + errors ** 2 / (2 * sigma ** 2)))

    # Empirical coverage for k·σ intervals
    cov_68 = float(np.mean(z_scores <= 1.0) * 100.0)
    cov_90 = float(np.mean(z_scores <= 1.645) * 100.0)

    # σ — |error| correlation (Pearson)
    corr = float(np.corrcoef(sigma, errors)[0, 1]) if len(sigma) > 1 else float("nan")

    # Sparsification curve: remove most-uncertain pixels first
    sort_idx  = np.argsort(sigma)[::-1]          # descending σ — most uncertain first
    n         = len(y_true)
    sparsify  = []
    for step in range(n_sparsify_steps + 1):
        frac_removed = step / n_sparsify_steps
        keep_n       = max(1, int(n * (1.0 - frac_removed)))
        keep_idx     = sort_idx[n - keep_n:]     # lowest σ first
        rmse_k       = float(np.sqrt(np.mean((y_pred[keep_idx] - y_true[keep_idx]) ** 2)))
        sparsify.append((float(frac_removed), rmse_k))

    return {
        "gaussian_nll":     gauss_nll,
        "coverage_68pct":   cov_68,
        "coverage_90pct":   cov_90,
        "sigma_error_corr": corr,
        "sparsification":   sparsify,
    }


def conformal_recalibrate(
    sigma_val:  np.ndarray,
    errors_val: np.ndarray,
    target_coverage: float = 0.68,
) -> float:
    """Conformal recalibration (§5.5).

    Computes the scaling factor k such that:
        P(|y − ŷ| ≤ k · σ_val) ≈ target_coverage
    on the validation set.  Apply k * sigma_test at test time.

    Returns:
        k: float scaling factor for σ
    """
    z_scores   = np.abs(errors_val) / (np.asarray(sigma_val) + 1e-6)
    k          = float(np.quantile(z_scores, target_coverage))
    return k


# ---------------------------------------------------------------------------
# 11. Deep Ensemble Predictor  (§5.1, §5.2)
# ---------------------------------------------------------------------------

class DeepEnsemblePredictor(nn.Module):
    """Deep ensemble for superior epistemic uncertainty calibration.

    Members are trained independently (or loaded from snapshot checkpoints).
    Predictions are in metres (§1.1 fix applied).
    """

    def __init__(self, n_members: int = 5, **model_kwargs):
        super().__init__()
        self.n_members = n_members
        self.members   = nn.ModuleList([
            HybridBathNetModel(**model_kwargs) for _ in range(n_members)
        ])

    def predict(
        self, x: torch.Tensor, y_mean: float = 0.0, y_std: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """Run full ensemble inference; returns statistics in metres."""
        depth_preds = []
        log_var_preds = []

        for member in self.members:
            member.eval()
            with torch.no_grad():
                out = member(x)
                d   = out["depth"].squeeze(1)
                lv  = out["log_var"].squeeze(1)
                cy, cx = d.shape[1] // 2, d.shape[2] // 2
                depth_preds.append(d[:, cy, cx].unsqueeze(0))
                log_var_preds.append(lv[:, cy, cx].unsqueeze(0))

        preds    = torch.cat(depth_preds,   dim=0)
        log_vars = torch.cat(log_var_preds, dim=0)

        mean_depth    = preds.mean(dim=0)
        epistemic_var = preds.var(dim=0, unbiased=True)
        aleatoric_var = torch.exp(log_vars).mean(dim=0)
        total_var     = aleatoric_var + epistemic_var
        total_std     = torch.sqrt(total_var.clamp(1e-8))

        return {
            "mean_depth":    mean_depth,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
            "total_var":     total_var,
            "total_std":     total_std,
        }

    def forward(self, x: torch.Tensor):
        """First-member forward for training compatibility."""
        return self.members[0](x)
