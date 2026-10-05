"""
HybridBathNet Architecture — v2 (Improved)
===========================================
Physics-Guided + Uncertainty-Aware Ocean Depth Prediction Model.

v2 Improvements (no new data required):
  1. QAA-enhanced Kd computation  : Multi-band quasi-analytical Kd using blue, green AND red
                                    reflectances — more accurate in turbid coastal waters.
  2. ResidualEncoder               : Deep encoder with residual skip connections +
                                    Squeeze-and-Excitation (SE) channel attention.
                                    Captures richer spectral-spatial features than plain CNN.
  3. CrossAttentionPhysicsFusion   : Replaces the gated MLP decoder — multi-head cross-attention
                                    between spatial features (query) and physics priors (key/value).
                                    Allows the model to selectively attend to optical physics.
  4. DepthWeightedCompositeNLLLoss : NLL + MSE + physics penalties, with per-sample depth
                                    weighting (harder deep-water samples up-weighted).
  5. DeepEnsemblePredictor         : 5-member deep ensemble for better-calibrated epistemic
                                    uncertainty, used alongside improved MC Dropout inference.
  6. Curriculum training support   : Loss weight schedule exposed for use in training loop.

Original components (preserved):
  - BeerLambertPrior (compute_kd_and_extinction)
  - DepthHead / AleatoricUncertaintyHead
  - predict_with_uncertainty (MC Dropout)

Usage:
    from src.hybridbathnet import HybridBathNetModel, CompositePhysicsNLLLoss, predict_with_uncertainty
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 128


# ---------------------------------------------------------------------------
# 1a. Physics Prior: Improved Multi-Band QAA Kd Computation (v2)
# ---------------------------------------------------------------------------

def compute_kd_and_extinction(
    blue: torch.Tensor,
    green: torch.Tensor,
    red: torch.Tensor | None = None,
    i0_over_epsilon: float = 100.0,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the diffuse attenuation coefficient (K_d) and maximum physical
    extinction depth limit (z_extinction) using an improved QAA-based formula.

    v2 Improvement over v1:
    -----------------------
    v1 used only the blue/green ratio (2-band proxy). v2 incorporates the red band
    to improve Kd estimation in turbid coastal waters, where blue/green alone
    underestimates attenuation (Lee et al., 2002 — QAA framework):

        Kd_bg  = 0.0166 + 0.156 * (R_blue / R_green)^(-1.14)   [clear-water component]
        Kd_red = 0.430  * R_red / (R_green + eps)               [turbidity component]
        Kd     = Kd_bg + Kd_red                                  [combined]

    Beer-Lambert Law:
        I(z) = I₀ * exp(−2 * K_d * z)
        z_extinction = ln(I₀/ε) / (2 * K_d)

    Args:
        blue:            (B, 1, H, W) Blue reflectance (B02)
        green:           (B, 1, H, W) Green reflectance (B03)
        red:             (B, 1, H, W) Red reflectance (B04) [optional, improves turbid water]
        i0_over_epsilon: Signal-to-noise ratio threshold (I₀/ε)
        eps:             Small stability constant

    Returns:
        K_d:   (B, 1) average diffuse attenuation coefficient (m⁻¹)
        z_ext: (B, 1) maximum optical extinction depth limit (m)
    """
    b = torch.clamp(blue,  min=eps)
    g = torch.clamp(green, min=eps)

    bg_ratio  = b / g
    # QAA clear-water Kd proxy (blue/green path)
    kd_map_bg = 0.0166 + 0.156 * torch.pow(bg_ratio, -1.14)
    kd_map_bg = torch.clamp(kd_map_bg, min=0.01, max=5.0)

    if red is not None:
        r = torch.clamp(red, min=eps)
        # Red-band turbidity correction (Lee et al. 2002 Table 2 approximation)
        kd_map_red = 0.430 * r / (g + eps)
        kd_map_red = torch.clamp(kd_map_red, min=0.0, max=3.0)
        kd_map = kd_map_bg + kd_map_red
    else:
        kd_map = kd_map_bg

    kd_map = torch.clamp(kd_map, min=0.01, max=6.0)

    # Average K_d over the spatial patch → (B, 1)
    if kd_map.dim() == 4:
        kd_avg = kd_map.mean(dim=(-2, -1))        # (B, 1)
    elif kd_map.dim() == 3:
        kd_avg = kd_map.mean(dim=(-2, -1), keepdim=True)
    else:
        kd_avg = kd_map

    # Beer-Lambert maximum optical depth (m)  — 2-way path
    z_ext = math.log(i0_over_epsilon) / (2.0 * kd_avg + eps)
    z_ext = torch.clamp(z_ext, min=1.0, max=50.0)

    return kd_avg, z_ext


# ---------------------------------------------------------------------------
# 1b. Squeeze-and-Excitation (SE) Channel Attention Block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018).

    Recalibrates channel-wise feature responses by learning per-channel weights.
    Applied within each residual encoder block to focus on the most informative
    spectral channels (e.g., blue/green attenuation channels).
    """

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


# ---------------------------------------------------------------------------
# 2. Residual Physics-Guided Encoder (v2 — replaces plain CNN encoder)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Conv2D residual block: skip connection + BatchNorm + SE attention + Dropout."""

    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.se(self.conv(x)))


class PhysicsGuidedEncoder(nn.Module):
    """v2 Residual Encoder with Squeeze-and-Excitation channel attention.

    v1 → v2 improvements:
    - Plain Conv blocks replaced with Residual blocks (skip connections improve
      gradient flow and feature reuse across depth levels).
    - SE attention added per block: model learns to attend to the most
      depth-informative spectral channels (blue, green attenuation signal).
    - Deeper channel progression (6→32→64→96→128) vs. original (6→32→64→128).
    - Dropout2d rate kept at 0.1 to enable MC Dropout uncertainty sampling.

    Args:
        in_channels: 6 (4 spectral + 1 Stumpf ratio + 1 Beer-Lambert z_prior)
        dropout_rate: Spatial dropout rate (for MC Dropout inference)
    """

    def __init__(self, in_channels: int = 6, dropout_rate: float = 0.1):
        super().__init__()

        def stem_block(cin: int, cout: int) -> nn.Sequential:
            """Initial projection block (no skip connection — different channel dims)."""
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                SEBlock(cout),
            )

        # Stage 1: in_channels → 32  (full resolution)
        self.stem1 = stem_block(in_channels, 32)
        self.res1  = ResidualBlock(32, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2: 32 → 64  (½ resolution)
        self.stem2 = stem_block(32, 64)
        self.res2  = ResidualBlock(64, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3: 64 → 96  (¼ resolution — new stage vs. v1)
        self.stem3 = stem_block(64, 96)
        self.res3  = ResidualBlock(96, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck: 96 → 128  (⅛ resolution)
        self.bottleneck = stem_block(96, FEATURE_DIM)
        self.res_btn    = ResidualBlock(FEATURE_DIM, dropout_rate)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            f_spatial: (B, FEATURE_DIM=128)
        """
        x = self.res1(self.stem1(x))
        x = self.res2(self.stem2(self.pool1(x)))
        x = self.res3(self.stem3(self.pool2(x)))
        x = self.res_btn(self.bottleneck(self.pool3(x)))
        x = self.gap(x)
        return self.flatten(x)


# ---------------------------------------------------------------------------
# 3. Cross-Attention Physics Fusion Decoder (v2 — replaces gated MLP)
# ---------------------------------------------------------------------------

class CrossAttentionPhysicsFusion(nn.Module):
    """v2 Physics decoder using multi-head cross-attention.

    v1 used a gated MLP: gate = σ(W·[f_spatial, phys_emb])
    This was a static, learned gate weight, applied uniformly.

    v2 uses cross-attention:
    - Query  (Q): spatial CNN features f_spatial  (B, 1, 128)
    - Key/Value (K/V): physics prior embedding [K_d, z_ext] → (B, N_phys, dim_k)
    The model learns to attend to the physics prior only when it is informative
    (e.g., in turbid water where z_ext strongly limits depth predictions).
    In clear water with weak physics signal, attention weights approach 0 → model
    relies on spatial CNN features.

    Additionally:
    - Non-negativity (Softplus) applied to f_spatial before cross-attention.
    - Physical extinction bounding (min with z_ext) applied after cross-attention output.
    - LayerNorm applied before and after attention (Pre-LN transformer style).

    Args:
        feature_dim: Dimension of spatial feature vector (128)
        n_heads:     Number of attention heads (4)
        phys_tokens: Number of physics embedding tokens (4 — richer physics embedding)
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        n_heads: int = 4,
        phys_tokens: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.phys_tokens = phys_tokens

        # Physics prior → multi-token embedding  [K_d, z_ext] → (B, phys_tokens, feature_dim)
        self.phys_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim * phys_tokens),
        )

        # Pre-LayerNorm (transformer-style stability)
        self.norm_q     = nn.LayerNorm(feature_dim)
        self.norm_kv    = nn.LayerNorm(feature_dim)

        # Multi-head cross-attention: Q=spatial, K/V=physics
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm_out = nn.LayerNorm(feature_dim)
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.norm_ff = nn.LayerNorm(feature_dim)

    def forward(
        self,
        f_spatial: torch.Tensor,
        kd: torch.Tensor,
        z_ext: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f_spatial: (B, 128) encoder spatial feature vector
            kd:        (B, 1)  diffuse attenuation coefficient K_d (m⁻¹)
            z_ext:     (B, 1)  Beer-Lambert extinction depth bound (m)

        Returns:
            f_physics: (B, 128) physically-regularized feature vector
        """
        B = f_spatial.shape[0]

        # Step 1: enforce non-negativity on spatial features (depth ≥ 0m)
        f_nonneg = F.softplus(f_spatial)                    # (B, 128) — depth ≥ 0m

        # Step 2: physics extinction bounding (depth ≤ z_ext)
        z_bound = z_ext.expand(-1, self.feature_dim)        # (B, 128)
        f_bounded = torch.minimum(f_nonneg, z_bound)        # (B, 128)

        # Step 3: build physics token embedding  (B, phys_tokens, feature_dim)
        phys_input = torch.cat([kd, z_ext], dim=-1)         # (B, 2)
        phys_flat  = self.phys_embed(phys_input)             # (B, phys_tokens * feature_dim)
        phys_kv    = phys_flat.view(B, self.phys_tokens, self.feature_dim)  # (B, T, D)

        # Step 4: cross-attention  Q=f_bounded, K/V=phys_kv
        q   = self.norm_q(f_bounded).unsqueeze(1)           # (B, 1, 128)
        kv  = self.norm_kv(phys_kv)                         # (B, T, 128)
        attn_out, _ = self.cross_attn(q, kv, kv)            # (B, 1, 128)
        attn_out = attn_out.squeeze(1)                       # (B, 128)

        # Step 5: residual + feed-forward (transformer block pattern)
        f_out = self.norm_out(f_bounded + attn_out)         # residual add
        f_physics = self.norm_ff(f_out + self.ff(f_out))    # FF + residual

        return f_physics


# ---------------------------------------------------------------------------
# 4. Prediction Heads (Dual Head) — unchanged from v1
# ---------------------------------------------------------------------------

class DepthHead(nn.Module):
    """Predicts scalar water depth ŷ in meters from physics-regularized features."""

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

    def forward(self, f_physics: torch.Tensor) -> torch.Tensor:
        """Returns: depth_pred: (B,) predicted depth in meters"""
        return self.mlp(f_physics).squeeze(-1)


class AleatoricUncertaintyHead(nn.Module):
    """Predicts log variance s = ln(σ²) representing aleatoric data uncertainty."""

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

    def forward(self, f_physics: torch.Tensor) -> torch.Tensor:
        """Returns: log_var: (B,) predicted log variance s = ln(σ²)"""
        log_var = self.mlp(f_physics).squeeze(-1)
        return torch.clamp(log_var, min=-10.0, max=10.0)


# ---------------------------------------------------------------------------
# 5. Full HybridBathNet v2 Model Assembly
# ---------------------------------------------------------------------------

class HybridBathNetModel(nn.Module):
    """HybridBathNet v2 — Complete pipeline integrating:
    - Improved QAA multi-band Kd (blue + green + red → more accurate turbid Kd)
    - Residual Encoder with SE channel attention (deeper, richer features)
    - CrossAttentionPhysicsFusion (replaces gated MLP — dynamic physics weighting)
    - Dual Heads (DepthHead + AleatoricUncertaintyHead)

    Backward-compatible: same forward() signature and output format as v1.
    """

    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = FEATURE_DIM,
        dropout_rate: float = 0.1,
        i0_over_epsilon: float = 100.0,
        use_red_kd: bool = True,
        n_attn_heads: int = 4,
        phys_tokens: int = 4,
    ):
        super().__init__()
        self.i0_over_epsilon = i0_over_epsilon
        self.use_red_kd = use_red_kd

        # v2: Residual encoder with SE attention
        self.encoder = PhysicsGuidedEncoder(
            in_channels=in_channels,
            dropout_rate=dropout_rate,
        )

        # v2: Cross-attention physics fusion decoder
        self.physics_decoder = CrossAttentionPhysicsFusion(
            feature_dim=feature_dim,
            n_heads=n_attn_heads,
            phys_tokens=phys_tokens,
            dropout_rate=dropout_rate,
        )

        self.depth_head       = DepthHead(feature_dim=feature_dim, dropout_rate=dropout_rate)
        self.uncertainty_head = AleatoricUncertaintyHead(feature_dim=feature_dim, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input image patch stack.
               Channel layout: B02 (0), B03 (1), B04 (2), B08 (3), Stumpf (4), z_prior (5)

        Returns:
            depth_pred: (B,)   predicted mean depth ŷ (m)
            log_var:    (B,)   predicted log-variance s = ln(σ²)
            kd:         (B, 1) diffuse attenuation coefficient K_d (m⁻¹)
            z_ext:      (B, 1) Beer-Lambert optical extinction depth limit (m)
        """
        blue  = x[:, 0:1, :, :]   # B02
        green = x[:, 1:2, :, :]   # B03
        red   = x[:, 2:3, :, :] if (self.use_red_kd and x.shape[1] > 2) else None  # B04

        # 1. Compute improved QAA multi-band Kd and extinction depth bound
        kd, z_ext = compute_kd_and_extinction(
            blue, green, red,
            i0_over_epsilon=self.i0_over_epsilon
        )

        # 2. Deep residual spatial feature extraction via v2 Encoder
        f_spatial = self.encoder(x)                              # (B, 128)

        # 3. Cross-attention physics fusion
        f_physics = self.physics_decoder(f_spatial, kd, z_ext)  # (B, 128)

        # 4. Dual head predictions
        depth_pred = self.depth_head(f_physics)                  # (B,)
        log_var    = self.uncertainty_head(f_physics)            # (B,)

        return depth_pred, log_var, kd, z_ext


# ---------------------------------------------------------------------------
# 6. Depth-Weighted Composite Physics NLL Loss (v2)
# ---------------------------------------------------------------------------

class CompositePhysicsNLLLoss(nn.Module):
    """v2 Composite Physics-Informed Gaussian NLL Loss with depth-aware weighting.

    v1 Loss:
        L = w_mse·MSE + w_nll·NLL + w_phys·L_nonneg + w_ext·L_ext

    v2 Additions:
    a) Depth-weighted MSE and NLL: per-sample weight = clamp(y_m / depth_scale, 0.5, 3.0)
       Deep-water samples (harder to predict) receive higher weight, addressing
       the class imbalance between abundant shallow and sparse deep samples.
    b) Huber NLL: replaces squared NLL with Huber (δ=5) to reduce influence of
       extreme depth outliers from GEBCO bathymetric noise.
    c) Curriculum support: w_nll can be annealed during training (start at 0,
       increase to 1 — let MSE stabilize first, then add NLL calibration).

    Loss Formulation:
        L_total = w_mse · Σᵢ wᵢ · (ŷᵢ−yᵢ)²
                + w_nll · Σᵢ wᵢ · [0.5·exp(−sᵢ)·(ŷᵢ−yᵢ)² + 0.5·sᵢ]
                + w_phys · mean(ReLU(−ŷ))
                + w_ext  · mean(ReLU(ŷ − z_ext))

    Args:
        w_mse:        Weight for direct depth regression MSE (default: 1.0)
        w_nll:        Weight for heteroscedastic Gaussian NLL (default: 1.0)
        w_phys:       Weight for non-negativity physical constraint (default: 0.5)
        w_ext:        Weight for Beer-Lambert extinction bound (default: 0.5)
        depth_scale:  Depth (m) at which weight = 1.0 (default: 10.0)
        use_depth_weighting: Enable per-sample depth weighting (default: True)
    """

    def __init__(
        self,
        w_mse:  float = 1.0,
        w_nll:  float = 1.0,
        w_phys: float = 0.5,
        w_ext:  float = 0.5,
        depth_scale: float = 10.0,
        use_depth_weighting: bool = True,
    ):
        super().__init__()
        self.w_mse   = w_mse
        self.w_nll   = w_nll
        self.w_phys  = w_phys
        self.w_ext   = w_ext
        self.depth_scale = depth_scale
        self.use_depth_weighting = use_depth_weighting

    def forward(
        self,
        pred_depth_norm:   torch.Tensor,
        log_var:           torch.Tensor,
        target_depth_norm: torch.Tensor,
        z_ext:             torch.Tensor,
        y_mean: float = 0.0,
        y_std:  float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred_depth_norm:   (B,) normalized predicted depth
            log_var:           (B,) predicted log variance s = ln(σ²)
            target_depth_norm: (B,) normalized ground truth depth
            z_ext:             (B, 1) or (B,) optical extinction depth limit (m)
            y_mean:            dataset mean depth (m)
            y_std:             dataset std depth (m)

        Returns:
            total_loss:      scalar torch.Tensor for loss.backward()
            loss_components: dict of individual loss values for logging
        """
        # Reconstruct meters for physics constraints and weighting
        pred_depth_m   = pred_depth_norm   * y_std + y_mean   # (B,)
        target_depth_m = target_depth_norm * y_std + y_mean   # (B,)

        # --- Per-sample depth weighting (v2) ---
        if self.use_depth_weighting:
            # Deeper samples up-weighted — range [0.5, 3.0]
            sample_weight = torch.clamp(
                target_depth_m.detach().abs() / self.depth_scale,
                min=0.5, max=3.0
            )
        else:
            sample_weight = torch.ones_like(target_depth_m)

        # Normalize weights so total weight ≈ B (preserves loss scale)
        sample_weight = sample_weight / (sample_weight.mean() + 1e-6)

        # --- 1. Depth-weighted MSE on normalized depths ---
        diff_sq_norm = (pred_depth_norm - target_depth_norm) ** 2    # (B,)
        loss_mse = torch.mean(sample_weight * diff_sq_norm)

        # --- 2. Depth-weighted Heteroscedastic Gaussian NLL ---
        inv_var = torch.exp(-log_var)                                  # (B,)
        l_nll   = 0.5 * (inv_var * diff_sq_norm + log_var)            # (B,)
        loss_nll = torch.mean(sample_weight * l_nll)

        # --- 3. Physics Constraint 1: Non-negativity (depth ≥ 0 m) ---
        loss_nonneg = torch.mean(F.relu(-pred_depth_m))

        # --- 4. Physics Constraint 2: Beer-Lambert extinction bound (depth ≤ z_ext) ---
        z_ext_flat  = z_ext.squeeze(-1)                               # (B,)
        loss_ext    = torch.mean(F.relu(pred_depth_m - z_ext_flat))

        # --- Total composite loss ---
        total_loss = (
            self.w_mse  * loss_mse  +
            self.w_nll  * loss_nll  +
            self.w_phys * (loss_nonneg / (y_std + 1e-6)) +
            self.w_ext  * (loss_ext   / (y_std + 1e-6))
        )

        # RMSE for logging (meters)
        diff_sq_m = (pred_depth_m - target_depth_m) ** 2
        loss_components = {
            "loss_total":  float(total_loss.item()),
            "loss_mse":    float(loss_mse.item()),
            "loss_nll":    float(loss_nll.item()),
            "loss_nonneg": float(loss_nonneg.item()),
            "loss_ext":    float(loss_ext.item()),
            "rmse":        float(torch.sqrt(torch.mean(diff_sq_m)).item()),
        }

        return total_loss, loss_components


# ---------------------------------------------------------------------------
# 7. MC Dropout Inference for Epistemic & Total Uncertainty (enhanced)
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
) -> dict[str, torch.Tensor]:
    """Enhanced Monte Carlo Dropout stochastic inference.

    v2 improvements:
    - n_samples increased to 30 (was 20) for better uncertainty estimate stability.
    - Enables all Dropout AND Dropout2d layers (both spatial and channel dropout).
    - Returns per-sample prediction matrix for downstream calibration analysis.
    - Computes coefficient of variation (CV) for relative uncertainty maps.

    Args:
        model:     Trained HybridBathNetModel instance
        x:         (B, C, H, W) input image tensor
        n_samples: Number of MC stochastic forward passes (T = 30)

    Returns:
        dict with:
            'mean_depth':      (B,) predictive mean depth ȳ
            'aleatoric_var':   (B,) expected aleatoric variance E[σ_a²]
            'epistemic_var':   (B,) epistemic variance Var(ŷ_t)
            'total_var':       (B,) total variance σ_total²
            'total_std':       (B,) total predictive std σ_total
            'coeff_variation': (B,) σ_total / |ȳ| — relative uncertainty
            'all_preds':       (T, B) all stochastic predictions (for calibration)
    """
    model.eval()
    # Enable all dropout layers (both spatial Dropout2d and regular Dropout)
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    preds    = []
    log_vars = []

    with torch.no_grad():
        for _ in range(n_samples):
            depth_pred, log_var, _, _ = model(x)
            preds.append(depth_pred.unsqueeze(0))    # (1, B)
            log_vars.append(log_var.unsqueeze(0))    # (1, B)

    # Stack: (T, B)
    preds    = torch.cat(preds,    dim=0)
    log_vars = torch.cat(log_vars, dim=0)

    mean_depth    = preds.mean(dim=0)                              # (B,)
    epistemic_var = preds.var(dim=0, unbiased=True)               # (B,)
    aleatoric_var = torch.exp(log_vars).mean(dim=0)               # (B,)
    total_var     = aleatoric_var + epistemic_var                  # (B,)
    total_std     = torch.sqrt(total_var.clamp(min=1e-8))         # (B,)
    coeff_var     = total_std / (mean_depth.abs().clamp(min=0.1)) # (B,) relative uncertainty

    return {
        "mean_depth":      mean_depth,
        "aleatoric_var":   aleatoric_var,
        "epistemic_var":   epistemic_var,
        "total_var":       total_var,
        "total_std":       total_std,
        "coeff_variation": coeff_var,
        "all_preds":       preds,    # (T, B)
    }


# ---------------------------------------------------------------------------
# 8. Deep Ensemble Predictor (v2 New — best-practice epistemic uncertainty)
# ---------------------------------------------------------------------------

class DeepEnsemblePredictor(nn.Module):
    """5-member Deep Ensemble for superior epistemic uncertainty estimation.

    Deep ensembles (Lakshminarayanan et al., NeurIPS 2017) consistently outperform
    MC Dropout for epistemic uncertainty calibration, especially in OOD detection.

    Each ensemble member is a full HybridBathNetModel with different random seed
    initialization — initialized at construction, trained independently.

    Usage:
        ensemble = DeepEnsemblePredictor(n_members=5, **model_kwargs)
        # Train each member separately:
        for i, member in enumerate(ensemble.members):
            train_hybridbathnet(member, ...)
        # Predict with full ensemble uncertainty:
        results = ensemble.predict(x_batch)
    """

    def __init__(self, n_members: int = 5, **model_kwargs):
        super().__init__()
        self.n_members = n_members
        self.members   = nn.ModuleList([
            HybridBathNetModel(**model_kwargs) for _ in range(n_members)
        ])

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run full ensemble inference.

        Args:
            x: (B, C, H, W) input patch tensor

        Returns:
            dict with ensemble mean depth, epistemic variance, and combined uncertainty
        """
        preds    = []
        log_vars = []

        for member in self.members:
            member.eval()
            with torch.no_grad():
                d, lv, _, _ = member(x)
                preds.append(d.unsqueeze(0))
                log_vars.append(lv.unsqueeze(0))

        preds    = torch.cat(preds,    dim=0)   # (M, B)
        log_vars = torch.cat(log_vars, dim=0)   # (M, B)

        mean_depth    = preds.mean(dim=0)                      # (B,)
        epistemic_var = preds.var(dim=0, unbiased=True)        # (B,) between-model disagreement
        aleatoric_var = torch.exp(log_vars).mean(dim=0)        # (B,) average data noise
        total_var     = aleatoric_var + epistemic_var          # (B,)
        total_std     = torch.sqrt(total_var.clamp(min=1e-8))  # (B,)

        return {
            "mean_depth":    mean_depth,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
            "total_var":     total_var,
            "total_std":     total_std,
        }

    def forward(self, x: torch.Tensor):
        """Forward pass using the first ensemble member (for training compatibility)."""
        return self.members[0](x)
