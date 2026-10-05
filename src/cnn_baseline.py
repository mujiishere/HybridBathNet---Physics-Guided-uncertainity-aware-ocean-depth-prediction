"""
CNN depth-regression baseline + HybridBathNet v2 training pipeline.

Architecture overview
---------------------
Input  : (B, C, patch_size, patch_size)   — raw spectral bands + Stumpf ratio
Encoder: Residual blocks + SE attention (v2) with MaxPool downsampling
Pool   : AdaptiveAvgPool2d(1) + flatten   → (B, FEATURE_DIM)  ← encoder output
Heads  : DepthHead (mean depth) + AleatoricUncertaintyHead (log σ²)

Label  : center-pixel GEBCO depth (scalar, metres, positive = below sea level).

v2 Training Improvements (no new data required):
  1. SpectralAugDataset     — spectral jitter + random spatial flips augmentation
  2. DepthStratifiedSampler — depth-balanced mini-batches (curriculum learning)
  3. Cosine LR w/ restarts  — CosineAnnealingWarmRestarts scheduler
  4. Gradient clipping      — torch.nn.utils.clip_grad_norm_ (max_norm=1.0)
  5. Curriculum NLL         — NLL weight linearly annealed 0→1 over first 20 epochs

Usage:
    python -m src.cnn_baseline --config config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("cnn_baseline")

# Dimensionality of the flat feature vector the encoder produces.
# Must equal the output channels of the bottleneck block.
FEATURE_DIM = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Loads pre-extracted patches from patches.npz with global channel z-score standardization.

    Using dataset-wide global mean/std per channel preserves physical unit scaling
    (especially for channel 5 Beer-Lambert z_prior in meters) across all patches.

    Normalization stats (mean/std for both inputs and targets) must be computed
    from the TRAINING split only, and passed to val/test via the constructor.

    Returns three items per sample:
        x_norm   : (C, H, W) z-score normalized input channels
        y_norm   : scalar, z-score normalized center-pixel depth
        y_meters : scalar, original center-pixel depth in meters (for physics loss)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, indices: np.ndarray,
                 mean: np.ndarray = None, std: np.ndarray = None,
                 y_mean: float = None, y_std: float = None):
        self.X = X[indices]   # (N, C, H, W)
        self.Y = Y[indices]   # (N, H, W)

        # --- Input channel normalization stats ---
        if mean is None:
            # Compute channel-wise mean and std across spatial dimensions
            self.mean = self.X.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
            self.std = (self.X.std(axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32)
        else:
            self.mean = mean
            self.std = std

        # --- Target depth normalization stats ---
        cy, cx = self.Y.shape[1] // 2, self.Y.shape[2] // 2
        center_depths = self.Y[:, cy, cx]
        if y_mean is None:
            self.y_mean = float(center_depths.mean())
            self.y_std  = float(center_depths.std()) + 1e-6
        else:
            self.y_mean = y_mean
            self.y_std  = y_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (C, H, W)

        # Global channel z-score standardization
        x = (x - self.mean.squeeze()[:, None, None]) / self.std.squeeze()[:, None, None]

        # Scalar depth label: center pixel of the depth patch
        cy, cx = x.shape[1] // 2, x.shape[2] // 2
        y_meters = float(self.Y[idx, cy, cx])

        # Normalized depth for loss computation
        y_norm = (y_meters - self.y_mean) / self.y_std

        return (
            torch.from_numpy(x).float(),
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(y_meters, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Dataset v2: Spectral Augmentation Wrapper
# ---------------------------------------------------------------------------

class SpectralAugDataset(Dataset):
    """Wraps PatchDataset and applies on-the-fly spectral + spatial augmentations.

    Augmentations (training only — disable for val/test):
    1. Spectral jitter: per-channel Gaussian noise (σ=0.02) simulates atmospheric
       variability and sensor calibration uncertainty between acquisitions.
    2. Random horizontal flip: doubles effective spatial diversity.
    3. Random vertical flip: additional symmetry augmentation.
    4. Random 90° rotation: valid for satellite patches (no preferred orientation).

    These augmentations help the model generalize better across:
    - Different atmospheric conditions (spectral jitter)
    - Varied ocean current / tide directions (spatial flips)

    Args:
        base_dataset: PatchDataset instance to wrap
        augment:      True for training, False for val/test
        noise_std:    Standard deviation of per-channel spectral jitter (default: 0.02)
    """

    def __init__(self, base_dataset: PatchDataset, augment: bool = True, noise_std: float = 0.02):
        self.base      = base_dataset
        self.augment   = augment
        self.noise_std = noise_std

    # Forward __len__ and normalization attributes from the base dataset
    def __len__(self) -> int:
        return len(self.base)

    @property
    def y_mean(self): return self.base.y_mean
    @property
    def y_std(self):  return self.base.y_std
    @property
    def mean(self):   return self.base.mean
    @property
    def std(self):    return self.base.std

    def __getitem__(self, idx):
        x, y_norm, y_meters = self.base[idx]   # x: (C, H, W) already normalized

        if self.augment:
            # 1. Spectral jitter (channels 0–3 = spectral; 4–5 = physics — skip jitter)
            n_spectral = min(4, x.shape[0])
            noise = torch.randn(n_spectral, 1, 1) * self.noise_std
            x = x.clone()
            x[:n_spectral] = x[:n_spectral] + noise

            # 2. Random horizontal flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])   # flip W axis

            # 3. Random vertical flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])   # flip H axis

            # 4. Random 90° rotation (k=0,1,2,3)
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                x = torch.rot90(x, k=k, dims=[1, 2])

        return x, y_norm, y_meters


# ---------------------------------------------------------------------------
# Depth-Stratified Sampler (curriculum learning — balanced depth batches)
# ---------------------------------------------------------------------------

class DepthStratifiedSampler(Sampler):
    """Yields mini-batch indices with balanced depth-stratum representation.

    Motivation:
    Shallow pixels (0–5 m) are heavily over-represented in coastal datasets.
    Without stratification, models overfit to shallow depths while performing
    poorly in deeper regions. This sampler ensures each batch draws equal numbers
    of samples from each depth stratum.

    Strategy:
    1. Divide training samples into N depth strata by percentile boundaries.
    2. Each epoch, sample equal numbers from each stratum (with replacement
       from smaller strata to match the largest).
    3. Shuffle within each stratum for intra-epoch randomness.

    Args:
        depths:    1-D array of center-pixel depths (m) for training samples.
        n_strata:  Number of depth quantile strata (default: 5).
        seed:      Random seed for reproducibility.
    """

    def __init__(self, depths: np.ndarray, n_strata: int = 5, seed: int = 42):
        self.rng      = np.random.RandomState(seed)
        self.n_strata = n_strata

        # Compute stratum boundaries as depth percentiles
        percentiles = np.linspace(0, 100, n_strata + 1)
        boundaries  = np.percentile(depths, percentiles)
        boundaries[0]  -= 1e-3   # inclusive lower bound
        boundaries[-1] += 1e-3   # inclusive upper bound

        # Group sample indices by stratum
        self.strata = []
        for i in range(n_strata):
            mask = (depths >= boundaries[i]) & (depths < boundaries[i + 1])
            idxs = np.where(mask)[0]
            self.strata.append(idxs)
            LOG.debug("Stratum %d: depth=[%.1f, %.1f] m  n=%d",
                      i, boundaries[i], boundaries[i+1], len(idxs))

        # Samples per stratum = size of largest stratum (upsample smaller)
        self.n_per_stratum = max(len(s) for s in self.strata)
        self._total = self.n_per_stratum * n_strata

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        indices = []
        for stratum in self.strata:
            if len(stratum) == 0:
                continue
            # Sample with replacement to match n_per_stratum
            chosen = self.rng.choice(stratum, size=self.n_per_stratum, replace=True)
            self.rng.shuffle(chosen)
            indices.extend(chosen.tolist())
        self.rng.shuffle(indices)
        return iter(indices)



# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Spatial feature encoder.

    Compresses an input patch (B, C, H, W) into a flat feature vector
    (B, FEATURE_DIM) using three convolutional blocks and global average
    pooling.  The decoder is intentionally removed — two prediction heads
    (DepthHead, UncertaintyHead) will attach to this output.

    Flow:
        enc1       : C  → 32    full resolution
        pool1      : ÷2 spatially
        enc2       : 32 → 64
        pool2      : ÷2 spatially
        bottleneck : 64 → FEATURE_DIM (128)
        gap        : AdaptiveAvgPool2d(1)  — works for ANY patch size
        flatten    → (B, FEATURE_DIM)

    Args:
        in_channels: number of input channels (spectral bands + physics channels)
    """

    def __init__(self, in_channels: int):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1       = block(in_channels, 32)
        self.pool1      = nn.MaxPool2d(2)
        self.enc2       = block(32, 64)
        self.pool2      = nn.MaxPool2d(2)
        self.bottleneck = block(64, FEATURE_DIM)

        # Resolution-agnostic pooling: (B, FEATURE_DIM, H', W') → (B, FEATURE_DIM)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            features: (B, FEATURE_DIM)  — NOT a depth prediction; feed to heads
        """
        x = self.enc1(x)
        x = self.enc2(self.pool1(x))
        x = self.bottleneck(self.pool2(x))
        x = self.gap(x)
        return self.flatten(x)   # (B, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Placeholder head  (TEMPORARY — replace with DepthHead + UncertaintyHead)
# ---------------------------------------------------------------------------

class _TempDepthHead(nn.Module):
    """**PLACEHOLDER — delete when DepthHead + UncertaintyHead are added.**

    Single Linear(FEATURE_DIM → 1) so training is runnable immediately.

    Swap-out checklist when adding the real heads:
      1. Delete this class.
      2. Add DepthHead(FEATURE_DIM) → scalar mean depth.
      3. Add UncertaintyHead(FEATURE_DIM) → scalar log-variance (σ²).
      4. Switch loss from scalar_mse → Gaussian NLL.
      5. Update train() and evaluate() to handle two head outputs.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FEATURE_DIM, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, FEATURE_DIM)
        Returns:
            depth_pred: (B,)
        """
        return self.fc(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------

def scalar_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE on per-patch scalar predictions vs. center-pixel GEBCO depth."""
    return torch.mean((pred - target) ** 2)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression + bathymetry-specific accuracy metrics (scalar labels)."""
    p, t = y_pred.flatten(), y_true.flatten()
    if len(p) == 0:
        nan = float("nan")
        return {"rmse": nan, "mae": nan, "r2": nan,
                "acc_within_1m_%": nan, "acc_within_2m_%": nan,
                "delta_1.25_%": nan}

    abs_diff = np.abs(p - t)
    rmse     = float(np.sqrt(np.mean((p - t) ** 2)))
    mae      = float(np.mean(abs_diff))
    ss_res   = np.sum((t - p) ** 2)
    ss_tot   = np.sum((t - t.mean()) ** 2)
    r2       = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Depth accuracy against GEBCO ground truth
    acc_1m  = float(np.mean(abs_diff <= 1.0) * 100.0)  # within ±1 m
    acc_2m  = float(np.mean(abs_diff <= 2.0) * 100.0)  # within ±2 m
    pos     = (t > 0.1) & (p > 0.1)
    delta_1 = (
        float(np.mean(np.maximum(t[pos] / p[pos], p[pos] / t[pos]) < 1.25) * 100.0)
        if np.any(pos) else float("nan")
    )

    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "acc_within_1m_%": acc_1m, "acc_within_2m_%": acc_2m,
        "delta_1.25_%": delta_1,
    }


# ---------------------------------------------------------------------------
# Training & evaluation with HybridBathNet (Physics Decoder + Dual Heads + NLL Loss)
# ---------------------------------------------------------------------------

from .hybridbathnet import HybridBathNetModel, CompositePhysicsNLLLoss, predict_with_uncertainty


def train_hybridbathnet(
    model, criterion, train_loader, val_loader,
    epochs, lr, device, y_mean, y_std,
    nll_warmup_epochs: int = 20,
    base_nll_weight: float = 1.0,
    grad_clip_norm: float = 1.0,
    lr_restart_period: int = 30,
):
    """Train HybridBathNet v2 with all no-new-data training improvements.

    v2 Training Improvements:
    1. Cosine Annealing Warm Restarts (CosineAnnealingWarmRestarts, T_0=lr_restart_period):
       LR oscillates between lr and 0 in cosine pattern, resetting periodically.
       This helps escape local minima and explore more of the loss landscape.

    2. Gradient Clipping (clip_grad_norm_, max_norm=grad_clip_norm=1.0):
       Prevents exploding gradients — critical with the physics loss terms which
       can produce large gradients when predictions violate z_ext bounds strongly.

    3. Curriculum NLL Weight Annealing (nll_warmup_epochs=20):
       - Epochs 1–20: w_nll linearly annealed from 0.0 → base_nll_weight.
       - The model first learns depth regression (MSE) to get sensible predictions,
         then gradually adds the NLL term to calibrate uncertainty estimates.
       - Prevents the log_var head from diverging early in training.

    4. Depth-Stratified Sampler (applied via train_loader — set up in main()):
       Curriculum-aware balanced batches across depth strata.

    Args:
        model:              HybridBathNetModel instance
        criterion:          CompositePhysicsNLLLoss instance
        train_loader:       DataLoader (with DepthStratifiedSampler recommended)
        val_loader:         DataLoader (standard)
        epochs:             Total training epochs
        lr:                 Initial learning rate
        device:             'cuda' or 'cpu'
        y_mean, y_std:      Depth normalization statistics
        nll_warmup_epochs:  Epochs to linearly ramp up NLL weight (default: 20)
        base_nll_weight:    Final NLL weight after warmup (default: 1.0)
        grad_clip_norm:     Max gradient norm for clipping (default: 1.0)
        lr_restart_period:  CosineAnnealingWarmRestarts T_0 period in epochs (default: 30)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine Annealing Warm Restarts: LR restarts every lr_restart_period epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=lr_restart_period, T_mult=1, eta_min=lr * 0.01
    )

    best_val_rmse = float("inf")
    best_state    = None

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training HybridBathNet v2", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss  = 0.0
        train_nll   = 0.0
        train_nonneg = 0.0
        train_ext   = 0.0
        n_batches   = 0

        # --- Curriculum NLL weight annealing ---
        # Ramp from 0 → base_nll_weight over the first nll_warmup_epochs
        if epoch <= nll_warmup_epochs:
            curr_nll_w = base_nll_weight * (epoch / nll_warmup_epochs)
        else:
            curr_nll_w = base_nll_weight
        criterion.w_nll = curr_nll_w

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr

        for x, y_norm, y_m in train_loader:
            x, y_norm, y_m = x.to(device), y_norm.to(device), y_m.to(device)
            optimizer.zero_grad()

            pred_norm, log_var, kd, z_ext = model(x)
            loss, loss_dict = criterion(
                pred_norm, log_var, y_norm, z_ext,
                y_mean=y_mean, y_std=y_std
            )

            loss.backward()

            # Gradient clipping — prevents physics constraint loss explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            train_loss   += loss_dict["loss_total"]
            train_nll    += loss_dict["loss_nll"]
            train_nonneg += loss_dict["loss_nonneg"]
            train_ext    += loss_dict["loss_ext"]
            n_batches    += 1

        # Step the LR scheduler after each epoch
        scheduler.step()

        # Normalize by batch count (sampler may produce variable dataset size)
        if n_batches > 0:
            train_loss   /= n_batches
            train_nll    /= n_batches
            train_nonneg /= n_batches
            train_ext    /= n_batches

        val_metrics = evaluate_hybridbathnet(model, val_loader, device, y_mean, y_std)

        epoch_bar.set_postfix(
            loss=f"{train_loss:.3f}",
            nll_w=f"{curr_nll_w:.2f}",
            lr=f"{current_lr:.2e}",
            val_rmse=f"{val_metrics['rmse']:.2f}m",
            val_r2=f"{val_metrics['r2']:.3f}",
        )
        LOG.info(
            "Epoch %3d/%d | Loss: %.4f (NLL[w=%.2f]: %.4f, NonNeg: %.4f, Ext: %.4f) "
            "| LR: %.2e | Val → RMSE: %.3fm  MAE: %.3fm  R²: %.3f  "
            "Acc(±1m): %.1f%%  Acc(±2m): %.1f%%  δ<1.25: %.1f%%",
            epoch, epochs, train_loss, curr_nll_w, train_nll, train_nonneg, train_ext,
            current_lr,
            val_metrics["rmse"], val_metrics["mae"], val_metrics["r2"],
            val_metrics["acc_within_1m_%"], val_metrics["acc_within_2m_%"],
            val_metrics["delta_1.25_%"],
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        LOG.info("Loaded best checkpoint (val RMSE: %.3f m)", best_val_rmse)

    return model


@torch.no_grad()
def evaluate_hybridbathnet(model, loader, device, y_mean, y_std) -> dict:
    model.eval()
    all_pred, all_true, all_aleatoric = [], [], []

    for x, _, y_m in loader:
        x = x.to(device)
        pred_norm, log_var, _, _ = model(x)
        pred_m = pred_norm.cpu().numpy() * y_std + y_mean
        all_pred.append(pred_m)
        all_true.append(y_m.numpy())
        all_aleatoric.append(torch.exp(log_var).cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(all_true),
        np.concatenate(all_pred),
    )
    metrics["mean_aleatoric_std"] = float(np.sqrt(np.mean(np.concatenate(all_aleatoric))))
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    feat_dir = cfg["features"]["features_dir"]
    data = np.load(Path(feat_dir) / "patches.npz")
    X, Y  = data["X"], data["Y"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    LOG.info(
        "Patches — X: %s  Y: %s  |  channels: %d (spectral + physics)",
        X.shape, Y.shape, X.shape[1],
    )

    train_cfg = cfg["training"]
    hnet_cfg  = cfg.get("hybridbathnet", {})
    device    = train_cfg["device"] if torch.cuda.is_available() else "cpu"
    LOG.info("Using device: %s", device)

    # --- v2: Build datasets with spectral augmentation ---
    base_train_ds = PatchDataset(X, Y, train_idx)
    base_val_ds   = PatchDataset(
        X, Y, val_idx  if len(val_idx)  else train_idx,
        mean=base_train_ds.mean, std=base_train_ds.std,
        y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std
    )
    base_test_ds  = PatchDataset(
        X, Y, test_idx if len(test_idx) else train_idx,
        mean=base_train_ds.mean, std=base_train_ds.std,
        y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std
    )

    if len(test_idx) == 0:
        LOG.warning("test_idx is empty — falling back to train set for test evaluation.")
    if len(val_idx) == 0:
        LOG.warning("val_idx is empty — falling back to train set for validation.")

    # Wrap training dataset with spectral+spatial augmentation (val/test: no augmentation)
    noise_std = hnet_cfg.get("aug_noise_std", 0.02)
    train_ds  = SpectralAugDataset(base_train_ds, augment=True,  noise_std=noise_std)
    val_ds    = SpectralAugDataset(base_val_ds,   augment=False)
    test_ds   = SpectralAugDataset(base_test_ds,  augment=False)

    LOG.info("SpectralAugDataset: train augmentation ON (noise_std=%.3f), val/test OFF", noise_std)

    # --- v2: Depth-stratified sampler for curriculum-aware balanced batches ---
    n_strata = hnet_cfg.get("depth_strata", 5)
    cy_train = base_train_ds.Y.shape[1] // 2
    cx_train = base_train_ds.Y.shape[2] // 2
    train_depths = base_train_ds.Y[:, cy_train, cx_train]   # center-pixel depths (m)
    depth_sampler = DepthStratifiedSampler(train_depths, n_strata=n_strata, seed=42)
    LOG.info("DepthStratifiedSampler: %d strata, %d samples/epoch", n_strata, len(depth_sampler))

    batch_size   = train_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=depth_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # --- Instantiate HybridBathNet v2 ---
    loss_weights = hnet_cfg.get("loss_weights", {"w_mse": 1.0, "w_nll": 1.0, "w_phys": 0.5, "w_ext": 0.5})
    model = HybridBathNetModel(
        in_channels=X.shape[1],
        feature_dim=hnet_cfg.get("feature_dim", FEATURE_DIM),
        dropout_rate=hnet_cfg.get("dropout_rate", 0.1),
        i0_over_epsilon=hnet_cfg.get("i0_over_epsilon", 100.0),
        use_red_kd=hnet_cfg.get("use_red_kd", True),
        n_attn_heads=hnet_cfg.get("n_attn_heads", 4),
        phys_tokens=hnet_cfg.get("phys_tokens", 4),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("HybridBathNet v2 | Trainable parameters: %s", f"{n_params:,}")

    criterion = CompositePhysicsNLLLoss(
        w_mse=loss_weights.get("w_mse", 1.0),
        w_nll=loss_weights.get("w_nll", 1.0),   # will be annealed by curriculum
        w_phys=loss_weights.get("w_phys", 0.5),
        w_ext=loss_weights.get("w_ext", 0.5),
        depth_scale=hnet_cfg.get("depth_weight_scale", 10.0),
        use_depth_weighting=hnet_cfg.get("use_depth_weighting", True),
    )

    LOG.info("HybridBathNet v2 initialized from random weights.")
    LOG.info("Normalization stats (from TRAINING split only):")
    LOG.info("  Input channels  -> mean: %s", str(base_train_ds.mean.squeeze().tolist()))
    LOG.info("  Target depth    -> y_mean: %.3f m  y_std: %.3f m", base_train_ds.y_mean, base_train_ds.y_std)
    LOG.info("Loss Weights: MSE=%.2f | NLL(→%.2f) | NonNeg=%.2f | Ext=%.2f",
             loss_weights["w_mse"], loss_weights["w_nll"],
             loss_weights["w_phys"], loss_weights["w_ext"])

    nll_warmup   = hnet_cfg.get("nll_warmup_epochs",  20)
    lr_restart   = hnet_cfg.get("lr_restart_period",  30)
    grad_clip    = hnet_cfg.get("grad_clip_norm",      1.0)

    model = train_hybridbathnet(
        model, criterion,
        train_loader, val_loader,
        train_cfg["epochs"], train_cfg["learning_rate"], device,
        y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std,
        nll_warmup_epochs=nll_warmup,
        base_nll_weight=loss_weights.get("w_nll", 1.0),
        grad_clip_norm=grad_clip,
        lr_restart_period=lr_restart,
    )

    ensure_dir(train_cfg["checkpoint_dir"])
    torch.save(
        {
            "model":  model.state_dict(),
            "y_mean": base_train_ds.y_mean,
            "y_std":  base_train_ds.y_std,
            "version": "v2",
        },
        Path(train_cfg["checkpoint_dir"]) / "hybridbathnet.pt",
    )

    test_metrics = evaluate_hybridbathnet(
        model, test_loader, device,
        y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std
    )
    LOG.info(
        "HybridBathNet v2 (test) → RMSE %.3f  MAE %.3f  R² %.3f  (Aleatoric std: %.3fm)",
        test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"], test_metrics["mean_aleatoric_std"],
    )


    ensure_dir(cfg["evaluation"]["results_dir"])
    with open(Path(cfg["evaluation"]["results_dir"]) / "hybridbathnet_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()

