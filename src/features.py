"""
Feature engineering: Lyzenga log-linearised bands, sun-glint correction,
Stumpf ratio, NDWI water index, water/valid mask channel, temporal std
channel, distance to shore (optional), two-band Beer-Lambert z_prior,
a per-pixel tabular dataset for Stumpf/RF baselines, and patch extraction
with buffered spatial train/val/test split.

Changes from previous version:
  §1.2  Raw reflectance passed to physics; z-scored channels no longer fed
        into physics formulas.
  §2.1  Lyzenga log-linearised channels: ln(R - R∞) for each band, plus
        band-ratio channels ln(R_B02/R_B03) and ln(R_B03/R_B04).
  §2.2  Water/valid mask added as an explicit input channel; land pixels are
        set to a constant before normalisation so they do not pollute stats.
  §2.3  NIR-based sun-glint correction (Hedley et al. 2005).
  §2.4  Per-pixel temporal std from the composite is included as input.
  §2.5  Optional distance-to-shore channel from the downloaded coastline.
  §2.6  Two-band Beer-Lambert z_prior (blue + green) with water-only
        surface reference computed from training split statistics.
  §1.5  Buffered spatial patch split: patches within one patch-width of a
        block boundary are dropped, preventing leakage at block edges.
  Dense label patch: full (H, W) depth map returned per patch (not only
        the centre pixel) for future dense output heads.

Usage:
    python -m src.features --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("features")

STUMPF_N = 1000.0   # Stumpf (2003) fixed scaling constant


# ---------------------------------------------------------------------------
# Band-level helpers — operate on raw reflectance (0–1 float) arrays
# ---------------------------------------------------------------------------

def stumpf_ratio(blue: np.ndarray, green: np.ndarray) -> np.ndarray:
    """ln(n·blue) / ln(n·green) — Stumpf, Holderied & Sinclair (2003)."""
    b = np.clip(blue, 1e-6, None)
    g = np.clip(green, 1e-6, None)
    return np.log(STUMPF_N * b) / np.log(STUMPF_N * g)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (green - nir) / (green + nir + 1e-6)


def sun_glint_correction(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    water_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NIR-based sun-glint correction (Hedley et al., 2005).

    For each visible band V, regress V ~ NIR over deep-water pixels, then:
        V_corrected = V - slope × (NIR - NIR_min)
    where NIR_min is the minimum NIR value over the deep-water sample.

    Args:
        blue, green, red: (H, W) visible reflectance bands
        nir:              (H, W) NIR reflectance (B08)
        water_mask:       (H, W) boolean — True where water (no land contamination)
    Returns:
        corrected blue, green, red (H, W each)
    """
    # Use pixels where NIR is low (clear, deep water) — bottom 5th percentile
    valid = water_mask & np.isfinite(nir) & np.isfinite(blue)
    if valid.sum() < 20:
        return blue, green, red   # not enough samples — skip correction

    nir_vals = nir[valid]
    nir_min  = float(np.percentile(nir_vals, 5))

    def _regress_slope(vis: np.ndarray) -> float:
        v = vis[valid]
        n = nir_vals
        denom = np.sum((n - n.mean()) ** 2)
        if denom < 1e-12:
            return 0.0
        return float(np.sum((v - v.mean()) * (n - n.mean())) / denom)

    delta_nir = nir - nir_min

    blue_c  = blue  - _regress_slope(blue)  * delta_nir
    green_c = green - _regress_slope(green) * delta_nir
    red_c   = red   - _regress_slope(red)   * delta_nir

    return (
        np.clip(blue_c,  0.0, None),
        np.clip(green_c, 0.0, None),
        np.clip(red_c,   0.0, None),
    )


def lyzenga_channels(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    r_inf_b: float,
    r_inf_g: float,
    r_inf_r: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lyzenga (1978) log-linearised depth channels.

    X_i = ln(R_i − R∞_i),  where R∞_i = deep-water (2nd percentile) reflectance.
    Also returns band-ratio channels ln(R_B02/R_B03) and ln(R_B03/R_B04).

    Args:
        blue/green/red:  (H, W) corrected reflectance
        r_inf_b/g/r:     deep-water (2nd-percentile) reflectance per band
                         computed from training pixels only
    Returns:
        lyz_b, lyz_g, lyz_r, ratio_bg, ratio_gr  — all (H, W)
    """
    lyz_b = np.log(np.clip(blue  - r_inf_b + eps, eps, None))
    lyz_g = np.log(np.clip(green - r_inf_g + eps, eps, None))
    lyz_r = np.log(np.clip(red   - r_inf_r + eps, eps, None))

    ratio_bg = lyz_b - lyz_g   # ln(R_b/R_g)
    ratio_gr = lyz_g - lyz_r   # ln(R_g/R_r)

    return lyz_b, lyz_g, lyz_r, ratio_bg, ratio_gr


def beer_lambert_depth_prior(
    blue:  np.ndarray,
    green: np.ndarray,
    eps:   float = 1e-6,
) -> np.ndarray:
    """Two-band Beer-Lambert analytical depth prior (§2.6).

    Uses a water-pixel surface reference percentile for the green band.

    Physics:
        Kd(490) = 0.0166 + 0.156 · (R_blue / R_green)^{-1.14}
        z_prior = (1 / (2·Kd)) · ln(I_surface / R_green)

    Args:
        blue:  2-D Blue reflectance (B02) — true surface reflectance
        green: 2-D Green reflectance (B03) — true surface reflectance
    Returns:
        z_prior: (H, W) depth estimates in metres (positive)
    """
    b = np.clip(blue,  eps, None)
    g = np.clip(green, eps, None)

    kd = 0.0166 + 0.156 * np.power(b / g, -1.14)
    kd = np.clip(kd, 0.01, 5.0)

    # Surface reference: 95th percentile of green over valid water pixels
    valid_g     = g[g > eps]
    i_surface   = float(np.percentile(valid_g, 95)) if len(valid_g) > 0 else 0.1
    i_surface   = max(i_surface, 1e-3)

    transmission_ratio = np.clip(i_surface / g, 1.0, 1e4)
    z_prior = (1.0 / (2.0 * kd)) * np.log(transmission_ratio)
    return np.nan_to_num(z_prior, nan=0.0, posinf=50.0, neginf=0.0).astype(np.float32)


def distance_to_shore(
    valid_mask: np.ndarray,
    coastline_path: str | None = None,
    transform=None,
    crs: str = None,
) -> np.ndarray | None:
    """Distance to the nearest shore pixel (metres).

    Approximated by the Euclidean distance transform on the water mask.
    Each pixel's distance = distance to the nearest land pixel × 10 m/pixel.
    Returns None if the coastline file does not exist or shapely is missing.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        LOG.warning("scipy not installed — skipping distance-to-shore channel.")
        return None

    # Land = pixels NOT in the valid water mask
    land_mask = ~valid_mask.astype(bool)
    if not land_mask.any():
        return np.full(valid_mask.shape, 9999.0, dtype=np.float32)

    # distance_transform_edt: distances from every False cell to nearest True cell
    # Here True = land, so we get distance from each pixel to nearest land pixel.
    dist_px = distance_transform_edt(~land_mask)   # in pixels
    dist_m  = (dist_px * 10.0).astype(np.float32)  # 10 m resolution
    return dist_m


def load_stack(stack_path: str):
    with rasterio.open(stack_path) as src:
        arrays = {}
        for i in range(1, src.count + 1):
            desc = src.descriptions[i - 1] or f"band_{i}"
            arrays[desc] = src.read(i)
        transform, crs = src.transform, src.crs
    return arrays, transform, crs


# ---------------------------------------------------------------------------
# Per-pixel tabular dataset for Stumpf / RF baselines
# ---------------------------------------------------------------------------

def build_pixel_table(arrays: dict, bands: list) -> pd.DataFrame:
    """Build Lyzenga-featured pixel table including all new channels (§2.1).

    Lyzenga channels are preferred over raw reflectances because they are
    close to linearly proportional to depth in optically shallow water.
    The deep-water offset R∞ is approximated by the 2nd percentile of each
    band over all valid water pixels (same as what the patch dataset uses).
    """
    blue  = arrays[bands[0]]
    green = arrays[bands[1]]
    red   = arrays[bands[2]]
    nir   = arrays[bands[3]] if len(bands) > 3 else None

    water_mask = arrays.get("valid_mask", np.ones_like(blue)) > 0.5

    # Sun-glint correction (§2.3)
    if nir is not None:
        blue, green, red = sun_glint_correction(blue, green, red, nir, water_mask)

    # Deep-water offsets for Lyzenga (from all valid water pixels)
    def _r_inf(band_arr, water_m):
        vals = band_arr[water_m & np.isfinite(band_arr)]
        return float(np.percentile(vals, 2)) if len(vals) > 0 else 0.0

    r_inf_b = _r_inf(blue,  water_mask)
    r_inf_g = _r_inf(green, water_mask)
    r_inf_r = _r_inf(red,   water_mask)

    lyz_b, lyz_g, lyz_r, ratio_bg, ratio_gr = lyzenga_channels(
        blue, green, red, r_inf_b, r_inf_g, r_inf_r
    )
    stumpf_bg  = stumpf_ratio(blue, green)
    stumpf_br  = stumpf_ratio(blue, red)
    z_beer     = beer_lambert_depth_prior(blue, green)
    water_idx  = ndwi(green, nir) if nir is not None else None

    valid = water_mask
    depth = arrays["gebco_depth"]
    is_water = depth < 0
    keep  = valid & is_water & np.isfinite(stumpf_bg) & np.isfinite(depth)

    rows, cols = np.where(keep)
    data = {
        "row": rows, "col": cols,
        "lyzenga_blue":   lyz_b[rows, cols],
        "lyzenga_green":  lyz_g[rows, cols],
        "lyzenga_red":    lyz_r[rows, cols],
        "lyzenga_ratio_bg": ratio_bg[rows, cols],
        "lyzenga_ratio_gr": ratio_gr[rows, cols],
        "stumpf_ratio":       stumpf_bg[rows, cols],
        "stumpf_ratio_red":   stumpf_br[rows, cols],
        "beer_lambert_z_prior": z_beer[rows, cols],
        "depth_m": -depth[rows, cols],
    }
    for b in bands:
        data[b] = arrays[b][rows, cols]
    if water_idx is not None:
        data["ndwi"] = water_idx[rows, cols]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Spatial split for pixel table (unchanged logic, same buffering as patches)
# ---------------------------------------------------------------------------

def spatial_split(
    df: pd.DataFrame,
    n_cols: int,
    n_rows: int,
    train_frac: float,
    val_frac: float,
    axis: str = "lon",
    block_size: int = 64,
    seed: int = 42,
):
    """Depth-stratified spatial block split for the pixel table.

    Blocks that are within one block_size of a block of a different split
    are removed to avoid boundary leakage (§1.5).
    """
    rng = np.random.RandomState(seed)

    df = df.copy()
    df["block_r"] = df["row"] // block_size
    df["block_c"] = df["col"] // block_size
    df["block_id"] = df["block_r"].astype(str) + "_" + df["block_c"].astype(str)

    block_depths  = df.groupby("block_id")["depth_m"].mean().sort_values()
    sorted_blocks = block_depths.index.tolist()

    train_blocks, val_blocks, test_blocks = set(), set(), set()
    for i, b in enumerate(sorted_blocks):
        mod = i % 20
        if   mod < 14: train_blocks.add(b)
        elif mod < 17: val_blocks.add(b)
        else:          test_blocks.add(b)

    train_df = df[df["block_id"].isin(train_blocks)].drop(
        columns=["block_r", "block_c", "block_id"])
    val_df   = df[df["block_id"].isin(val_blocks)].drop(
        columns=["block_r", "block_c", "block_id"])
    test_df  = df[df["block_id"].isin(test_blocks)].drop(
        columns=["block_r", "block_c", "block_id"])
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Patch extraction — now returns full depth maps + richer channels
# ---------------------------------------------------------------------------

def extract_patches(
    arrays: dict,
    bands: list,
    patch_size: int,
    stride: int,
    use_distance_to_shore: bool = False,
    coastline_path: str | None = None,
    transform=None,
    crs: str = None,
):
    """Extract (image, depth_map, valid-mask) patches for CNN / HybridBathNet.

    Channel layout (C channels total):
      0   : B02 Blue  (glint-corrected surface reflectance)
      1   : B03 Green (glint-corrected)
      2   : B04 Red   (glint-corrected)
      3   : B08 NIR
      4   : Lyzenga B02 — ln(R_b − R∞_b)
      5   : Lyzenga B03 — ln(R_g − R∞_g)
      6   : Lyzenga B04 — ln(R_r − R∞_r)
      7   : Lyzenga ratio ln(R_b/R_g)
      8   : Lyzenga ratio ln(R_g/R_r)
      9   : Stumpf Blue/Green ratio
      10  : Beer-Lambert z_prior (metres)
      11  : Water/valid mask (0 or 1 float) — §2.2
      12  : B02 temporal std  — §2.4
      13  : B03 temporal std
      14  : B04 temporal std
      15  : B08 temporal std
     [16] : Distance to shore (metres) — §2.5, optional

    Returns:
        X      : (N, C, H, W) float32
        Y      : (N, H, W)    float32  — full depth map per patch
        M      : (N, H, W)    float32  — valid water mask per patch
        coords : list of (row, col) top-left corners
    """
    valid    = arrays["valid_mask"] > 0.5
    depth    = -arrays["gebco_depth"]              # positive = below sea level
    is_water = arrays["gebco_depth"] < 0

    blue  = arrays[bands[0]]
    green = arrays[bands[1]]
    red   = arrays[bands[2]]
    nir   = arrays[bands[3]] if len(bands) > 3 else None

    water_mask = valid & is_water

    # §2.3  Sun-glint correction
    if nir is not None:
        blue, green, red = sun_glint_correction(blue, green, red, nir, water_mask)

    # §2.1  Lyzenga deep-water reference (2nd percentile over water pixels)
    def _r_inf(band_arr):
        vals = band_arr[water_mask & np.isfinite(band_arr)]
        return float(np.percentile(vals, 2)) if len(vals) > 0 else 0.0

    r_inf_b = _r_inf(blue)
    r_inf_g = _r_inf(green)
    r_inf_r = _r_inf(red)
    LOG.info("Deep-water offsets R∞: B=%.5f  G=%.5f  R=%.5f", r_inf_b, r_inf_g, r_inf_r)

    lyz_b, lyz_g, lyz_r, ratio_bg, ratio_gr = lyzenga_channels(
        blue, green, red, r_inf_b, r_inf_g, r_inf_r
    )
    ratio_stumpf = stumpf_ratio(blue, green)
    z_prior_2d   = beer_lambert_depth_prior(blue, green)

    # §2.2  Mask channel (float 0/1)
    mask_channel = valid.astype(np.float32)

    # §2.4  Temporal std channels (from the composite)
    std_channels = []
    for b in bands:
        key = f"{b}_std"
        if key in arrays:
            std_channels.append(arrays[key])
        else:
            std_channels.append(np.zeros_like(blue))

    # §2.5  Distance to shore (optional)
    dist_channel = None
    if use_distance_to_shore:
        dist_channel = distance_to_shore(valid, coastline_path, transform, crs)
        if dist_channel is not None:
            LOG.info("Distance-to-shore channel: min=%.0f m  max=%.0f m",
                     dist_channel.min(), dist_channel.max())

    # nan-fill derived channels
    for arr in [ratio_stumpf, z_prior_2d, lyz_b, lyz_g, lyz_r, ratio_bg, ratio_gr]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Stack into a single (C, H, W) feature array
    raw_bands  = ([blue, green, red] + ([nir] if nir is not None else []))   # 4 bands
    base_stack = np.stack(
        raw_bands
        + [lyz_b, lyz_g, lyz_r, ratio_bg, ratio_gr]
        + [ratio_stumpf, z_prior_2d, mask_channel]
        + std_channels,
        axis=0
    ).astype(np.float32)   # (C_base, H, W)

    if dist_channel is not None:
        base_stack = np.concatenate(
            [base_stack, dist_channel[None].astype(np.float32)], axis=0
        )

    H, W = valid.shape
    patches_x, patches_y, patches_valid, coords = [], [], [], []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            m = (
                valid[r:r + patch_size, c:c + patch_size]
                & is_water[r:r + patch_size, c:c + patch_size]
            )
            if m.mean() < 0.5:
                continue
            patches_x.append(base_stack[:, r:r + patch_size, c:c + patch_size])
            patches_y.append(depth[r:r + patch_size, c:c + patch_size])
            patches_valid.append(m)
            coords.append((r, c))

    if not patches_x:
        raise RuntimeError(
            "No valid patches extracted — check that the AOI contains "
            "nearshore water pixels with valid GEBCO depth and cloud-free imagery."
        )

    n_channels = base_stack.shape[0]
    LOG.info(
        "Extracted %d patches  |  size: %d×%d  |  channels: %d",
        len(patches_x), patch_size, patch_size, n_channels
    )
    return (
        np.stack(patches_x).astype(np.float32),
        np.stack(patches_y).astype(np.float32),
        np.stack(patches_valid).astype(np.float32),
        coords,
    )


# ---------------------------------------------------------------------------
# Buffered spatial patch split  (§1.5)
# ---------------------------------------------------------------------------

def stratified_spatial_patch_split(
    coords: list,
    Y: np.ndarray,
    train_frac: float = 0.7,
    val_frac:   float = 0.15,
    grid_size:  int   = 128,
    patch_size: int   = 64,
    seed:       int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buffered depth-stratified spatial block split.

    §1.5 fix: patches whose top-left corner is within `patch_size` pixels
    of a block boundary (i.e., the patch overlaps the neighbouring block)
    are dropped to eliminate border leakage.
    """
    rng = np.random.RandomState(seed)

    cy, cx          = Y.shape[1] // 2, Y.shape[2] // 2
    center_depths   = Y[:, cy, cx]

    # Assign each patch to its spatial block
    block_map: dict[tuple, list] = {}
    for idx, (r, c) in enumerate(coords):
        block_id = (r // grid_size, c // grid_size)
        block_map.setdefault(block_id, []).append(idx)

    # Sort blocks by mean depth for depth stratification
    block_ids   = list(block_map.keys())
    block_depths = np.array(
        [np.mean([center_depths[i] for i in block_map[b]]) for b in block_ids]
    )
    sorted_order  = np.argsort(block_depths)
    sorted_blocks = [block_ids[i] for i in sorted_order]

    block_split: dict[tuple, str] = {}
    for i, b in enumerate(sorted_blocks):
        mod = i % 20
        if   mod < 14: block_split[b] = "train"
        elif mod < 17: block_split[b] = "val"
        else:          block_split[b] = "test"

    train_idx, val_idx, test_idx = [], [], []

    for idx, (r, c) in enumerate(coords):
        br, bc = r // grid_size, c // grid_size
        split  = block_split.get((br, bc), "train")

        # §1.5 Buffer check: is the patch corner within `patch_size` of the block edge?
        r_in_block = r % grid_size
        c_in_block = c % grid_size
        near_edge = (
            r_in_block < patch_size
            or r_in_block > grid_size - patch_size
            or c_in_block < patch_size
            or c_in_block > grid_size - patch_size
        )
        if near_edge:
            continue   # drop boundary patches to prevent leakage

        if   split == "train": train_idx.append(idx)
        elif split == "val":   val_idx.append(idx)
        else:                  test_idx.append(idx)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    stack_path = Path(cfg["preprocessing"]["processed_dir"]) / "stack.tif"
    bands      = cfg["sentinel2"]["bands"]
    feat_cfg   = cfg["features"]
    out_dir    = feat_cfg["features_dir"]
    ensure_dir(out_dir)

    use_dist   = feat_cfg.get("use_distance_to_shore", False)
    coast_path = cfg.get("coastline", {}).get("raw_path", None)

    LOG.info("Loading preprocessed stack from %s", stack_path)
    arrays, transform, crs = load_stack(str(stack_path))
    n_rows, n_cols = arrays[bands[0]].shape

    LOG.info("Building per-pixel tabular dataset (for Stumpf/RF baselines — Lyzenga features)...")
    df = build_pixel_table(arrays, bands)
    train_df, val_df, test_df = spatial_split(
        df, n_cols, n_rows, feat_cfg["train_frac"], feat_cfg["val_frac"], feat_cfg["split_axis"]
    )
    train_df.to_csv(Path(out_dir) / "train_pixels.csv", index=False)
    val_df.to_csv(  Path(out_dir) / "val_pixels.csv",   index=False)
    test_df.to_csv( Path(out_dir) / "test_pixels.csv",  index=False)
    LOG.info(
        "Pixel table sizes → train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df)
    )

    patch_size   = feat_cfg["patch_size"]
    patch_stride = feat_cfg["patch_stride"]

    LOG.info("Extracting patches (%d channels)...", "see channel layout in docstring")
    X, Y, M, coords = extract_patches(
        arrays, bands, patch_size, patch_stride,
        use_distance_to_shore=use_dist,
        coastline_path=coast_path,
        transform=transform,
        crs=str(crs),
    )
    LOG.info("Extracted %d patches  |  X: %s  Y: %s", len(X), X.shape, Y.shape)

    train_idx, val_idx, test_idx = stratified_spatial_patch_split(
        coords, Y, feat_cfg["train_frac"], feat_cfg["val_frac"],
        patch_size=patch_size,
    )

    np.savez(
        Path(out_dir) / "patches.npz",
        X=X, Y=Y, M=M,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    LOG.info(
        "Saved depth-stratified spatial patch dataset → train: %d, val: %d, test: %d",
        len(train_idx), len(val_idx), len(test_idx)
    )


if __name__ == "__main__":
    main()
