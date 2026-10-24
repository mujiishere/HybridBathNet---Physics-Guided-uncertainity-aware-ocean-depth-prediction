"""
Compare all baseline and HybridBathNet results in one table + plots.

§7 fix: reads hybridbathnet_results.json (not the missing cnn_baseline_results.json)
       and adds a HybridBathNet row to the comparison.
Also reads tabular_baseline_results.json for Stumpf, RF, and GradientBoosting.

Outputs:
    results/baseline_comparison.csv       — full metrics table
    results/baseline_comparison.png       — RMSE bar chart
    results/depth_bin_comparison.png      — per-depth-bin RMSE grouped bar chart
    results/uncertainty_calibration.png   — σ-error scatter + sparsification curve

Usage:
    python -m src.compare_baselines --config config.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import load_config, get_logger

LOG = get_logger("compare_baselines")

# Colour palette
PALETTE = {
    "Stumpf (2003)":        "#4C72B0",
    "Random Forest":         "#55A868",
    "Gradient Boosting":     "#8172B2",
    "HybridBathNet v3":      "#C44E52",
    "CNN Baseline":          "#DD8452",
}


def _safe_load(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    LOG.warning("File not found (skipped): %s", path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    results_dir = Path(cfg["evaluation"]["results_dir"])

    # --- Load all result files ---
    tab  = _safe_load(results_dir / "tabular_baseline_results.json") or {}
    hnet = _safe_load(results_dir / "hybridbathnet_results.json")    or {}
    cnn  = _safe_load(results_dir / "cnn_baseline_results.json")     or {}

    rows = []
    if "stumpf" in tab:
        rows.append({"model": "Stumpf (2003)", **tab["stumpf"]})
    if "random_forest" in tab:
        rows.append({"model": "Random Forest", **tab["random_forest"]})
    if "gradient_boosting" in tab:
        rows.append({"model": "Gradient Boosting", **tab["gradient_boosting"]})
    if cnn and "rmse" in cnn:
        rows.append({"model": "CNN Baseline", **cnn})
    if hnet and "rmse" in hnet:
        rows.append({"model": "HybridBathNet v3", **hnet})

    if not rows:
        LOG.warning("No result files found — run models first.")
        return

    df = pd.DataFrame(rows).set_index("model")
    LOG.info("\n%s", df.to_string())
    df.to_csv(results_dir / "baseline_comparison.csv")

    # ── 1. RMSE bar chart ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    rmse_vals   = df["rmse"].dropna()
    model_names = rmse_vals.index.tolist()
    colors      = [PALETTE.get(m, "#888888") for m in model_names]
    bars = ax.bar(range(len(model_names)), rmse_vals.values, color=colors, edgecolor="white")
    for bar, v in zip(bars, rmse_vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}m",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("RMSE (m)", fontsize=11)
    ax.set_title("Nearshore Depth RMSE — Model Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0, rmse_vals.max() * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(results_dir / "baseline_comparison.png", dpi=150)
    plt.close(fig)

    # ── 2. Per-depth-bin RMSE grouped bar chart ────────────────────────────
    bin_cols = [c for c in df.columns if c.startswith("rmse_") and "m" in c]
    if bin_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        n_models = len(df)
        n_bins   = len(bin_cols)
        w        = 0.8 / n_models
        x_base   = np.arange(n_bins)
        for i, (model_name, row) in enumerate(df.iterrows()):
            vals   = [row.get(c, float("nan")) for c in bin_cols]
            offset = (i - n_models / 2 + 0.5) * w
            color  = PALETTE.get(model_name, "#888888")
            ax.bar(x_base + offset, vals, width=w * 0.9, label=model_name, color=color, alpha=0.85)
        ax.set_xticks(x_base)
        ax.set_xticklabels([c.replace("rmse_", "") for c in bin_cols], fontsize=10)
        ax.set_ylabel("RMSE (m)", fontsize=11)
        ax.set_title("RMSE by Depth Bin", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(results_dir / "depth_bin_comparison.png", dpi=150)
        plt.close(fig)

    # ── 3. Calibration plot (sparsification curve) for HybridBathNet ───────
    if "sparsification" in hnet:
        sparsify = hnet["sparsification"]
        fracs    = [s[0] for s in sparsify]
        rmses    = [s[1] for s in sparsify]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.plot(fracs, rmses, "o-", color=PALETTE.get("HybridBathNet v3", "red"), linewidth=2)
        ax.set_xlabel("Fraction of pixels removed (most uncertain first)")
        ax.set_ylabel("RMSE (m)")
        ax.set_title("Sparsification Curve (well-calibrated → monotone decrease)")
        ax.grid(linestyle="--", alpha=0.4)

        # Coverage summary
        ax = axes[1]
        calib_keys = ["coverage_68pct", "coverage_90pct"]
        calib_labels = ["68% interval", "90% interval"]
        target       = [68.0, 90.0]
        vals_hnet    = [hnet.get(k, float("nan")) for k in calib_keys]
        ax.bar(calib_labels, vals_hnet,  color=PALETTE.get("HybridBathNet v3", "red"),
               label="HybridBathNet v3", alpha=0.8)
        ax.plot(calib_labels, target, "k--", label="Ideal")
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Interval Coverage (ideal = dashed line)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        fig.savefig(results_dir / "uncertainty_calibration.png", dpi=150)
        plt.close(fig)
        LOG.info("Saved calibration plot → %s/uncertainty_calibration.png", results_dir)

    LOG.info("Saved comparison table → %s/baseline_comparison.csv", results_dir)
    LOG.info("Saved RMSE bar chart   → %s/baseline_comparison.png",  results_dir)


if __name__ == "__main__":
    main()
