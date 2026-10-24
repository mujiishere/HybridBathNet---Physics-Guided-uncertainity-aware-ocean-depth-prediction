"""
End-to-end pipeline orchestrator for HybridBathNet.

§7 fixes applied:
  - HybridBathNet training is now an explicit stage (was missing).
  - --synthetic flag now calls src.generate_synthetic_data instead of
    being silently ignored.
  - Stages labelled clearly; compare_baselines reads the correct files.

Usage:
    python run_baseline_pipeline.py --config config.yaml
    python run_baseline_pipeline.py --config config.yaml --synthetic
    python run_baseline_pipeline.py --config config.yaml --skip-download
    python run_baseline_pipeline.py --config config.yaml --seed 0 --seed 1 ...
"""
import argparse
import subprocess
import sys


def run(cmd: list, stage_name: str = ""):
    label = stage_name or " ".join(cmd[-2:])
    print(f"\n{'='*60}")
    print(f"  STAGE: {label}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n  ✓ {label} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ {label} FAILED with exit code {e.returncode}.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="HybridBathNet end-to-end training pipeline."
    )
    parser.add_argument("--config",          default="config.yaml")
    parser.add_argument("--skip-download",   action="store_true",
                        help="Skip data download (use existing data/raw/*)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (use existing data/processed/stack.tif)")
    parser.add_argument("--synthetic",       action="store_true",
                        help="Generate synthetic Sentinel-2+GEBCO data instead of downloading")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Random seed for training (run multiple seeds for §8 ablations)")
    parser.add_argument("--no-amp",          action="store_true",
                        help="Disable AMP (useful for debugging)")
    parser.add_argument("--no-rotations",    action="store_true",
                        help="Disable 90° rotation augmentation (ablation §3.8)")
    args = parser.parse_args()

    py = sys.executable

    # ── Data acquisition ──────────────────────────────────────────────────
    if args.synthetic:
        # §7: --synthetic is now wired to the actual synthetic data generator
        run([py, "-m", "src.generate_synthetic_data", "--config", args.config],
            "Generate synthetic Sentinel-2 + GEBCO data")
    elif not args.skip_download:
        run([py, "-m", "src.download_sentinel2",  "--config", args.config],
            "Download Sentinel-2 (real tiles from CDSE)")
        run([py, "-m", "src.download_gebco",      "--config", args.config],
            "Download + convert GEBCO (NOAA ERDDAP)")
        run([py, "-m", "src.download_coastline",  "--config", args.config],
            "Download OSM coastline")

    # ── Preprocessing ─────────────────────────────────────────────────────
    if not args.skip_preprocess:
        run([py, "-m", "src.preprocess", "--config", args.config],
            "Preprocess: BOA offset, median composite, temporal std")

    # ── Feature engineering ───────────────────────────────────────────────
    run([py, "-m", "src.features", "--config", args.config],
        "Feature engineering: Lyzenga channels, glint correction, mask, split")

    # ── Tabular baselines ─────────────────────────────────────────────────
    run([py, "-m", "src.baseline_models", "--config", args.config],
        "Tabular baselines: Stumpf (Lyzenga), RandomForest, GradientBoosting")

    # ── HybridBathNet v3 ──────────────────────────────────────────────────
    train_cmd = [py, "-m", "src.cnn_baseline", "--config", args.config,
                 "--seed", str(args.seed)]
    if args.no_amp:
        train_cmd.append("--no-amp")
    if args.no_rotations:
        train_cmd.append("--no-rotations")
    run(train_cmd, f"HybridBathNet v3 training (seed={args.seed})")

    # ── Comparison & plots ────────────────────────────────────────────────
    run([py, "-m", "src.compare_baselines", "--config", args.config],
        "Compare all baselines + generate plots")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("  Results:  results/baseline_comparison.csv  + .png")
    print("            results/depth_bin_comparison.png")
    print("            results/uncertainty_calibration.png")
    print("            results/hybridbathnet_results.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
