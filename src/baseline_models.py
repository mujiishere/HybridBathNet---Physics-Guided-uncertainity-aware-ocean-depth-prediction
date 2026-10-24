"""
Tabular baselines: Stumpf (2003) and Random Forest, now using the same
Lyzenga-feature pixel table that the CNN sees  (§8 — give baselines the
same features so the comparison is fair).

Improvements:
  §8    Same Lyzenga features as HybridBathNet input.
  §1.6  Diagnosis of split depth distribution printed at startup (check for
        distribution mismatch that would cause spurious negative R²).
  §8    Paired bootstrap 95% CI on RMSE difference between any two models.
  Depth-bin reporting: RMSE per 0-5 / 5-10 / 10-15 / 15-20 / >20 m bin.

Usage:
    python -m src.baseline_models --config config.yaml
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("baseline_models")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def depth_bin_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-depth-bin RMSE (§8 ablation)."""
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 9999)]
    out  = {}
    for lo, hi in bins:
        label = f"{lo}-{hi}m" if hi < 9999 else f">{lo}m"
        mask  = (y_true >= lo) & (y_true < hi)
        if mask.sum() < 10:
            out[f"rmse_{label}"] = float("nan")
        else:
            out[f"rmse_{label}"] = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    abs_diff = np.abs(y_p - y_t)
    rmse     = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
    mae      = float(np.mean(abs_diff))
    ss_res   = np.sum((y_t - y_p) ** 2)
    ss_tot   = np.sum((y_t - y_t.mean()) ** 2)
    r2       = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    acc_1m   = float(np.mean(abs_diff <= 1.0) * 100.0)
    acc_2m   = float(np.mean(abs_diff <= 2.0) * 100.0)
    pos      = (y_t > 0.1) & (y_p > 0.1)
    delta_1  = float(
        np.mean(np.maximum(y_t[pos] / y_p[pos], y_p[pos] / y_t[pos]) < 1.25) * 100.0
    ) if np.any(pos) else float("nan")

    out = {
        "rmse": rmse, "mae": mae, "r2": r2,
        "acc_within_1m_%": acc_1m, "acc_within_2m_%": acc_2m,
        "delta_1.25_%": delta_1,
    }
    out.update(depth_bin_metrics(y_t, y_p))
    return out


def bootstrap_rmse_ci(
    y_true:     np.ndarray,
    pred_a:     np.ndarray,
    pred_b:     np.ndarray,
    n_boot:     int   = 1000,
    alpha:      float = 0.05,
    seed:       int   = 42,
) -> dict:
    """Paired bootstrap 95% CI on RMSE(A) − RMSE(B).

    A negative delta means model A is worse (higher RMSE) than B.
    """
    rng     = np.random.RandomState(seed)
    n       = len(y_true)
    deltas  = []
    for _ in range(n_boot):
        idx  = rng.choice(n, size=n, replace=True)
        rmse_a = np.sqrt(np.mean((pred_a[idx] - y_true[idx]) ** 2))
        rmse_b = np.sqrt(np.mean((pred_b[idx] - y_true[idx]) ** 2))
        deltas.append(rmse_a - rmse_b)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {
        "delta_rmse_mean": float(np.mean(deltas)),
        "delta_rmse_lo":   float(lo),
        "delta_rmse_hi":   float(hi),
    }


def check_split_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """§1.6 — Print depth distribution stats for both splits.  A large
    discrepancy (e.g. test is much deeper) explains negative R²."""
    for name, df in [("train", train_df), ("test", test_df)]:
        d = df["depth_m"]
        LOG.info(
            "Split %-5s  n=%6d  depth: mean=%.1f m  std=%.1f m  "
            "min=%.1f m  p25=%.1f m  p75=%.1f m  max=%.1f m",
            name, len(d), d.mean(), d.std(),
            d.min(), d.quantile(0.25), d.quantile(0.75), d.max()
        )


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def lyzenga_feature_cols(df: pd.DataFrame, bands: list) -> list:
    """Return all Lyzenga + physics feature columns present in the dataframe."""
    candidates = (
        [f"lyzenga_{b.lower()}" for b in ["blue", "green", "red"]]
        + ["lyzenga_ratio_bg", "lyzenga_ratio_gr"]
        + ["stumpf_ratio", "stumpf_ratio_red", "beer_lambert_z_prior"]
        + bands
        + ["ndwi"]
    )
    return [c for c in candidates if c in df.columns]


def train_stumpf(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Stumpf (2003): linear regression on Lyzenga ratio columns.

    §8 upgrade: uses all Lyzenga band-ratio features instead of only the
    raw Stumpf ratio, so the baseline is as strong as the physics allows.
    """
    lyzenga_cols = [c for c in ["lyzenga_ratio_bg", "lyzenga_ratio_gr", "stumpf_ratio"]
                    if c in train_df.columns]
    if not lyzenga_cols:
        lyzenga_cols = ["stumpf_ratio"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    model.fit(train_df[lyzenga_cols], train_df["depth_m"])
    preds = model.predict(test_df[lyzenga_cols])
    m = metrics(test_df["depth_m"].values, preds)
    LOG.info("Stumpf/Lyzenga linear  → RMSE %.3f m  MAE %.3f m  R² %.3f",
             m["rmse"], m["mae"], m["r2"])
    return model, m, preds


def train_random_forest(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list
) -> tuple:
    max_samples = 100_000 if len(train_df) > 100_000 else None
    model = RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=3,
        max_features="sqrt", max_samples=max_samples,
        n_jobs=-1, random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["depth_m"])
    preds = model.predict(test_df[feature_cols])
    m = metrics(test_df["depth_m"].values, preds)
    LOG.info("RandomForest           → RMSE %.3f m  MAE %.3f m  R² %.3f",
             m["rmse"], m["mae"], m["r2"])
    return model, m, preds


def train_gradient_boosting(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list
) -> tuple:
    """GradientBoosting as a stronger tabular baseline."""
    n_sub  = min(len(train_df), 200_000)
    idx    = np.random.RandomState(42).choice(len(train_df), n_sub, replace=False)
    sub_df = train_df.iloc[idx]

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42,
    )
    model.fit(sub_df[feature_cols], sub_df["depth_m"])
    preds = model.predict(test_df[feature_cols])
    m = metrics(test_df["depth_m"].values, preds)
    LOG.info("GradientBoosting       → RMSE %.3f m  MAE %.3f m  R² %.3f",
             m["rmse"], m["mae"], m["r2"])
    return model, m, preds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    feat_dir    = cfg["features"]["features_dir"]
    results_dir = cfg["evaluation"]["results_dir"]
    ensure_dir(results_dir)
    ensure_dir(cfg["training"]["checkpoint_dir"])

    train_df = pd.read_csv(Path(feat_dir) / "train_pixels.csv")
    test_df  = pd.read_csv(Path(feat_dir) / "test_pixels.csv")
    LOG.info("Loaded pixel tables — train: %d rows  test: %d rows", len(train_df), len(test_df))

    # §1.6 — check split distributions
    check_split_distribution(train_df, test_df)

    bands        = cfg["sentinel2"]["bands"]
    feature_cols = lyzenga_feature_cols(train_df, bands)
    LOG.info("Feature columns for tabular models: %s", feature_cols)

    # Train models
    stumpf_model, stumpf_metrics, stumpf_preds = train_stumpf(train_df, test_df)
    rf_model,     rf_metrics,     rf_preds     = train_random_forest(train_df, test_df, feature_cols)
    gb_model,     gb_metrics,     gb_preds     = train_gradient_boosting(train_df, test_df, feature_cols)

    y_test = test_df["depth_m"].values

    # §8 — paired bootstrap confidence intervals
    boot_stumpf_rf = bootstrap_rmse_ci(y_test, stumpf_preds, rf_preds)
    boot_rf_gb     = bootstrap_rmse_ci(y_test, rf_preds,     gb_preds)
    LOG.info("Bootstrap CI  Stumpf vs RF:          ΔRMSE=%.3f [%.3f, %.3f]",
             boot_stumpf_rf["delta_rmse_mean"],
             boot_stumpf_rf["delta_rmse_lo"],
             boot_stumpf_rf["delta_rmse_hi"])
    LOG.info("Bootstrap CI  RF vs GradientBoosting: ΔRMSE=%.3f [%.3f, %.3f]",
             boot_rf_gb["delta_rmse_mean"],
             boot_rf_gb["delta_rmse_lo"],
             boot_rf_gb["delta_rmse_hi"])

    # Save model artefacts
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    joblib.dump(stumpf_model, Path(ckpt_dir) / "stumpf_model.joblib")
    joblib.dump(rf_model,     Path(ckpt_dir) / "rf_model.joblib")
    joblib.dump(gb_model,     Path(ckpt_dir) / "gb_model.joblib")

    results = {
        "stumpf":              stumpf_metrics,
        "random_forest":       rf_metrics,
        "gradient_boosting":   gb_metrics,
        "bootstrap_stumpf_rf": boot_stumpf_rf,
        "bootstrap_rf_gb":     boot_rf_gb,
    }
    out_path = Path(results_dir) / "tabular_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Saved results → %s", out_path)


if __name__ == "__main__":
    main()
