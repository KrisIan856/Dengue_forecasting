"""
Evaluation module with bootstrap confidence intervals.

Implements proposal §5.9 and §5.10:
  - MAE, RMSE, R² on held-out test years (2020–2021, per-city + pooled)
  - Bootstrap 95% CIs using contiguous weekly blocks within each city
    (block bootstrap to preserve serial dependence)
  - Weather ablation comparison (xgb_case_only vs xgb_full)

Usage:
    from src.evaluation.evaluate import evaluate_all_models
    report = evaluate_all_models(results, splits, X, y, city)
"""

import numpy as np
import pandas as pd
import json
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import BOOTSTRAP_N, BOOTSTRAP_BLOCK_SIZE, OUTPUT_DIR

warnings.filterwarnings("ignore")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    ci: float = 0.95,
    rng_seed: int = 42,
    city: np.ndarray | None = None,
) -> dict:
    """Block bootstrap for 95% CIs on MAE and RMSE.

    Contiguous blocks of `block_size` weeks are resampled with replacement
    *within each city* when `city` is provided, so blocks never straddle
    city boundaries in the pooled panel.
    """
    rng = np.random.default_rng(rng_seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    def _resample_indices(length: int) -> np.ndarray:
        if length < block_size:
            return rng.integers(0, length, size=length)
        n_blocks = length // block_size
        remainder = length % block_size
        max_start = length - block_size
        out = np.zeros(length, dtype=int)
        pos = 0
        for _ in range(n_blocks):
            start = int(rng.integers(0, max_start + 1))
            out[pos:pos + block_size] = start + np.arange(block_size)
            pos += block_size
        if remainder > 0:
            start = int(rng.integers(0, max_start + 1))
            out[pos:pos + remainder] = start + np.arange(remainder)
        return out

    maes = np.zeros(n_bootstrap)
    rmses = np.zeros(n_bootstrap)

    if city is None:
        for b in range(n_bootstrap):
            idx = _resample_indices(n)
            maes[b] = mean_absolute_error(y_true[idx], y_pred[idx])
            rmses[b] = np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))
    else:
        city = np.asarray(city)
        city_slices = {
            c: np.where(city == c)[0] for c in np.unique(city)
        }
        for b in range(n_bootstrap):
            parts = []
            for idxs in city_slices.values():
                local = _resample_indices(len(idxs))
                parts.append(idxs[local])
            idx = np.concatenate(parts)
            maes[b] = mean_absolute_error(y_true[idx], y_pred[idx])
            rmses[b] = np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))

    alpha = 1 - ci
    lo = alpha / 2
    hi = 1 - alpha / 2
    return {
        "MAE_mean": float(np.mean(maes)),
        "MAE_ci": [float(np.percentile(maes, lo * 100)),
                   float(np.percentile(maes, hi * 100))],
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_ci": [float(np.percentile(rmses, lo * 100)),
                   float(np.percentile(rmses, hi * 100))],
    }


def _evaluate_model_on_split(
    result: dict,
    split_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    city: pd.Series,
    splits: dict,
) -> dict:
    """Evaluate one model on a given split (test), per-city and pooled."""
    mask = splits[split_name]
    y_true = y.values[mask]
    y_pred = result["predictions"][split_name]
    city_labels = city.values[mask]

    # Pooled metrics (city-stratified block bootstrap)
    pooled = _metrics(y_true, y_pred)
    pooled_ci = _block_bootstrap_ci(y_true, y_pred, city=city_labels)

    # Per-city metrics
    per_city = {}
    for c in np.unique(city_labels):
        city_mask = city_labels == c
        if city_mask.sum() < 3:
            continue
        ci = (
            _block_bootstrap_ci(y_true[city_mask], y_pred[city_mask])
            if city_mask.sum() >= BOOTSTRAP_BLOCK_SIZE
            else None
        )
        city_metrics = _metrics(y_true[city_mask], y_pred[city_mask])
        if ci:
            city_metrics["ci"] = ci
        per_city[c] = city_metrics

    # Error ratio against persistence (relative MAE)
    rmse = pooled["RMSE"]
    r2 = pooled["R2"]

    return {
        "pooled": pooled,
        "bootstrap_ci": pooled_ci,
        "per_city": per_city,
        "n_samples": int(mask.sum()),
    }


def evaluate_all_models(
    results: list[dict],
    splits: dict,
    X: pd.DataFrame,
    y: pd.Series,
    city: pd.Series,
) -> dict:
    """Full evaluation: test set (2020–2021), all models, ablation comparison.

    Models evaluated:
        - persistence (reference baseline)
        - xgb_case_only (case-history-only XGBoost — ablation baseline)
        - xgb_full (full 28-feature XGBoost — main model)

    The key comparison is the weather ablation: case-only vs full XGBoost
    on identical rows, splits, and tuning protocol. The paired MAE
    difference with its 95% block-bootstrap interval is interpreted as
    predictive contribution of meteorological features.
    """
    print("\n" + "=" * 60)
    print("EVALUATION (held-out test set)")
    print("=" * 60)

    report = {"models": {}, "ablation": {}}

    for result in results:
        name = result["name"]
        print(f"\n  {name}")

        test_eval = _evaluate_model_on_split(result, "test", X, y, city, splits)
        tm = test_eval["pooled"]
        print(f"    Test MAE={tm['MAE']:.1f} "
              f"(CI: {test_eval['bootstrap_ci']['MAE_ci'][0]:.1f}–"
              f"{test_eval['bootstrap_ci']['MAE_ci'][1]:.1f}), "
              f"RMSE={tm['RMSE']:.1f}, R²={tm['R2']:.3f}")

        # Per-city
        for c, m in test_eval["per_city"].items():
            print(f"      {c:25s} MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  R²={m['R2']:.3f}")

        report["models"][name] = {"test": test_eval}

    # ── Weather ablation comparison (proposal §5.10) ────────────────────────────
    print("\n" + "-" * 40)
    print("WEATHER ABLATION (case-only vs full XGBoost)")
    print("-" * 40)

    case_only = report["models"]["xgb_case_only"]["test"]["pooled"]
    full = report["models"]["xgb_full"]["test"]["pooled"]

    case_ci = report["models"]["xgb_case_only"]["test"]["bootstrap_ci"]
    full_ci = report["models"]["xgb_full"]["test"]["bootstrap_ci"]

    improvement_mae = case_only["MAE"] - full["MAE"]
    improvement_rmse = case_only["RMSE"] - full["RMSE"]
    pct_mae = improvement_mae / case_only["MAE"] * 100 if case_only["MAE"] > 0 else 0
    pct_rmse = improvement_rmse / case_only["RMSE"] * 100 if case_only["RMSE"] > 0 else 0

    print(f"  Case-only MAE: {case_only['MAE']:.1f} "
          f"(CI: {case_ci['MAE_ci'][0]:.1f}–{case_ci['MAE_ci'][1]:.1f})")
    print(f"  Full MAE:      {full['MAE']:.1f} "
          f"(CI: {full_ci['MAE_ci'][0]:.1f}–{full_ci['MAE_ci'][1]:.1f})")
    print(f"  MAE improvement: {improvement_mae:.1f} ({pct_mae:+.1f}%)")

    print(f"\n  Case-only RMSE: {case_only['RMSE']:.1f} "
          f"(CI: {case_ci['RMSE_ci'][0]:.1f}–{case_ci['RMSE_ci'][1]:.1f})")
    print(f"  Full RMSE:      {full['RMSE']:.1f} "
          f"(CI: {full_ci['RMSE_ci'][0]:.1f}–{full_ci['RMSE_ci'][1]:.1f})")
    print(f"  RMSE improvement: {improvement_rmse:.1f} ({pct_rmse:+.1f}%)")

    # Relative MAE vs persistence baseline
    persist_mae = report["models"]["persistence"]["test"]["pooled"]["MAE"]
    print(f"\n  Persistence MAE: {persist_mae:.1f}")
    print(f"  Full XGBoost relative MAE (full/persist): {full['MAE']/persist_mae:.3f}")
    print(f"  Case-only XGBoost relative MAE: {case_only['MAE']/persist_mae:.3f}")

    report["ablation"] = {
        "case_only_MAE": case_only["MAE"],
        "full_MAE": full["MAE"],
        "MAE_improvement": improvement_mae,
        "MAE_improvement_pct": pct_mae,
        "case_only_RMSE": case_only["RMSE"],
        "full_RMSE": full["RMSE"],
        "RMSE_improvement": improvement_rmse,
        "RMSE_improvement_pct": pct_rmse,
        "case_only_ci": case_ci,
        "full_ci": full_ci,
        "persistence_MAE": persist_mae,
        "full_relative_mae": float(full["MAE"] / persist_mae) if persist_mae > 0 else float("nan"),
        "case_only_relative_mae": float(case_only["MAE"] / persist_mae) if persist_mae > 0 else float("nan"),
    }

    # Save full report
    report_path = OUTPUT_DIR / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull evaluation report saved → {report_path}")

    return report
