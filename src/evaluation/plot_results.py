"""
Visualization module — generates plots for the results.

Creates:
  1. Actual vs predicted time series per model (test set), one line per city
  2. Per-city MAE comparison bar chart
  3. Ablation comparison (case-only vs full XGBoost)

Usage:
    from src.evaluation.plot_results import plot_all
    plot_all(results, splits, X, y, city, report, dates)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from config import OUTPUT_DIR

CITY_COLORS = {
    "Davao City": "#4e79a7",
    "Cagayan de Oro City": "#f28e2b",
    "Zamboanga City": "#59a14f",
}


def plot_actual_vs_predicted(
    results: list[dict],
    splits: dict,
    y: pd.Series,
    city: pd.Series,
    dates: pd.Series,
) -> Path:
    """Plot actual vs predicted dengue cases for each model on the test set.

    One panel per model. Within a panel each city is drawn with the observed
    series as a solid line and the model's forecast as a dashed line, on a
    real calendar x-axis.
    """
    test_mask = splits["test"]
    y_true_all = y.values[test_mask]
    city_all = np.asarray(city.values)[test_mask]
    date_all = pd.to_datetime(pd.Series(np.asarray(dates))[test_mask]).values

    n_models = len(results)
    ncols = min(2, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows),
                             squeeze=False)
    flat_axes = axes.flatten()

    for i, result in enumerate(results):
        ax = flat_axes[i]
        pred = np.asarray(result["predictions"]["test"])

        for c in sorted(pd.unique(city_all)):
            cm = city_all == c
            color = CITY_COLORS.get(c, None)
            ax.plot(date_all[cm], y_true_all[cm], "-", color=color, linewidth=1.8,
                    label=f"{c} — actual")
            ax.plot(date_all[cm], pred[cm], "--", color=color, linewidth=1.4,
                    alpha=0.85, label=f"{c} — predicted")

        mae = result["test_metrics"]["MAE"]
        ax.set_title(f"{result['name']}  (MAE = {mae:.2f})", fontsize=12)
        ax.set_xlabel("Forecast-origin date")
        ax.set_ylabel("Dengue cases at t+2")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

    for j in range(n_models, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.suptitle("Actual vs Predicted Dengue Cases (2020–2021 Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "figures" / "actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_per_city_mae(
    report: dict,
    split: str = "test",
) -> Path:
    """Bar chart of MAE by city for each model."""
    models = list(report["models"].keys())
    all_cities = set()
    for m in models:
        all_cities.update(report["models"][m][split]["per_city"].keys())
    all_cities = sorted(all_cities)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_cities))
    width = 0.8 / max(len(models), 1)
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    for i, model_name in enumerate(models):
        maes = []
        for c in all_cities:
            city_data = report["models"][model_name][split]["per_city"].get(c, {})
            maes.append(city_data.get("MAE", 0))
        ax.bar(x + i * width, maes, width, label=model_name,
               color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(all_cities, rotation=12)
    ax.set_ylabel("MAE (cases)")
    ax.set_title("Per-City MAE — 2020–2021 Test Year")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = OUTPUT_DIR / "figures" / "per_city_mae.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_ablation(report: dict) -> Path:
    """Plot the weather ablation comparison (MAE with bootstrap CIs)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    abl = report["ablation"]

    models = ["Case-only XGBoost", "Full XGBoost"]
    maes = [abl["case_only_MAE"], abl["full_MAE"]]
    ci_lo = [abl["case_only_ci"]["MAE_ci"][0], abl["full_ci"]["MAE_ci"][0]]
    ci_hi = [abl["case_only_ci"]["MAE_ci"][1], abl["full_ci"]["MAE_ci"][1]]
    err_lo = [m - lo for m, lo in zip(maes, ci_lo)]
    err_hi = [hi - m for hi, m in zip(ci_hi, maes)]

    bars = ax.bar(models, maes, color=["#4e79a7", "#f28e2b"], width=0.5,
                  yerr=[err_lo, err_hi], capsize=5, alpha=0.85)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"MAE={mae:.1f}", ha="center", fontsize=10)

    ax.set_ylabel("MAE (cases)")
    ax.set_title("Weather Ablation: Case-only vs Full Weather Features\n"
                 "(2020–2021 Test Year, 95% bootstrap CI)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = OUTPUT_DIR / "figures" / "ablation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_all(results: list[dict], splits: dict, X: pd.DataFrame,
             y: pd.Series, city: pd.Series, report: dict,
             dates: pd.Series | None = None):
    """Generate all plots."""
    print("\nGenerating plots...")
    if dates is None:
        raise ValueError("plot_all requires the `dates` series from build_feature_matrix")
    plot_actual_vs_predicted(results, splits, y, city, dates)
    plot_per_city_mae(report, split="test")
    plot_ablation(report)
    print("All plots generated.")
