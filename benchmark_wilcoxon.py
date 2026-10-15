"""
Statistical Benchmarking — Wilcoxon Signed-Rank Test

Project: Forecasting Weekly Dengue Cases in Mindanao Cities (Bandiola & Balani)

Benchmarks the models across 10 random seeds:
  - Proposed: xgb_full (Full XGBoost — 28 features incl. weather)
  - Baseline: xgb_case_only (ablation: 6 case/seasonality features)
  - Reference: persistence (naive baseline)

One-tailed Wilcoxon signed-rank test (α = 0.05).
Comparison tables: mean ± std, W, p-value, % gain.
Runtime benchmarking + per-city tests + plots.

Models trained: 2016–2019 | Test: 2020–2021 (70/30 chronological split)
"""

import sys, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PROJECT = Path(r"C:\Users\princ\miniconda3\envs\forecasting")
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

from config import XGB_PARAMS, ALL_FEATURES, CASE_ONLY_FEATURES
from src.features.build_features import build_feature_matrix

print("Project:", PROJECT)
print("Full features:", len(ALL_FEATURES), "| Case-only:", len(CASE_ONLY_FEATURES))

# ============================================================
# Load data & build feature matrix
# ============================================================
try:
    from src.data_loader import load_real_panel
    df = load_real_panel()
except Exception as exc:
    print(f"Real panel unavailable ({exc}); generating synthetic data.")
    from data.generate_synthetic_data import generate_panel, add_derived_weather
    df = add_derived_weather(generate_panel())

X, y, city, splits, dates = build_feature_matrix(df)
print(f"X={X.shape}  train={splits['train'].sum()}  test={splits['test'].sum()}")

# ============================================================
# Benchmark runner: 3 models × 10 seeds
# ============================================================
N_RUNS = 10
SEEDS = list(range(42, 42 + N_RUNS))
MODEL_NAMES = ["persistence", "xgb_case_only", "xgb_full"]
BASELINES = ["persistence", "xgb_case_only"]

test_idx = np.where(splits["test"])[0]
y_test = y[splits["test"]].values
city_test = city.values[test_idx]


def _metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


from src.models.models import ResidualXGBRegressor
from config import XGB_PARAMS, SHRINKAGE, ALL_FEATURES, CASE_ONLY_FEATURES


def train_one(name, seed):
    t0 = time.time()
    if name == "persistence":
        pred_test = X.loc[splits["test"], "cases_lag1"].values.astype(float)
    else:
        if name == "xgb_case_only":
            feats = CASE_ONLY_FEATURES
            params = {**XGB_PARAMS, "random_state": seed}
        elif name == "xgb_full":
            feats = ALL_FEATURES
            params = {**XGB_PARAMS, "random_state": seed}
        else:
            raise ValueError(name)
        mdl = ResidualXGBRegressor(feature_set=feats, shrinkage=SHRINKAGE, **params)
        mdl.fit(X.loc[splits["train"]], y[splits["train"]])
        pred_test = mdl.predict(X.loc[splits["test"]])
    runtime = time.time() - t0
    return {
        "seed": seed, "runtime_sec": runtime,
        "test": _metrics(y_test, pred_test),
        "pred_test": pred_test,
    }


print(f"Running {len(MODEL_NAMES)} models x {N_RUNS} seeds ...")
raw_results = {m: [] for m in MODEL_NAMES}
for m in MODEL_NAMES:
    for s in SEEDS:
        raw_results[m].append(train_one(m, s))
    print(f"  done: {m}")

# ============================================================
# Tidy DataFrame + descriptive stats
# ============================================================
rows = []
for m in MODEL_NAMES:
    for r in raw_results[m]:
        rows.append({
            "model": m, "seed": r["seed"], "runtime_sec": r["runtime_sec"],
            "test_MAE": r["test"]["MAE"], "test_RMSE": r["test"]["RMSE"],
            "test_R2": r["test"]["R2"],
        })
bench_df = pd.DataFrame(rows)

desc = bench_df.groupby("model").agg(
    MAE_mean=("test_MAE", "mean"), MAE_std=("test_MAE", "std"),
    RMSE_mean=("test_RMSE", "mean"), RMSE_std=("test_RMSE", "std"),
    R2_mean=("test_R2", "mean"), R2_std=("test_R2", "std"),
    time_mean=("runtime_sec", "mean"), time_std=("runtime_sec", "std"),
).round(4)
print("=== Mean ± Std across 10 seeds (2020–2021 Test Year) ===")
print(desc.to_string())

bench_df.to_csv("outputs/benchmark_runs.csv", index=False)
print("\nSaved: outputs/benchmark_runs.csv")


def extract(model, metric):
    col = "runtime_sec" if metric == "runtime_sec" else f"test_{metric}"
    return bench_df.loc[bench_df["model"] == model, col].values


# ============================================================
# Wilcoxon comparison function
# ============================================================
def compare_and_print_table(proposed, baseline, metric_name,
                            proposed_label="Full XGBoost",
                            baseline_label="Baseline",
                            higher_is_better=False, alpha=0.05):
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    if higher_is_better:
        gain = proposed - baseline
        alternative = "greater"
    else:
        gain = baseline - proposed
        alternative = "less"

    df_cmp = pd.DataFrame({
        "Run": range(1, len(proposed) + 1),
        proposed_label: proposed.round(4),
        baseline_label: baseline.round(4),
        "Gain (better if +)": gain.round(4),
    })
    print(f"\n===== {metric_name} Comparison =====")
    print(f"Proposed : {proposed_label}   |   Baseline : {baseline_label}")
    print(df_cmp.to_string(index=False))

    try:
        stat, p_value = wilcoxon(proposed, baseline, alternative=alternative)
    except ValueError:
        stat, p_value = 0.0, 1.0

    direction = "Proposed > Baseline" if higher_is_better else "Proposed < Baseline"
    print(f"\nWilcoxon Signed-Rank Test (One-Tailed: {direction})")
    print(f"Statistic (W): {stat}")
    print(f"P-value      : {p_value:.4f}")
    if p_value < alpha:
        print(f"Result: Significant improvement at alpha={alpha}")
    else:
        print(f"Result: No statistically significant difference at alpha={alpha}")

    return {
        "metric": metric_name,
        "baseline_label": baseline_label,
        "W": float(stat), "p": float(p_value),
        "significant": bool(p_value < alpha),
        "mean_proposed": float(np.mean(proposed)),
        "std_proposed": float(np.std(proposed, ddof=1)),
        "mean_baseline": float(np.mean(baseline)),
        "std_baseline": float(np.std(baseline, ddof=1)),
        "gain_pct": float(np.mean(gain) / np.mean(baseline) * 100)
                    if np.mean(baseline) != 0 else float("nan"),
    }


# ============================================================
# MAE benchmark (lower is better)
# ============================================================
PROPOSED = "xgb_full"
mae_results = [
    compare_and_print_table(
        extract(PROPOSED, "MAE"), extract(b, "MAE"),
        metric_name="Test MAE (2020–2021)",
        proposed_label="Full XGBoost", baseline_label=b,
        higher_is_better=False,
    ) for b in BASELINES
]

# ============================================================
# RMSE benchmark (lower is better)
# ============================================================
rmse_results = [
    compare_and_print_table(
        extract(PROPOSED, "RMSE"), extract(b, "RMSE"),
        metric_name="Test RMSE (2020–2021)",
        proposed_label="Full XGBoost", baseline_label=b,
        higher_is_better=False,
    ) for b in BASELINES
]

# ============================================================
# R² benchmark (higher is better)
# ============================================================
r2_results = [
    compare_and_print_table(
        extract(PROPOSED, "R2"), extract(b, "R2"),
        metric_name="Test R² (2020–2021)",
        proposed_label="Full XGBoost", baseline_label=b,
        higher_is_better=True,
    ) for b in BASELINES
]

# ============================================================
# Runtime benchmark (lower is better)
# ============================================================
time_results = [
    compare_and_print_table(
        extract(PROPOSED, "runtime_sec"), extract(b, "runtime_sec"),
        metric_name="Training Runtime (seconds)",
        proposed_label="Full XGBoost", baseline_label=b,
        higher_is_better=False,
    ) for b in BASELINES
]

# ============================================================
# Master summary table
# ============================================================
def significance_stars(p):
    if p < 0.01:
        return "YES (p<0.01)"
    elif p < 0.05:
        return "YES (p<0.05)"
    return "no (n.s.)"


ALL_GROUPS = [
    ("Test MAE (2020–2021)", mae_results),
    ("Test RMSE (2020–2021)", rmse_results),
    ("Test R² (2020–2021)", r2_results),
    ("Runtime (s)", time_results),
]

rows = []
idx = 1
for group_name, group in ALL_GROUPS:
    for res in group:
        rows.append({
            "#": idx,
            "Metric": group_name,
            "Baseline Model": res["baseline_label"],
            "Baseline (mean ± std)": f"{res['mean_baseline']:.4f} ± {res['std_baseline']:.4f}",
            "Full XGB (mean ± std)": f"{res['mean_proposed']:.4f} ± {res['std_proposed']:.4f}",
            "Gain": f"{res['gain_pct']:+.2f}%",
            "W": res["W"],
            "p-Value": f"{res['p']:.4f}",
            "Significant?": significance_stars(res["p"]),
        })
        idx += 1

summary_tbl = pd.DataFrame(rows)
print("=" * 110)
print("TABLE: STATISTICAL BENCHMARK — Full XGBoost (Proposed) vs Baselines")
print("One-Tailed Wilcoxon Signed-Rank Test | alpha = 0.05 | 10 seeds (42-51)")
print("=" * 110)
print(summary_tbl.to_string(index=False))

summary_tbl.to_csv("outputs/wilcoxon_benchmark_summary.csv", index=False)
print("\nSaved: outputs/wilcoxon_benchmark_summary.csv")

# ============================================================
# Per-city benchmark + Wilcoxon
# ============================================================
cities_unique = sorted(set(city_test))
per_city_records = []
for m in ["persistence", "xgb_case_only", "xgb_full"]:
    for s in SEEDS:
        if m == "persistence":
            pred = X.loc[splits["test"], "cases_lag1"].values.astype(float)
        else:
            feats = CASE_ONLY_FEATURES if m == "xgb_case_only" else ALL_FEATURES
            params = {**XGB_PARAMS, "random_state": s}
            mdl = ResidualXGBRegressor(feature_set=feats, shrinkage=SHRINKAGE, **params)
            mdl.fit(X.loc[splits["train"]], y[splits["train"]])
            pred = mdl.predict(X.loc[splits["test"]])
        for c in cities_unique:
            cm = city_test == c
            per_city_records.append({
                "model": m, "seed": s, "city": c,
                "MAE": mean_absolute_error(y_test[cm], pred[cm]),
            })

per_city_df = pd.DataFrame(per_city_records)
print("=" * 80)
print("PER-CITY MAE — Mean ± Std across 10 seeds (2020–2021 Test)")
print("=" * 80)
print(per_city_df.groupby(["city", "model"])["MAE"].agg(["mean", "std"]).round(3).to_string())

print("\n" + "=" * 80)
print("PER-CITY WILCOXON TESTS (Full XGB vs Case-only XGB, MAE, 10 seeds)")
print("=" * 80)
city_rows = []
for c in cities_unique:
    full_c = per_city_df[(per_city_df["city"] == c) &
                         (per_city_df["model"] == "xgb_full")]["MAE"].values
    case_c = per_city_df[(per_city_df["city"] == c) &
                         (per_city_df["model"] == "xgb_case_only")]["MAE"].values
    try:
        stat, p = wilcoxon(full_c, case_c, alternative="less")
    except ValueError:
        stat, p = 0.0, 1.0
    city_rows.append({
        "City": c,
        "Case-only MAE (mean±std)": f"{case_c.mean():.2f} ± {case_c.std(ddof=1):.2f}",
        "Full XGB MAE (mean±std)": f"{full_c.mean():.2f} ± {full_c.std(ddof=1):.2f}",
        "W": stat, "p": round(p, 4),
        "Significant?": significance_stars(p),
    })

city_wilcoxon_df = pd.DataFrame(city_rows)
print(city_wilcoxon_df.to_string(index=False))
city_wilcoxon_df.to_csv("outputs/per_city_wilcoxon.csv", index=False)
print("\nSaved: outputs/per_city_wilcoxon.csv")

# ============================================================
# Visualizations
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ["#4e79a7", "#f28e2b", "#e15759"]

mae_data = [extract(m, "MAE") for m in MODEL_NAMES]
bp = axes[0].boxplot(mae_data, tick_labels=MODEL_NAMES, patch_artist=True)
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
axes[0].set_title("Test MAE across 10 Seeds (2020–2021)")
axes[0].set_ylabel("MAE (cases)")
axes[0].tick_params(axis="x", rotation=15)
axes[0].grid(True, alpha=0.3, axis="y")

time_data = [bench_df.loc[bench_df["model"] == m, "runtime_sec"].values for m in MODEL_NAMES]
bp2 = axes[1].boxplot(time_data, tick_labels=MODEL_NAMES, patch_artist=True)
for patch, col in zip(bp2["boxes"], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
axes[1].set_title("Training Runtime across 10 Seeds")
axes[1].set_ylabel("Seconds")
axes[1].tick_params(axis="x", rotation=15)
axes[1].grid(True, alpha=0.3, axis="y")

pvals = [r["p"] for r in mae_results]
bars = axes[2].bar(BASELINES, pvals, color=colors[1:])
axes[2].axhline(0.05, color="red", linestyle="--", label="alpha = 0.05")
axes[2].set_title("Wilcoxon p-values — Full XGB vs Baselines (MAE)")
axes[2].set_ylabel("p-value")
axes[2].set_ylim(0, max(1.0, max(pvals) * 1.3))
axes[2].legend()
axes[2].tick_params(axis="x", rotation=15)
for bar, pv in zip(bars, pvals):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{pv:.4f}", ha="center", fontsize=9)
axes[2].grid(True, alpha=0.3, axis="y")

plt.suptitle("Statistical Benchmark — Full XGBoost vs Baselines (10 Seeds)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/wilcoxon_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/figures/wilcoxon_benchmark.png")

# ============================================================
# Auto-generated interpretation
# ============================================================
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(f"""
One-tailed Wilcoxon signed-rank tests (alpha = 0.05) were conducted across
{N_RUNS} independent training runs (seeds 42-51) comparing the proposed
Full XGBoost model against each baseline on the held-out 2020–2021 test year.

Key findings (Test MAE, 2020–2021):
""")
for res in mae_results:
    verdict = ("statistically significant improvement"
               if res["significant"] else "no statistically significant difference")
    print(f"  * Full XGB vs {res['baseline_label']:>14s}:  "
          f"p = {res['p']:.4f}  ->  {verdict}")

print(f"""
Gain (%) is the average relative improvement of Full XGBoost over the
baseline; positive values mean the proposed model is better. A significant
result (p < 0.05) indicates the observed difference is unlikely under the
null hypothesis of no difference.
""")
