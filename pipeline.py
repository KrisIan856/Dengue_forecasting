"""
Main pipeline — ties everything together.

Implements proposal §5.11 (Implementation Pipeline):
  Step 1. Load   → Step 2. Validate → Step 3. Order → Step 4. Engineer
  → Step 5. Split → Step 6. Train → Step 7. Evaluate → Step 8. Report

  Updated temporal design: train 2016–2019 (~70%), test 2020–2021 (~30%),
  with a 2-week embargo between train and test.

  Models: persistence baseline + case-only XGBoost + full XGBoost
  (weather ablation — the core experiment). Random Forest removed per
  updated proposal; focus is XGBoost-only.

Usage:
    python pipeline.py            # runs the full pipeline with real data
    python pipeline.py --download # placeholder for real data download step
"""

import sys
import json
import warnings
from pathlib import Path

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

# Make console output safe on Windows terminals that default to cp1252,
# where characters like the arrow and en-dash would otherwise raise
# UnicodeEncodeError mid-run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def step1_load() -> pd.DataFrame:
    """Step 1: Load the data.

    Priority order:
      1. Real per-city panels in <env>/datasets/ (merged on the fly, then
         cached to data/merged_dengue_panel_real.csv).
      2. A previously cached merged real panel.
      3. The legacy raw panel, if present.
      4. Synthetic generation as a last-resort fallback.
    """
    print("=" * 60)
    print("STEP 1: Load")
    print("=" * 60)

    from config import RAW_DATA, MERGED_REAL_DATA, REAL_DATA_DIR, USE_REAL_DATA
    if USE_REAL_DATA:
        from src.data_loader import load_real_panel, _find_city_files
        try:
            _find_city_files()
            print(f"  Real per-city panels found in {REAL_DATA_DIR.parent.name}/datasets")
            df = load_real_panel()
            df.to_csv(MERGED_REAL_DATA, index=False)
            print(f"  Cached merged panel -> {MERGED_REAL_DATA}")
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"  Cities: {df['city'].unique().tolist()}")
            print(f"  Year range: {df['year'].min()}–{df['year'].max()}")
            return df
        except (FileNotFoundError, ValueError) as exc:
            print(f"  Real data unavailable ({exc}); falling back.")

    if MERGED_REAL_DATA.exists():
        print(f"  Loading cached real panel: {MERGED_REAL_DATA}")
        df = pd.read_csv(MERGED_REAL_DATA)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    if RAW_DATA.exists():
        print(f"  Loading raw data: {RAW_DATA}")
        df = pd.read_csv(RAW_DATA)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    print("  No raw data found. Generating synthetic panel...")
    from data.generate_synthetic_data import generate_panel, add_derived_weather
    df = generate_panel()
    df = add_derived_weather(df)
    RAW_DATA.parent.mkdir(exist_ok=True)
    df.to_csv(RAW_DATA, index=False)
    print(f"  Saved synthetic data to {RAW_DATA}")
    return df


def step2_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Validate the dataset against quality rules."""
    print("\n" + "=" * 60)
    print("STEP 2: Validate")
    print("=" * 60)
    from src.features.validate_data import validate_data
    report = validate_data(df)
    return df


def step3_order_and_engineer(df: pd.DataFrame):
    """Steps 3–5: Chronological sort, feature engineering, and temporal split."""
    print("\n" + "=" * 60)
    print("STEP 3-5: Order, Engineer Features, Split")
    print("=" * 60)
    from src.features.build_features import build_feature_matrix
    X, y, city, splits, dates = build_feature_matrix(df)
    print(f"\n  Features built: {X.shape[1]}")
    print(f"  Feature list: {list(X.columns)}")
    return X, y, city, splits, dates


def step6_train(X, y, splits, dates=None):
    """Step 6: Tune (temporal forward-chaining CV) and train all models."""
    print("\n" + "=" * 60)
    print("STEP 6: Train (temporal forward-chaining CV on training data only)")
    print("=" * 60)
    from src.models.train import train_all_models
    results = train_all_models(X, y, splits, dates=dates)
    return results


def step7_evaluate(results, splits, X, y, city):
    """Step 7: Evaluate on held-out test set (bootstrap CIs)."""
    print("\n" + "=" * 60)
    print("STEP 7: Evaluate (test: 2020–2021, bootstrap CIs)")
    print("=" * 60)
    from src.evaluation.evaluate import evaluate_all_models
    report = evaluate_all_models(results, splits, X, y, city)
    return report


def step8_report(results, report, splits, X, y, city, dates):
    """Step 8: Generate plots and save final report."""
    print("\n" + "=" * 60)
    print("STEP 8: Report (plots + summary)")
    print("=" * 60)
    from src.evaluation.plot_results import plot_all
    plot_all(results, splits, X, y, city, report, dates)

    # Save summary table as CSV
    summary_rows = []
    for name in report["models"]:
        test = report["models"][name]["test"]["pooled"]
        summary_rows.append({
            "model": name,
            "test_MAE": round(test["MAE"], 2),
            "test_RMSE": round(test["RMSE"], 2),
            "test_R2": round(test["R2"], 3),
            "bootstrap_MAE_ci_low": round(report["models"][name]["test"]["bootstrap_ci"]["MAE_ci"][0], 2),
            "bootstrap_MAE_ci_high": round(report["models"][name]["test"]["bootstrap_ci"]["MAE_ci"][1], 2),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary table saved → {summary_path}")
    print("\n" + summary_df.to_string(index=False))

    # Save per-city breakdown for the test year
    per_city_rows = []
    for name in report["models"]:
        for city_name, m in report["models"][name]["test"]["per_city"].items():
            per_city_rows.append({
                "City": city_name,
                "Model": name,
                "MAE": round(m["MAE"], 2),
                "RMSE": round(m["RMSE"], 2),
                "R2": round(m["R2"], 3),
            })
    per_city_df = pd.DataFrame(per_city_rows)
    per_city_path = OUTPUT_DIR / "per_city_results.csv"
    per_city_df.to_csv(per_city_path, index=False)
    print(f"\n  Per-city table saved → {per_city_path}")

    # Save ablation summary
    ablation_path = OUTPUT_DIR / "ablation_summary.json"
    with open(ablation_path, "w") as f:
        json.dump(report["ablation"], f, indent=2)
    print(f"\n  Ablation summary saved → {ablation_path}")

    return summary_df


def main():
    print("============================================================")
    print("   DENGUE FORECASTING PIPELINE (Mindanao 3-city panel)      ")
    print("   Proposal: Bandiola & Balani — Weekly t+2 forecasting    ")
    print("   Train: 2016–2019 | Test: 2020–2021 | XGBoost + ablation  ")
    print("============================================================")
    print()

    # Steps 1–8
    df = step1_load()
    df = step2_validate(df)
    X, y, city, splits, dates = step3_order_and_engineer(df)
    results = step6_train(X, y, splits, dates=dates)
    report = step7_evaluate(results, splits, X, y, city)
    step8_report(results, report, splits, X, y, city, dates)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"  Results JSON: {OUTPUT_DIR / 'results.json'}")
    print(f"  Summary CSV:  {OUTPUT_DIR / 'summary_table.csv'}")
    print(f"  Plots:        {OUTPUT_DIR / 'figures/'}")
    print()

    # Key finding summary
    abl = report["ablation"]
    print(f"  * Ablation MAE improvement: {abl['MAE_improvement']:+.1f} "
          f"({abl['MAE_improvement_pct']:+.1f}%)")
    print(f"  * Full vs Case-only RMSE: {abl['RMSE_improvement']:+.1f} "
          f"({abl['RMSE_improvement_pct']:+.1f}%)")
    print(f"  * Best test MAE: "
          f"{min(report['models'][m]['test']['pooled']['MAE'] for m in report['models']):.1f}")
    print(f"  * Persistence MAE: {abl['persistence_MAE']:.1f}")


from config import OUTPUT_DIR

if __name__ == "__main__":
    main()
