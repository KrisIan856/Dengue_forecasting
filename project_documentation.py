# -*- coding: utf-8 -*-
"""
Dengue Forecasting Project - Complete Documentation
====================================================

Forecasting Weekly Dengue Cases in Mindanao Cities Using XGBoost 
and Lagged Meteorological Features

Based on the research proposal by Bandiola & Balani 
(South Philippines Adventist College)

Run this file to display the full project documentation:
    python project_documentation.py
"""

import sys
import io

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

def print_subsection(title):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

def main():
    print("=" * 80)
    print()
    print("           DENGUE FORECASTING PROJECT - COMPLETE DOCUMENTATION")
    print()
    print("   Forecasting Weekly Dengue Cases in Mindanao Cities Using XGBoost")
    print("   and Lagged Meteorological Features")
    print()
    print("   Based on: Bandiola & Balani (South Philippines Adventist College)")
    print()
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    print_section("1. PROJECT OVERVIEW")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
OBJECTIVE:
    Develop a machine learning system to forecast dengue cases 2 weeks ahead 
    (t+2 horizon) for three Mindanao cities:
        • Davao City
        • Cagayan de Oro City
        • Zamboanga City

APPROACH:
    • Models: XGBoost with 28 engineered features vs. persistence baseline
    • Validation: Temporal (chronological) split — no random splitting
    • Data Period: 2016–2021 (weekly resolution)
    • Forecast Target: Cases at week t+2 (2-week lead time for public health)
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("2. RESEARCH QUESTION")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
PRIMARY QUESTION:
    Can XGBoost, given lagged meteorological features, forecast weekly dengue 
    cases two weeks ahead for Davao City, Cagayan de Oro City, and Zamboanga 
    City more accurately than a model using case history alone?

MODELS COMPARED:
    
    Model                  Features              Purpose
    ─────────────────    ──────────────────      ───────────────────────────
    Persistence          cases_lag1 only         Minimum practical benchmark
    Case-only XGBoost    6 case/seasonality      Ablation: case history only
    Full XGBoost         All 28 features         Main model: weather + cases
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("3. DATA SOURCES & PROCESSING")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
RAW DATA:
    Three per-city weekly panels (2016–2021):
        • datasets/dengue_davao_city_week_panel_2016_2021.csv
        • datasets/dengue_cagayan_de_oro_city_week_panel_2016_2021.csv
        • datasets/dengue_zamboanga_city_week_panel_2016_2021.csv

VARIABLES PER CITY-WEEK:
    • weekly_dengue_cases — target series
    • rainfall, temperature, humidity — raw climate
    • tmin, tmax, heat_index, diurnal_temp_range — temperature indices
    • ndvi, spi3, pnp — bioclimatic indices

DATA CLEANING PIPELINE (src/data_loader.py):
    
    1. Drop spurious records: Remove ISO week 53 (invalid continuation weeks)
    2. Collapse duplicates: Median of conflicting (city, year, week) entries
    3. Reconstruct date: pd.Timestamp.fromisocalendar(year, week, 1)
    4. Drop missing targets: Remove rows with NaN in weekly_dengue_cases
    5. Winsorize outliers: Cap extreme counts at city-specific IQR fence
    
    Example: Zamboanga 2019 week 28 = 3,382 cases (surrounding: 250–380)
             → Capped to prevent data-entry error from dominating loss

FINAL DATASET:
    ~771 city-weeks after cleaning
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("4. FEATURE ENGINEERING (28 FEATURES)")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
All features use ONLY information available at the forecast origin 
(no future leakage).

FEATURE GROUPS:
    
    Group                  Features                              Count
    ─────────────────      ──────────────────────────────────    ─────
    Weather Lags           rainfall_lag1–4,                         12
                           temperature_lag1–4,
                           humidity_lag1–4
    
    Rolling Aggregates     rainfall_rollsum4,                        3
                           temperature_rollmean4,
                           humidity_rollmean4
    
    Case Momentum          cases_lag1, cases_lag2,                   4
                           cases_lag4, cases_rollmean4
    
    Bioclimatic            heat_index, tmin, tmax,                   7
                           diurnal_temp_range, spi3,
                           pnp, ndvi
    
    Seasonality            sin_week, cos_week                        2
    
    TOTAL                                                         ─────
                                                                     28

FEATURE CONSTRUCTION LOGIC (src/features/build_features.py):

1. LAG FEATURES (Past-Only):
   
   # Cases
   cases_lag1 = weekly_dengue_cases.shift(1)  # last week
   cases_lag2 = weekly_dengue_cases.shift(2)  # 2 weeks ago
   cases_lag4 = weekly_dengue_cases.shift(4)  # 4 weeks ago
   
   # Weather
   rainfall_lag1 = rainfall.shift(1)
   # ... same for temperature, humidity

2. ROLLING AGGREGATES (4-Week Window):
   
   rainfall_rollsum4 = rainfall.shift(1).rolling(window=4).sum()
   temperature_rollmean4 = temperature.shift(1).rolling(window=4).mean()

3. CYCLICAL SEASONALITY:
   
   angle = 2π × (week_number / 52)
   sin_week = sin(angle)
   cos_week = cos(angle)
   
   Encodes week-of-year as circular features (captures seasonal patterns).

4. TARGET VARIABLE:
   
   target_cases_t2 = weekly_dengue_cases.shift(-2)  # cases 2 weeks future
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("5. MODEL ARCHITECTURE")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
1. PERSISTENCE BASELINE:
   
   Simplest possible forecast: predict last week's cases.
   
   y_pred[t+2] = cases_lag1[t]
   
   Purpose: Minimum practical benchmark. If XGBoost can't beat this, 
            it's useless.

2. XGBOOST MODELS (Case-Only & Full):
   
   RESIDUAL FORMULATION:
   Instead of predicting cases directly, predict CHANGE from persistence:
   
   y_pred[t+2] = max(0, cases_lag1 + shrinkage × Δ̂[t+2])
   
   Where:
     • cases_lag1 = persistence anchor (last week's cases)
     • Δ̂[t+2] = XGBoost's predicted innovation (change)
     • shrinkage = 0.3 (prevents overfitting to outbreak spikes)
   
   WHY RESIDUAL?
     • Trees extrapolate poorly in regime shifts (e.g., outbreak onset)
     • Anchoring to persistence keeps predictions stable
     • Weather features guide *adjustments*, not absolute levels

   XGBOOST HYPERPARAMETERS (config.py):
   
   {
       "objective": "reg:absoluteerror",  # L1 loss (robust to skewed)
       "n_estimators": 30,                 # shallow trees
       "max_depth": 3,                     # prevent overfitting
       "learning_rate": 0.03,              # slow learning
       "subsample": 0.8,                   # row sampling
       "colsample_bytree": 0.8,            # feature sampling
       "min_child_weight": 3,              # regularization
       "reg_lambda": 5.0,                  # L2 regularization
       "reg_alpha": 5.0,                   # L1 regularization
   }
   
   Tuning: Forward-chaining cross-validation (3 folds, chronological)

3. MODEL CLASSES (src/models/models.py):
   
   class ResidualXGBRegressor:
       def __init__(self, feature_set, shrinkage=0.3, **xgb_params):
           self.feature_set = feature_set
           self.shrinkage = shrinkage
           self.model = XGBRegressor(**xgb_params)
       
       def fit(self, X, y):
           # Train on residuals: y - persistence
           persistence = X["cases_lag1"]
           residual = y - persistence
           self.model.fit(X[self.feature_set], residual)
           return self
       
       def predict(self, X):
           persistence = X["cases_lag1"]
           predicted_residual = self.model.predict(X[self.feature_set])
           return max(0, persistence + self.shrinkage * predicted_residual)
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("6. TRAINING PIPELINE")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
TEMPORAL SPLIT DESIGN:
    
    2016 ── 2017 ── 2018 ── 2019 ── [2wks] ── 2020 ── 2021
     [         Train + CV Tuning        ]  [    Test    ]
              (~606 rows)                    (~144 rows)
    
    • Train: Feb 2016 – Dec 2019 (70%)
    • Embargo: 2 weeks skipped (prevents lag-window overlap)
    • Test: 2020–2021 (30%, one-time out-of-sample)
    
    WHY NO RANDOM SPLIT?
    Dengue is time-series. Random splits leak future information into 
    training (e.g., model sees week 30 before predicting week 28).

FORWARD-CHAINING CROSS-VALIDATION:
    
    def make_temporal_forward_chaining_folds(dates, n_splits=3):
        for i in range(n_splits):
            train_dates = dates[:t_end]   # past
            val_dates = dates[t_end:v_end]  # future
            yield train_idx, val_idx
    
    Fold structure:
        • Fold 1: Train on 2016–2017, validate on 2018
        • Fold 2: Train on 2016–2018, validate on 2019
        • Fold 3: Train on 2016–2019, validate on early 2020
    
    Mimics real forecasting: always predict the future using the past.

TRAINING STEPS (src/models/train.py):
    
    1. Build feature matrix (28 features + target)
    2. Apply temporal split (train/embargo/test)
    3. Tune hyperparameters via forward-chaining CV on train set
    4. Train final models on full train set
    5. Predict on test set (2020–2021)
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("7. EVALUATION FRAMEWORK")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
METRICS:
    
    Metric       Formula                    Interpretation
    ─────────    ───────────────────────    ───────────────────────────
    MAE          mean(|y - ŷ|)              Average error in cases
    RMSE         sqrt(mean((y - ŷ)²))       Penalizes large errors
    R²           1 - (SS_res / SS_tot)      Variance explained (0–1)
    Rel. MAE     MAE_model / MAE_persist    Ratio vs. baseline (<1 = better)

BOOTSTRAP CONFIDENCE INTERVALS:
    
    Block bootstrap (4-week contiguous blocks within city):
        • Preserves serial dependence (autocorrelation in time series)
        • 2,000 resamples → 95% CI for each metric
    
    def block_bootstrap(y_true, y_pred, city, n_boot=2000, block_size=4):
        for _ in range(n_boot):
            # Sample 4-week blocks (with replacement) within each city
            # Compute metric on resampled data
        return np.percentile(metrics, [2.5, 97.5])

STATISTICAL TESTING:
    
    Wilcoxon signed-rank test (one-tailed, α = 0.05):
        • Tests if Full XGBoost is significantly better than baselines
        • 10 independent training runs (seeds 42–51)
        • Non-parametric (doesn't assume normal distribution of errors)
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("8. RESULTS & KEY FINDINGS")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
MAIN RESULTS (Test Set: 2020–2021):
    
    Model                  MAE      RMSE     R²      Relative MAE
    ─────────────────      ─────    ─────    ─────   ─────────────
    Persistence            6.0      9.5      0.19    1.00 (baseline)
    Case-only XGBoost      6.1      9.6      0.19    1.02
    Full XGBoost           5.9      9.4      0.27    0.98
    
    KEY FINDING: Full XGBoost marginally beats persistence (~1% MAE improvement).

PER-CITY BREAKDOWN:
    
    City                 Persistence MAE    Full XGBoost MAE    Improvement
    ─────────────────    ─────────────────    ────────────────    ───────────
    Davao City           7.2                  6.5                 -9.7% ✓
    Cagayan de Oro       5.1                  4.8                 -5.9% ✓
    Zamboanga City       4.3                  6.4                 +48.8% ✗
    
    INSIGHT: Weather features help Davao and CDO, but hurt Zamboanga 
             (different climate dynamics).

STATISTICAL SIGNIFICANCE:
    
    Wilcoxon test (10 seeds, one-tailed):
        • Full XGBoost vs. Persistence: p = 0.41 (not significant)
        • Full XGBoost vs. Case-only:   p = 0.38 (not significant)
    
    INTERPRETATION: Marginal gain is within noise range. The 1% improvement 
                    is not statistically significant at α = 0.05.

WHY LOW R²?
    
    All models have R² < 0.3. Reasons:
        1. COVID-19 disruption (2020): Training data (2016–2019) doesn't 
           contain pandemic-era dengue collapse
        2. Low signal-to-noise: Dengue dynamics driven by many unobserved 
           factors (mobility, immunity, interventions)
        3. Marginal feature contribution: Weather explains only a small 
           fraction of case variance
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("9. CODE STRUCTURE")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
forecasting/
├── config.py                    # All paths, features, hyperparameters
├── pipeline.py                  # Main entry point (9-step pipeline)
├── benchmark_wilcoxon.py        # Statistical benchmarking
├── datasets/                    # Raw per-city data
│   ├── dengue_davao_city_week_panel_2016_2021.csv
│   ├── dengue_cagayan_de_oro_city_week_panel_2016_2021.csv
│   └── dengue_zamboanga_city_week_panel_2016_2021.csv
├── src/
│   ├── data_loader.py           # Merge, clean, reconstruct dates
│   ├── features/
│   │   ├── validate_data.py     # Data quality checks
│   │   └── build_features.py    # 28-feature engineering + t+2 target
│   ├── models/
│   │   ├── models.py            # Model registry + forward-chaining CV
│   │   └── train.py             # Training with leakage-free CV
│   └── evaluation/
│       ├── evaluate.py          # Metrics + block bootstrap CIs
│       └── plot_results.py      # Visualization
├── notebooks/
│   └── dengue_forecasting.ipynb # Exploratory notebook
└── outputs/                     # Generated results
    ├── results.json
    ├── summary_table.csv
    ├── per_city_results.csv
    └── figures/

KEY MODULES:

1. config.py — Configuration Hub
   All paths, feature lists, model hyperparameters, temporal splits.
   
   from config import ALL_FEATURES, XGB_PARAMS, TRAIN_END_YEAR

2. src/data_loader.py — Data Ingestion
   Merges per-city CSVs, applies cleaning rules, reconstructs dates.
   
   from src.data_loader import load_real_panel
   df = load_real_panel()  # ~771 rows after cleaning

3. src/features/build_features.py — Feature Engineering
   Builds 28 features + t+2 target, applies temporal split.
   
   from src.features.build_features import build_feature_matrix
   X, y, city, splits, dates = build_feature_matrix(df)

4. src/models/models.py — Model Definitions
   ResidualXGBRegressor class + forward-chaining CV generator.
   
   from src.models.models import build_model
   model = build_model("xgb_full")

5. src/evaluation/evaluate.py — Metrics & CIs
   MAE/RMSE/R² with block bootstrap confidence intervals.
   
   from src.evaluation.evaluate import evaluate_all_models
   report = evaluate_all_models(results, splits, X, y, city)
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("10. HOW TO RUN")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
1. ACTIVATE ENVIRONMENT:
   conda activate forecasting

2. RUN FULL PIPELINE:
   python pipeline.py
   
   Executes 9 steps:
       1. Load data
       2. Validate quality
       3. Engineer features
       4. Split (train/embargo/test)
       5. Tune hyperparameters (forward-chaining CV)
       6. Train models
       7. Evaluate (metrics + bootstrap CIs)
       8. Generate plots
       9. Save results

3. RUN STATISTICAL BENCHMARK:
   python benchmark_wilcoxon.py
   
   Runs 10 seeds × 3 models, computes Wilcoxon tests, generates tables.

4. EXPLORE NOTEBOOK:
   jupyter lab notebooks/dengue_forecasting.ipynb
   
   Interactive walkthrough of the full pipeline.
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("11. KEY ALGORITHMS SUMMARY")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
1. XGBOOST (EXTREME GRADIENT BOOSTING):
   • Type: Ensemble of decision trees
   • Loss: L1 (absolute error) — robust to outliers
   • Regularization: L1 + L2 penalties on leaf weights
   • Strengths: Handles non-linear relationships, feature interactions
   • Weaknesses: Extrapolates poorly in regime shifts

2. RESIDUAL FORECASTING:
   • Idea: Predict CHANGE from baseline, not absolute level
   • Formula: ŷ = persistence + shrinkage × Δ̂
   • Benefit: Anchors predictions to recent observation, prevents wild 
              extrapolation

3. FORWARD-CHAINING CROSS-VALIDATION:
   • Idea: Always train on past, validate on future
   • Benefit: No temporal leakage, mimics real forecasting
   • Contrast: Random CV assumes i.i.d. data (invalid for time series)

4. BLOCK BOOTSTRAP:
   • Idea: Resample contiguous blocks (not individual points)
   • Benefit: Preserves autocorrelation structure
   • Use: Compute confidence intervals for metrics

5. WILCOXON SIGNED-RANK TEST:
   • Idea: Non-parametric test for paired differences
   • Use: Is model A significantly better than model B across 10 seeds?
   • Contrast: t-test assumes normality (often violated for error 
               distributions)
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("12. DEPENDENCIES")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
Required packages:
    
    pandas>=2.0
    scikit-learn>=1.3
    xgboost>=2.0
    matplotlib>=3.7
    numpy>=1.24
    seaborn>=0.13

Install:
    pip install pandas scikit-learn xgboost matplotlib seaborn numpy
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("13. CITATION")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
@misc{bandiola2025forecasting,
  title={Forecasting Weekly Dengue Cases in Mindanao Cities Using XGBoost 
         and Lagged Meteorological Features},
  author={Bandiola, K. I. and Balani, A.},
  year={2025},
  institution={South Philippines Adventist College}
}
""")

    # ─────────────────────────────────────────────────────────────────────────
    print_section("DOCUMENTATION COMPLETE")
    # ─────────────────────────────────────────────────────────────────────────
    
    print("""
For questions about the code, data, or methodology, refer to:
    
    • Proposal:     papers/ directory (.docx files)
    • Notebook:     notebooks/dengue_forecasting.ipynb (interactive)
    • Results:      outputs/results.json (full evaluation report)
    • Source code:  src/ directory (all Python modules)

Last Updated: October 14, 2026

""")


if __name__ == "__main__":
    main()
