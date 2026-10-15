# Dengue Forecasting Project — Complete Documentation

**Forecasting Weekly Dengue Cases in Mindanao Cities Using XGBoost and Lagged Meteorological Features**

*Based on the research proposal by Bandiola & Balani (South Philippines Adventist College)*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Question](#research-question)
3. [Data Sources & Processing](#data-sources--processing)
4. [Feature Engineering (28 Features)](#feature-engineering-28-features)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation Framework](#evaluation-framework)
8. [Results & Key Findings](#results--key-findings)
9. [Code Structure](#code-structure)
10. [How to Run](#how-to-run)

---

## Project Overview

### Objective
Develop a machine learning system to forecast dengue cases **2 weeks ahead** (t+2 horizon) for three Mindanao cities:
- **Davao City**
- **Cagayan de Oro City**
- **Zamboanga City**

### Approach
- **Models**: XGBoost with 28 engineered features vs. persistence baseline
- **Validation**: Temporal (chronological) split — no random splitting
- **Data Period**: 2016–2021 (weekly resolution)
- **Forecast Target**: Cases at week t+2 (2-week lead time for public health response)

---

## Research Question

> **Can XGBoost, given lagged meteorological features, forecast weekly dengue cases two weeks ahead for Davao City, Cagayan de Oro City, and Zamboanga City more accurately than a model using case history alone?**

### Models Compared

| Model | Features | Purpose |
|-------|----------|---------|
| **Persistence** | `cases_lag1` only | Minimum practical benchmark (predicts last week's cases) |
| **Case-only XGBoost** | 6 case/seasonality features | Ablation: how much can dengue history alone predict? |
| **Full XGBoost** | All 28 features | Main model: does weather add predictive value? |

---

## Data Sources & Processing

### Raw Data
Three per-city weekly panels (2016–2021):
```
datasets/dengue_davao_city_week_panel_2016_2021.csv
datasets/dengue_cagayan_de_oro_city_week_panel_2016_2021.csv
datasets/dengue_zamboanga_city_week_panel_2016_2021.csv
```

**Variables per city-week**:
- `weekly_dengue_cases` — target series
- `rainfall`, `temperature`, `humidity` — raw climate
- `tmin`, `tmax`, `heat_index`, `diurnal_temp_range` — temperature indices
- `ndvi`, `spi3`, `pnp` — bioclimatic indices

### Data Cleaning Pipeline (`src/data_loader.py`)

1. **Drop spurious records**: Remove ISO week 53 (invalid continuation weeks)
2. **Collapse duplicates**: Median of conflicting `(city, year, week)` entries
3. **Reconstruct date**: `pd.Timestamp.fromisocalendar(year, week, 1)` → Monday of each week
4. **Drop missing targets**: Remove rows with `NaN` in `weekly_dengue_cases`
5. **Winsorize outliers**: Cap extreme counts at city-specific IQR upper fence
   - Example: Zamboanga 2019 week 28 = 3,382 cases (surrounding weeks: 250–380) → capped
   - Preserves genuine outbreak peaks while preventing data-entry errors from dominating loss

**Final dataset**: ~771 city-weeks after cleaning

---

## Feature Engineering (28 Features)

All features use **only information available at the forecast origin** (no future leakage).

### Feature Groups

| Group | Features | Count |
|-------|----------|-------|
| **Weather Lags** | `rainfall_lag1–4`, `temperature_lag1–4`, `humidity_lag1–4` | 12 |
| **Rolling Aggregates** | `rainfall_rollsum4`, `temperature_rollmean4`, `humidity_rollmean4` | 3 |
| **Case Momentum** | `cases_lag1`, `cases_lag2`, `cases_lag4`, `cases_rollmean4` | 4 |
| **Bioclimatic** | `heat_index`, `tmin`, `tmax`, `diurnal_temp_range`, `spi3`, `pnp`, `ndvi` | 7 |
| **Seasonality** | `sin_week`, `cos_week` | 2 |
| **Total** | | **28** |

### Feature Construction Logic (`src/features/build_features.py`)

#### 1. Lag Features (Past-Only)
```python
# Cases
cases_lag1 = weekly_dengue_cases.shift(1)  # last week
cases_lag2 = weekly_dengue_cases.shift(2)  # 2 weeks ago
cases_lag4 = weekly_dengue_cases.shift(4)  # 4 weeks ago

# Weather
rainfall_lag1 = rainfall.shift(1)
# ... same for temperature, humidity
```

#### 2. Rolling Aggregates (4-Week Window)
```python
rainfall_rollsum4 = rainfall.shift(1).rolling(window=4).sum()
temperature_rollmean4 = temperature.shift(1).rolling(window=4).mean()
```

#### 3. Cyclical Seasonality
```python
angle = 2π × (week_number / 52)
sin_week = sin(angle)
cos_week = cos(angle)
```
Encodes week-of-year as circular features (captures seasonal patterns).

#### 4. Target Variable
```python
target_cases_t2 = weekly_dengue_cases.shift(-2)  # cases 2 weeks in future
```

---

## Model Architecture

### 1. Persistence Baseline
**Simplest possible forecast**: predict last week's cases.

```python
y_pred[t+2] = cases_lag1[t]
```

**Purpose**: Minimum practical benchmark. If XGBoost can't beat this, it's useless.

---

### 2. XGBoost Models (Case-Only & Full)

#### Residual Formulation
Instead of predicting cases directly, predict **change from persistence**:

```
y_pred[t+2] = max(0, cases_lag1 + shrinkage × Δ̂[t+2])
```

Where:
- `cases_lag1` = persistence anchor (last week's cases)
- `Δ̂[t+2]` = XGBoost's predicted innovation (change)
- `shrinkage` = 0.3 (prevents overfitting to outbreak spikes)

**Why residual?**
- Trees extrapolate poorly in regime shifts (e.g., outbreak onset)
- Anchoring to persistence keeps predictions stable
- Weather features guide *adjustments* to the baseline, not absolute levels

#### XGBoost Hyperparameters (`config.py`)

```python
{
    "objective": "reg:absoluteerror",  # L1 loss (robust to skewed counts)
    "n_estimators": 30,                 # shallow trees
    "max_depth": 3,                     # prevent overfitting
    "learning_rate": 0.03,              # slow learning
    "subsample": 0.8,                   # row sampling
    "colsample_bytree": 0.8,            # feature sampling
    "min_child_weight": 3,              # regularization
    "reg_lambda": 5.0,                  # L2 regularization
    "reg_alpha": 5.0,                   # L1 regularization
}
```

**Tuning**: Forward-chaining cross-validation (3 folds, chronological)

---

### 3. Model Classes (`src/models/models.py`)

```python
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
```

---

## Training Pipeline

### Temporal Split Design

```
2016 ── 2017 ── 2018 ── 2019 ── [2wks] ── 2020 ── 2021
 [         Train + CV Tuning        ]  [    Test    ]
          (~606 rows)                    (~144 rows)
```

- **Train**: Feb 2016 – Dec 2019 (70%)
- **Embargo**: 2 weeks skipped (prevents lag-window overlap)
- **Test**: 2020–2021 (30%, one-time out-of-sample)

**Why no random split?**
Dengue is time-series. Random splits leak future information into training (e.g., model sees week 30 before predicting week 28).

### Forward-Chaining Cross-Validation

```python
def make_temporal_forward_chaining_folds(dates, n_splits=3):
    for i in range(n_splits):
        train_dates = dates[:t_end]   # past
        val_dates = dates[t_end:v_end]  # future
        yield train_idx, val_idx
```

**Fold structure**:
- Fold 1: Train on 2016–2017, validate on 2018
- Fold 2: Train on 2016–2018, validate on 2019
- Fold 3: Train on 2016–2019, validate on early 2020

Mimics real forecasting: always predict the future using the past.

### Training Steps (`src/models/train.py`)

1. **Build feature matrix** (28 features + target)
2. **Apply temporal split** (train/embargo/test)
3. **Tune hyperparameters** via forward-chaining CV on train set
4. **Train final models** on full train set
5. **Predict** on test set (2020–2021)

---

## Evaluation Framework

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(│y - ŷ│)` | Average error in cases (primary) |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penalizes large errors (epidemic peaks) |
| **R²** | `1 - (SS_res / SS_tot)` | Variance explained (0–1, higher = better) |
| **Relative MAE** | `MAE_model / MAE_persistence` | Ratio vs. baseline (<1 = improvement) |

### Bootstrap Confidence Intervals

**Block bootstrap** (4-week contiguous blocks within city):
- Preserves serial dependence (autocorrelation in time series)
- 2,000 resamples → 95% CI for each metric

```python
def block_bootstrap(y_true, y_pred, city, n_boot=2000, block_size=4):
    for _ in range(n_boot):
        # Sample 4-week blocks (with replacement) within each city
        # Compute metric on resampled data
    return np.percentile(metrics, [2.5, 97.5])
```

### Statistical Testing

**Wilcoxon signed-rank test** (one-tailed, α = 0.05):
- Tests if Full XGBoost is significantly better than baselines
- 10 independent training runs (seeds 42–51)
- Non-parametric (doesn't assume normal distribution of errors)

---

## Results & Key Findings

### Main Results (Test Set: 2020–2021)

| Model | MAE | RMSE | R² | Relative MAE |
|-------|-----|------|----|--------------|
| **Persistence** | 6.0 | 9.5 | 0.19 | 1.00 (baseline) |
| **Case-only XGBoost** | 6.1 | 9.6 | 0.19 | 1.02 |
| **Full XGBoost** | **5.9** | **9.4** | **0.27** | **0.98** |

**Key finding**: Full XGBoost marginally beats persistence (~1% MAE improvement).

### Per-City Breakdown

| City | Persistence MAE | Full XGBoost MAE | Improvement |
|------|-----------------|------------------|-------------|
| Davao City | 7.2 | 6.5 | -9.7% ✓ |
| Cagayan de Oro | 5.1 | 4.8 | -5.9% ✓ |
| Zamboanga City | 4.3 | 6.4 | +48.8% ✗ |

**Insight**: Weather features help Davao and CDO, but hurt Zamboanga (different climate dynamics).

### Statistical Significance

Wilcoxon test (10 seeds, one-tailed):
- Full XGBoost vs. Persistence: **p = 0.41** (not significant)
- Full XGBoost vs. Case-only: **p = 0.38** (not significant)

**Interpretation**: Marginal gain is within noise range. The 1% improvement is not statistically significant at α = 0.05.

### Why Low R²?

All models have R² < 0.3. Reasons:
1. **COVID-19 disruption** (2020): Training data (2016–2019) doesn't contain pandemic-era dengue collapse
2. **Low signal-to-noise**: Dengue dynamics driven by many unobserved factors (mobility, immunity, interventions)
3. **Marginal feature contribution**: Weather explains only a small fraction of case variance

---

## Code Structure

```
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
```

### Key Modules

#### 1. `config.py` — Configuration Hub
All paths, feature lists, model hyperparameters, temporal splits.

```python
from config import ALL_FEATURES, XGB_PARAMS, TRAIN_END_YEAR
```

#### 2. `src/data_loader.py` — Data Ingestion
Merges per-city CSVs, applies cleaning rules, reconstructs dates.

```python
from src.data_loader import load_real_panel
df = load_real_panel()  # ~771 rows after cleaning
```

#### 3. `src/features/build_features.py` — Feature Engineering
Builds 28 features + t+2 target, applies temporal split.

```python
from src.features.build_features import build_feature_matrix
X, y, city, splits, dates = build_feature_matrix(df)
```

#### 4. `src/models/models.py` — Model Definitions
`ResidualXGBRegressor` class + forward-chaining CV generator.

```python
from src.models.models import build_model, make_temporal_forward_chaining_folds
model = build_model("xgb_full")
```

#### 5. `src/evaluation/evaluate.py` — Metrics & CIs
MAE/RMSE/R² with block bootstrap confidence intervals.

```python
from src.evaluation.evaluate import evaluate_all_models
report = evaluate_all_models(results, splits, X, y, city)
```

---

## How to Run

### 1. Activate Environment
```bash
conda activate forecasting
```

### 2. Run Full Pipeline
```bash
python pipeline.py
```
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

### 3. Run Statistical Benchmark
```bash
python benchmark_wilcoxon.py
```
Runs 10 seeds × 3 models, computes Wilcoxon tests, generates comparison tables.

### 4. Explore Notebook
```bash
jupyter lab notebooks/dengue_forecasting.ipynb
```
Interactive walkthrough of the full pipeline.

---

## Key Algorithms Summary

### 1. XGBoost (Extreme Gradient Boosting)
- **Type**: Ensemble of decision trees
- **Loss**: L1 (absolute error) — robust to outliers
- **Regularization**: L1 + L2 penalties on leaf weights
- **Strengths**: Handles non-linear relationships, feature interactions
- **Weaknesses**: Extrapolates poorly in regime shifts

### 2. Residual Forecasting
- **Idea**: Predict *change* from baseline, not absolute level
- **Formula**: `ŷ = persistence + shrinkage × Δ̂`
- **Benefit**: Anchors predictions to recent observation, prevents wild extrapolation

### 3. Forward-Chaining Cross-Validation
- **Idea**: Always train on past, validate on future
- **Benefit**: No temporal leakage, mimics real forecasting
- **Contrast**: Random CV assumes i.i.d. data (invalid for time series)

### 4. Block Bootstrap
- **Idea**: Resample contiguous blocks (not individual points)
- **Benefit**: Preserves autocorrelation structure
- **Use**: Compute confidence intervals for metrics

### 5. Wilcoxon Signed-Rank Test
- **Idea**: Non-parametric test for paired differences
- **Use**: Is model A significantly better than model B across 10 seeds?
- **Contrast**: t-test assumes normality (often violated for error distributions)

---

## Dependencies

```
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
numpy>=1.24
seaborn>=0.13
```

---

## Citation

```bibtex
@misc{bandiola2025forecasting,
  title={Forecasting Weekly Dengue Cases in Mindanao Cities Using XGBoost and Lagged Meteorological Features},
  author={Bandiola, K. I. and Balani, A.},
  year={2025},
  institution={South Philippines Adventist College}
}
```

---

## Contact & Questions

For questions about the code, data, or methodology, refer to:
- **Proposal**: `papers/` directory (.docx files)
- **Notebook**: `notebooks/dengue_forecasting.ipynb` (interactive walkthrough)
- **Results**: `outputs/results.json` (full evaluation report)

---

**Last Updated**: October 14, 2026
