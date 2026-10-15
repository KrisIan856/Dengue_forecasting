# Dengue Forecasting Pipeline

**Forecasting Weekly Dengue Cases in Mindanao Cities Using XGBoost and Lagged Meteorological Features**

Based on the research proposal by Bandiola & Balani (South Philippines Adventist College).

---

## Quick Start

```bash
# 1. Activate the conda environment
conda activate forecasting

# 2. Run the full pipeline
python pipeline.py

# 3. (Optional) Run the statistical benchmark notebook
jupyter lab benchmark_wilcoxon.ipynb
```

That's it — the pipeline will:
1. Load and merge the real per-city panels from `datasets/`, rebuilding `date` and cleaning documented defects
2. Validate data quality
3. Engineer 28 features with a leakage-free t+2 target
4. Split: train (2016–2019), embargo, test (2020–2021)
5. Train XGBoost + persistence with forward-chaining CV
6. Evaluate with bootstrap 95% CIs
7. Generate plots

> 💡 **No real data?** Set `USE_REAL_DATA = False` in `config.py` to fall back
> to the synthetic generator. See [Data Sources](#data-sources) for details.

---

## Project Structure

The project lives at the root of the `forecasting` conda environment
(`~/miniconda3/envs/forecasting/`). Only these entries are project files —
everything else in that folder (`Lib/`, `Scripts/`, `DLLs/`, `conda-meta/`,
`python.exe`, …) belongs to the conda environment itself.

```
forecasting/                          # env root + project files
├── README.md                         # This guide
├── config.py                         # Paths, features, params, splits, cleaning policy
├── pipeline.py                       # Main entry point (9-step pipeline)
├── benchmark_wilcoxon.py             # Statistical benchmark (script form)
├── benchmark_wilcoxon.ipynb          # Statistical benchmark (notebook form, pre-run)
├── datasets/                         # REAL per-city weekly panels (input data)
│   ├── dengue_davao_city_week_panel_2016_2021.csv
│   ├── dengue_cagayan_de_oro_city_week_panel_2016_2021.csv
│   └── dengue_zamboanga_city_week_panel_2016_2021.csv
├── notebooks/
│   └── dengue_forecasting.ipynb      # Exploratory notebook version of the pipeline
├── papers/                           # Proposal documents (.docx)
├── scripts/
│   └── build_minutes.py              # Generates the meeting-minutes .docx
├── data/
│   ├── generate_synthetic_data.py    # Synthetic fallback generator
│   └── merged_dengue_panel_real.csv  # Cached merge of datasets/ (auto-generated)
├── src/
│   ├── data_loader.py                # Merges datasets/, rebuilds date, cleans outliers
│   ├── features/
│   │   ├── validate_data.py          # Data quality checks (§5.4)
│   │   └── build_features.py         # 28-feature engineering + t+2 target (§5.6)
│   ├── models/
│   │   ├── models.py                 # Model registry + forward-chaining CV
│   │   └── train.py                  # Training with leakage-free CV (§5.8, §5.11)
│   └── evaluation/
│       ├── evaluate.py               # Metrics + block bootstrap CIs (§5.9, §5.10)
│       └── plot_results.py           # Visualization (§5.9: report)
├── outputs/                          # All generated results
│   ├── results.json                  # Full evaluation report
│   ├── summary_table.csv             # Model comparison table
│   ├── per_city_results.csv          # Per-city metrics
│   ├── wilcoxon_benchmark_summary.csv# Wilcoxon benchmark table
│   ├── per_city_wilcoxon.csv         # Per-city Wilcoxon tests
│   ├── benchmark_runs.csv            # Raw per-seed run log
│   ├── ablation_summary.json
│   ├── data_validation_report.json
│   └── figures/
└── _archive/                         # Superseded files (kept for reference)
```

### Data flow

`datasets/*.csv` → `src/data_loader.py` (merge, rebuild `date`, drop
duplicates/week-53/missing, winsorize outliers) → `data/merged_dengue_panel_real.csv`
→ `build_features.py` (28 features + t+2 target + temporal split) → training.

To inspect the raw panels without any cleaning, set `USE_REAL_DATA = False`
in `config.py`, or set `OUTLIER_MODE = "keep"` to disable outlier handling only.

---

## What the Project Does

### The Research Question
Can XGBoost, given lagged meteorological features, forecast weekly dengue cases **two weeks ahead** for Davao City, Cagayan de Oro City, and Zamboanga City more accurately than a model using case history alone — evaluated on a held-out test set (2020–2021)?

### Models Compared

|| Model | Features | Purpose |
||-------|----------|---------|
|| **Persistence** | cases_lag1 only | Minimum practical benchmark (not ML) |
|| **Case-only XGBoost** | 6 case/seasonicity features | Ablation baseline: how much can dengue history alone predict? |
|| **Full XGBoost** | All 28 features | Main model: does weather add value? |

### Temporal Validation Design (§5.8)

No random splits. The data is strictly chronological:

```
2016 ── 2017 ── 2018 ── 2019 ── [embargo 2wks] ── 2020 ── 2021 (test)
 [                       train + tuning         ]   [     out-of-sample         ]
```

- **Training/tuning**: Feb 2016 – Dec 2019 (~70% of panel), using forward-chaining cross-validation (4 folds)
- **Embargo**: 2 weeks skipped between train and test to prevent lag-window overlap
- **Test set**: 2020–2021 (one-time out-of-sample assessment, ~30%)

### 28 Engineered Features (§5.6)

| Group | Features | Count |
|-------|----------|-------|
| Weather lags | rainfall_lag1–4, temperature_lag1–4, humidity_lag1–4 | 12 |
| Rolling aggregates | rainfall_rollsum4, temperature_rollmean4, humidity_rollmean4 | 3 |
| Case momentum | cases_lag1, cases_lag2, cases_lag4, cases_rollmean4 | 4 |
| Bioclimatic | heat_index, tmin, tmax, diurnal_temp_range, spi3, pnp, ndvi | 7 |
| Seasonality | sin_week, cos_week | 2 |

All features use **only information available at the forecast origin** (no future leakage).

### Metrics (§5.9)

|- **MAE** — average error in case counts (primary)
|- **RMSE** — weights large misses during epidemic peaks
|- **R²** — share of variation explained
|- **Relative MAE** — ratio of model MAE to persistence MAE (error ratio against baseline)
|- **Bootstrap 95% CIs** — block bootstrap (4-week contiguous blocks within city) to preserve serial dependence

---

## Key Code Modules & How to Use

### 1. `config.py` — Configuration
All paths, feature lists, model hyperparameters, and temporal split parameters live here. Edit this file to tune models or change the temporal design.

```python
from config import ALL_FEATURES, XGB_PARAMS, TRAIN_END_YEAR, TEST_START_YEAR
```

### 3. `data/generate_synthetic_data.py` — Data
Since real DOH + CCHAIN data isn't bundled, this generates a realistic synthetic panel (786 city-weeks) with seasonal dengue dynamics, weather variables, and a COVID dip in 2020. **Real per-city panels are available** in `datasets/` — see [Data Sources](#data-sources).

```python
from data.generate_synthetic_data import generate_panel, add_derived_weather
df = generate_panel(start_year=2016, end_year=2021)
df = add_derived_weather(df)
```

### 3. `src/features/validate_data.py` — Validation
Checks schema completeness, missing weeks, plausibility bounds, and missing-value rates.

```python
from src.features.validate_data import validate_data
report = validate_data(df)
```

### 4. `src/features/build_features.py` — Feature Engineering
Builds all 28 features with strict lookback-only logic, creates the t+2 target, and applies the temporal split with embargo.

```python
from src.features.build_features import build_feature_matrix
X, y, city, splits = build_feature_matrix(df)
# splits contains boolean masks: 'train', 'embargo', 'test'
```

### 5. `src/models/models.py` — Models
Defines the three model approaches and the forward-chaining CV generator.

```python
from src.models.models import build_model, get_feature_set
model = build_model("xgb_full")  # returns XGBRegressor
features = get_feature_set("xgb_full")  # returns all 28 features
```

### 6. `src/models/train.py` — Training
Tunes XGBoost hyperparameters via forward-chaining CV on training data only, then trains all 3 models.

```python
from src.models.train import train_all_models
results = train_all_models(X, y, splits)
# Each result: {name, model, params, feature_set, train_metrics, test_metrics, predictions}
```

### 7. `src/evaluation/evaluate.py` — Evaluation
Computes MAE/RMSE/R² with block-bootstrap 95% CIs and runs the weather ablation comparison.

```python
from src.evaluation.evaluate import evaluate_all_models
report = evaluate_all_models(results, splits, X, y, city)
```

### 8. `src/evaluation/plot_results.py` — Visualization
Generates 3 plots: actual vs predicted, per-city MAE, ablation comparison.

```python
from src.evaluation.plot_results import plot_all
plot_all(results, splits, X, y, city, report)
```

### 9. `pipeline.py` — Full Pipeline
Runs all 9 steps end-to-end. This is your entry point.

```bash
python pipeline.py
```

---

## Data Sources

The pipeline runs on the real per-city weekly panels in `datasets/`:

```
datasets/dengue_davao_city_week_panel_2016_2021.csv
datasets/dengue_cagayan_de_oro_city_week_panel_2016_2021.csv
datasets/dengue_zamboanga_city_week_panel_2016_2021.csv
```

Each file has one row per city-week (262 rows × 36 columns) and carries the raw
weekly variables:

- `city`, `year`, `week_number` — identifiers (the loader rebuilds `date` from these)
- `weekly_dengue_cases` — target series
- `rainfall`, `temperature`, `humidity`, `tmin`, `tmax` — raw weekly climate
- `heat_index`, `diurnal_temp_range`, `ndvi`, `spi3`, `pnp` — bioclimatic indices
- `data_quality_flags` — upstream quality annotation (`ok`, `missing_target`,
  `insufficient_history`, `missing_week`)

### What the loader does (`src/data_loader.py`)

Merges the three files, then applies documented cleaning:

1. Drops ISO `week_number > 52` rows (spurious continuation records).
2. Collapses duplicate `(city, year, week_number)` rows by taking the median of
   the conflicting reported values.
3. Reconstructs the `date` column via `pd.Timestamp.fromisocalendar`.
4. Drops rows whose `weekly_dengue_cases` is missing.
5. Winsorises extreme weekly counts to the city-specific upper IQR fence
   (`OUTLIER_IQR_FACTOR = 8.0`), which caps clear keying errors such as
   Zamboanga 2019 week 28 (3,382 against an outbreak ramp of 250–380) while
   preserving genuine peaks such as the Davao 2016 peak.

Every step is logged at runtime. Controls live in `config.py`:

```python
USE_REAL_DATA = True           # False -> use the synthetic generator
OUTLIER_MODE = "winsorize"     # or "drop" / "keep"
OUTLIER_IQR_FACTOR = 8.0
```

### Using a different dataset

To point at your own data, drop a CSV matching the column list above into
`datasets/` and update `CITY_FILE_PATTERNS` in `config.py` with a glob matching
your filename. Feature engineering and training need no changes.

See `data/generate_synthetic_data.py` for the expected column structure and
data types.

---

## Reproducing the Paper's Results

The pipeline is designed to reproduce the experimental structure from the proposal:

|- **RQ1**: MAE/RMSE/R² on 2020–2021 test set → see `outputs/results.json → models.*.test`
|- **RQ2 (ablation)**: Case-only vs Full XGBoost → see `outputs/results.json → ablation`
|- **RQ3 (per-city consistency)**: → see `outputs/results.json → models.*.test.per_city`

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

Install:
```bash
pip install pandas scikit-learn xgboost matplotlib seaborn numpy
```

---

## Citation

If you use this code in your research, cite the underlying proposal:

> Bandiola, K. I., & Balani, A. (2025). *Forecasting Weekly Dengue Cases in Mindanao Cities Using XGBoost and Lagged Meteorological Features*. Bachelor's proposal, South Philippines Adventist College.
