"""
Configuration for the Dengue Forecasting Project.

Centralizes all paths, feature definitions, temporal splits, and model
hyperparameters so that every module imports from here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

RAW_DATA = DATA_DIR / "raw_dengue_panel_2016_2021.csv"

# ── Real-data source (per-city weekly panels) ───────────────────────────────
# The environment ships one tidy panel per city in <env>/datasets/. The
# loader in src/data_loader.py merges them, rebuilds the ISO `date` column,
# and applies the documented cleaning rules below.
REAL_DATA_DIR = BASE_DIR / "datasets"
MERGED_REAL_DATA = DATA_DIR / "merged_dengue_panel_real.csv"
USE_REAL_DATA = True  # set False to force the synthetic generator

CITY_FILE_PATTERNS = {
    "Davao City": "dengue_davao_city_week_panel_*.csv",
    "Cagayan de Oro City": "dengue_cagayan_de_oro_city_week_panel_*.csv",
    "Zamboanga City": "dengue_zamboanga_city_week_panel_*.csv",
}

# ── Outlier policy for the real panel ───────────────────────────────────────
# The source files contain a handful of implausible weekly counts (notably
# Zamboanga 2019 week 28 = 3382 against an outbreak ramp of 250-380).
#   "winsorize" caps them at the city-specific IQR upper fence (default)
#   "drop"      removes those rows
#   "keep"      disables any handling, for raw-data inspection
OUTLIER_MODE = "winsorize"
OUTLIER_IQR_FACTOR = 8.0

# ── Temporal design (weeks) ────────────────────────────────────────────────
# Updated split: train 2016–2019 (~70%), test 2020–2021 (~30%)
# 2-week embargo between train end (Dec 2019) and test start (Jan 2020)
# prevents lag-window overlap at the train/test boundary.
TRAIN_END_YEAR = 2019
TEST_START_YEAR = 2020
EMBARGO_WEEKS = 2  # two-week embargo between train and test
FORECAST_HORIZON = 2  # t+2 weeks

# ── Cities ─────────────────────────────────────────────────────────────────
CITIES = ["Davao City", "Cagayan de Oro City", "Zamboanga City"]
CITY_MAP = {
    "Davao City": "DAVAO",
    "Cagayan de Oro City": "CDO",
    "Zamboanga City": "ZAMBA",
}

# ── Feature groups (28 predictors) ─────────────────────────────────────────
# See proposal §5.6 for the full table.

WEATHER_LAGS = [
    f"rainfall_lag{i}" for i in range(1, 5)
] + [
    f"temperature_lag{i}" for i in range(1, 5)
] + [
    f"humidity_lag{i}" for i in range(1, 5)
]  # 12

ROLLING_AGGREGATES = [
    "rainfall_rollsum4",
    "temperature_rollmean4",
    "humidity_rollmean4",
]  # 3

CASE_MOMENTUM = [
    "cases_lag1", "cases_lag2", "cases_lag4", "cases_rollmean4",
]  # 4

BIOCLIMATIC = [
    "heat_index", "tmin", "tmax",
    "diurnal_temp_range", "spi3", "pnp", "ndvi",
]  # 7

SEASONALITY = ["sin_week", "cos_week"]  # 2

# Full feature list (order matters for ablation sets) — exactly 28 features
ALL_FEATURES = WEATHER_LAGS + ROLLING_AGGREGATES + CASE_MOMENTUM + BIOCLIMATIC + SEASONALITY

# Case-only feature subset for the ablation baseline — 6 features
CASE_ONLY_FEATURES = CASE_MOMENTUM + SEASONALITY

TARGET = "target_cases_t2"  # the t+2 shifted target column name

# ── Model configurations ───────────────────────────────────────────────────

# XGBoost hyperparameters (tuned via temporal forward-chaining CV in train.py)
# Uses robust L1 loss (reg:absoluteerror) to handle skewed dengue case counts,
# with regularization to prevent overfitting to historical outbreak spikes.
XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "n_estimators": 30,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_lambda": 5.0,
    "reg_alpha": 5.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

# Innovation shrinkage factor on persistence anchor (tuned via temporal CV)
SHRINKAGE = 0.3

# Forward-chaining CV: number of folds within the training period
N_CV_FOLDS = 3
CV_TEST_SIZE = 52  # ~1 year of weeks per fold

# Bootstrap
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK_SIZE = 4  # 4-week blocks for block bootstrap

# Data quality thresholds
MAX_MISSING_PCT = 0.05  # max 5% missing before dropping a variable
