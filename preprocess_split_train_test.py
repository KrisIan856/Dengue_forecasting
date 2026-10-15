"""
Pre-process + Splitting + Training + Testing 

This file is a verbatim copy (no logic changes) assembled from:
  - config.py                                  -> paths, temporal design, features, hyperparams
  - src/data_loader.py                         -> _build_date, _collapse_duplicates,
                                                  _apply_outlier_policy, load_real_panel (PRE-PROCESS / Load+Clean)
  - src/features/validate_data.py              -> _check_schema, _check_missing_weeks,
                                                  _check_plausibility, validate_data (PRE-PROCESS / Validate)
  - src/features/build_features.py             -> _add_lag_and_rolling, _add_seasonality,
                                                  _create_target, build_feature_matrix (PRE-PROCESS / Engineer + SPLITTING)
  - src/models/models.py                       -> ResidualXGBRegressor, persistence_predict,
                                                  make_temporal_forward_chaining_folds (TRAINING / model defs)
  - src/models/train.py                        -> _evaluate_y, _forward_chain_cv, train_model,
                                                  train_all_models (TRAINING)
  - src/evaluation/evaluate.py                 -> _metrics, _block_bootstrap_ci,
                                                  _evaluate_model_on_split, evaluate_all_models (TESTING / Evaluate)

                                                  
NOTE — What each stage shows:
  [1] PRE-PROCESS : load + clean (data_loader.py) -> validate (validate_data.py)
                    -> engineer lags/rolling/seasonality + t+2 target (build_features.py).
                    Input: per-city CSVs in datasets/. Output: cleaned panel + 28-feature matrix.
  [2] SPLITTING   : inside build_feature_matrix — temporal split with 2-week embargo.
                    Train = 2016-2019, Embargo = first 2 weeks of 2020 (skipped),
                    Test = rest of 2020-2021. No shuffling, no leakage.
  [3] TRAINING    : persistence baseline + case-only XGBoost + full XGBoost (models.py),
                    tuned by forward-chaining CV on train only, fit on train (train.py).
                    Test set never seen during tuning/fitting.
  [4] TESTING     : MAE/RMSE/R2 pooled + per-city on held-out test, block-bootstrap
                    95% CIs, weather ablation case-only vs full (evaluate.py).
"""

# NOTE: Stage [0] below is shared config (paths, split years, features, hyperparams).

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =====================================================================
# Copied from: config.py
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

RAW_DATA = DATA_DIR / "raw_dengue_panel_2016_2021.csv"

REAL_DATA_DIR = BASE_DIR / "datasets"
MERGED_REAL_DATA = DATA_DIR / "merged_dengue_panel_real.csv"
USE_REAL_DATA = True  # set False to force the synthetic generator

CITY_FILE_PATTERNS = {
    "Davao City": "dengue_davao_city_week_panel_*.csv",
    "Cagayan de Oro City": "dengue_cagayan_de_oro_city_week_panel_*.csv",
    "Zamboanga City": "dengue_zamboanga_city_week_panel_*.csv",
}

OUTLIER_MODE = "winsorize"
OUTLIER_IQR_FACTOR = 8.0

TRAIN_END_YEAR = 2019
TEST_START_YEAR = 2020
EMBARGO_WEEKS = 2  # two-week embargo between train and test
FORECAST_HORIZON = 2  # t+2 weeks

CITIES = ["Davao City", "Cagayan de Oro City", "Zamboanga City"]
CITY_MAP = {
    "Davao City": "DAVAO",
    "Cagayan de Oro City": "CDO",
    "Zamboanga City": "ZAMBA",
}

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

ALL_FEATURES = WEATHER_LAGS + ROLLING_AGGREGATES + CASE_MOMENTUM + BIOCLIMATIC + SEASONALITY

CASE_ONLY_FEATURES = CASE_MOMENTUM + SEASONALITY

TARGET = "target_cases_t2"  # the t+2 shifted target column name

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

SHRINKAGE = 0.3

N_CV_FOLDS = 3
CV_TEST_SIZE = 52  # ~1 year of weeks per fold

BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK_SIZE = 4  # 4-week blocks for block bootstrap

MAX_MISSING_PCT = 0.05  # max 5% missing before dropping a variable


# =====================================================================
# NOTE [1a] PRE-PROCESS — Load + Clean.
#   Shows: merge per-city panels, drop week-53 rows, collapse duplicates,
#   rebuild ISO date, drop missing targets, winsorize outliers, sort by city/date.
# Copied from: src/data_loader.py
# =====================================================================

REQUIRED_RAW_COLUMNS = [
    "city", "year", "week_number", "date", "weekly_dengue_cases",
    "rainfall", "temperature", "tmin", "tmax", "humidity",
    "heat_index", "diurnal_temp_range", "ndvi", "spi3", "pnp",
]


def _find_city_files() -> list[Path]:
    """Locate the per-city weekly panels inside the datasets directory."""
    files: list[Path] = []
    for city_name, pattern in CITY_FILE_PATTERNS.items():
        matches = sorted(REAL_DATA_DIR.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No file matching {pattern!r} for {city_name!r} in {REAL_DATA_DIR}"
            )
        files.append(matches[0])
    return files


def _build_date(year: pd.Series, week: pd.Series) -> pd.Series:
    """Reconstruct the Monday of each ISO epidemiological week.

    The per-city panels ship ``year`` and ``week_number`` but no ``date``;
    feature engineering sorts on ``date`` and derives the ISO week from it,
    so we rebuild it here with ``Timestamp.fromisocalendar``.
    """
    out = []
    for y, w in zip(year.to_numpy(), week.to_numpy()):
        y = int(y)
        w = int(w)
        # ISO year may legitimately be 52 or 53 only; clamp defensively.
        w = max(1, min(52, w))
        out.append(pd.Timestamp.fromisocalendar(y, w, 1))
    return pd.Series(out, index=year.index, dtype="datetime64[ns]")


def _collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (city, year, week_number) rows by median case count.

    Only the numeric columns are averaged; identifier and flag columns keep
    their first value. The dropped rows are reported so the caller can log
    exactly how many records were merged.
    """
    dup_mask = df.duplicated(subset=["city", "year", "week_number"], keep=False)
    n_dupes = int(dup_mask.sum())
    if n_dupes == 0:
        return df

    print(f"  Collapsing {n_dupes} duplicate city-year-week rows "
          f"(median of reported values)")

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    other = [c for c in df.columns if c not in numeric]
    agg = {c: "median" for c in numeric}
    agg.update({c: "first" for c in other})

    return (
        df.groupby(["city", "year", "week_number"], as_index=False)
          .agg(agg)
          .sort_values(["city", "year", "week_number"])
          .reset_index(drop=True)
    )


def _apply_outlier_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise (default) or drop extreme weekly case counts, per city.

    A single mis-keyed record (for example Zamboanga 2019 week 28 = 3382,
    where the surrounding outbreak weeks sit near 270-380) otherwise
    dominates squared-error metrics. Winsorising at
    ``median + OUTLIER_IQR_FACTOR * IQR`` keeps the row and its weather
    covariates while bounding its influence.
    """
    if OUTLIER_MODE == "keep":
        return df

    col = "weekly_dengue_cases"
    out = []
    for city, grp in df.groupby("city", sort=False):
        vals = grp[col]
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        cap = q3 + OUTLIER_IQR_FACTOR * iqr

        if OUTLIER_MODE == "winsorize":
            over = grp[vals > cap]
            if len(over):
                print(f"  {city}: winsorising {len(over)} week(s) above "
                      f"cap {cap:.0f} (max {vals.max():.0f})")
            grp = grp.copy()
            grp.loc[grp[col] > cap, col] = cap
        elif OUTLIER_MODE == "drop":
            grp = grp[grp[col] <= cap]
        out.append(grp)

    return pd.concat(out, ignore_index=True).sort_values(
        ["city", "year", "week_number"]
    ).reset_index(drop=True)


def load_real_panel(verbose: bool = True) -> pd.DataFrame:
    """Load, merge, and clean the real per-city dengue panels.

    Returns a tidy DataFrame with one row per city-week, including the
    reconstructed ``date`` column and all raw predictor columns required by
    the feature-engineering stage.
    """
    if verbose:
        print(f"Loading real panels from: {REAL_DATA_DIR}")

    frames = []
    for path in _find_city_files():
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
        if verbose:
            print(f"  read {path.name}: {frame.shape[0]} rows")

    df = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  merged: {len(df)} rows, {df.shape[1]} columns")

    # 1. Drop spurious ISO week 53 continuation rows.
    n_w53 = int((df["week_number"] > 52).sum())
    if n_w53:
        if verbose:
            print(f"  Dropping {n_w53} row(s) with week_number > 52")
        df = df[df["week_number"] <= 52]

    # 2. Collapse duplicate city-year-week rows.
    df = _collapse_duplicates(df)

    # 3. Reconstruct the ISO date column.
    df["date"] = _build_date(df["year"], df["week_number"])

    # 4. Drop rows with a missing target (flagged missing_target upstream).
    n_nan = int(df["weekly_dengue_cases"].isna().sum())
    if n_nan:
        if verbose:
            print(f"  Dropping {n_nan} row(s) with missing weekly_dengue_cases")
        df = df.dropna(subset=["weekly_dengue_cases"])

    # 5. Apply the configured outlier policy.
    df = _apply_outlier_policy(df)

    # 6. Sort chronologically and reset the index.
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Verify the schema the downstream stages require.
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Loaded panel is missing required columns: {missing}")

    if verbose:
        print(f"  final: {len(df)} rows, {df['city'].nunique()} cities, "
              f"years {df['year'].min()}-{df['year'].max()}")
    return df


# =====================================================================
# NOTE [1b] PRE-PROCESS (validation) — Data-quality checks.
#   Shows: schema check, missing-week check (52 weeks per city-year),
#   plausibility check (cases/rain/temp/humidity + missing% thresholds).
# Copied from: src/features/validate_data.py
# =====================================================================

def _check_schema(df: pd.DataFrame) -> list[str]:
    """Verify all columns required for feature engineering exist."""
    required = {
        "city", "year", "week_number", "weekly_dengue_cases",
        "rainfall", "temperature", "tmin", "tmax", "humidity",
        "heat_index", "diurnal_temp_range", "ndvi", "spi3", "pnp",
    }
    missing = required - set(df.columns)
    if missing:
        return [f"Schema error: missing columns — {sorted(missing)}"]
    return []


def _check_missing_weeks(df: pd.DataFrame) -> list[str]:
    """Each city-year should have 52 weeks. Flag gaps."""
    issues = []
    pivot = df.groupby(["city", "year"])["week_number"].nunique().reset_index()
    incomplete = pivot[pivot["week_number"] < 52]
    for _, row in incomplete.iterrows():
        issues.append(
            f"Missing weeks: {row['city']} {row['year']} has "
            f"{row['week_number']} weeks (expected 52)"
        )
    # Check for duplicate (city, year, week) combos
    dup = df.groupby(["city", "year", "week_number"]).size()
    dupes = dup[dup > 1]
    if len(dupes) > 0:
        issues.append(f"Duplicate city-year-week rows: {len(dupes)}")
    return issues


def _check_plausibility(df: pd.DataFrame) -> pd.DataFrame:
    """Flag implausible values; return a flagged copy (not modified)."""
    issues = []
    df_flagged = df.copy()

    # Dengue cases: should be non-negative integers
    neg_cases = df_flagged[df_flagged["weekly_dengue_cases"] < 0]
    if len(neg_cases) > 0:
        issues.append(f"Negative case counts: {len(neg_cases)} rows")

    # Rainfall: non-negative
    neg_rain = df_flagged[df_flagged["rainfall"] < 0]
    if len(neg_rain) > 0:
        issues.append(f"Negative rainfall: {len(neg_rain)} rows")

    # Temperature: 15-45 C is sane for the Philippines
    temp_oob = df_flagged[(df_flagged["temperature"] < 15) | (df_flagged["temperature"] > 45)]
    if len(temp_oob) > 0:
        issues.append(f"Temperature out of [15,45]C: {len(temp_oob)} rows")

    # Humidity: 20-100%
    hum_oob = df_flagged[(df_flagged["humidity"] < 20) | (df_flagged["humidity"] > 100)]
    if len(hum_oob) > 0:
        issues.append(f"Humidity out of [20,100]%: {len(hum_oob)} rows")

    # Missing-value percentages
    for col in df_flagged.columns:
        pct = df_flagged[col].isna().mean()
        if pct > MAX_MISSING_PCT:
            issues.append(
                f"Column '{col}' has {pct:.1%} missing "
                f"(threshold {MAX_MISSING_PCT:.0%})"
            )

    if issues:
        print("Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[OK] All data quality checks passed.")

    return df_flagged


def validate_data(df: pd.DataFrame) -> dict:
    """Run all validation checks and return a report dictionary."""
    errors = _check_schema(df)
    if errors:
        # If schema is broken, we can't proceed with other checks
        report = {"status": "FAILED", "errors": errors}
    else:
        issues = _check_missing_weeks(df)
        df = _check_plausibility(df)
        report = {
            "status": "PASSED" if not issues else "PASSED_WITH_WARNINGS",
            "n_rows": len(df),
            "n_cities": df["city"].nunique(),
            "years": sorted(df["year"].unique().tolist()),
            "total_missing_cells": int(df.isna().sum().sum()),
            "missing_pct": round(df.isna().mean().mean() * 100, 2),
            "city_breakdown": {
                c: len(df[df["city"] == c]) for c in df["city"].unique()
            },
            "plausibility_warnings": issues,
        }

    # Save report
    report_path = OUTPUT_DIR / "data_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Validation report saved — {report_path}")
    return report


# =====================================================================
# NOTE [1c + 2] PRE-PROCESS (feature engineering) + SPLITTING.
#   Shows pre-process: past-only lags/rolling (cases + weather), sin/cos seasonality,
#   t+2 target shift, drop first-4-weeks + NaNs -> X (28 cols), y, city, dates.
#   Shows splitting: year<=2019=train, first 2 weeks 2020=embargo (skipped),
#   rest 2020-2021=test. See `splits` dict in build_feature_matrix.
# Copied from: src/features/build_features.py
# =====================================================================

def _add_lag_and_rolling(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Add lag and rolling features per city, all using past information only.

    The forecast origin for row at position i is week i itself.
    - lags are relative to the current week (shift within city group)
    - rolling windows use min_periods to avoid lookahead.
    """
    # ── Case momentum features ────────────────────────────────────────
    df["cases_lag1"] = df.groupby(group_col)["weekly_dengue_cases"].shift(1)
    df["cases_lag2"] = df.groupby(group_col)["weekly_dengue_cases"].shift(2)
    df["cases_lag4"] = df.groupby(group_col)["weekly_dengue_cases"].shift(4)
    df["cases_rollmean4"] = (
        df.groupby(group_col)["weekly_dengue_cases"]
        .shift(1)
        .rolling(window=4, min_periods=4)
        .mean()
    )

    # ── Weather lags ──────────────────────────────────────────────────
    weather_vars = ["rainfall", "temperature", "humidity"]
    for var in weather_vars:
        for lag in range(1, 5):
            df[f"{var}_lag{lag}"] = df.groupby(group_col)[var].shift(lag)

    # ── Rolling aggregates (past-only) ────────────────────────────────
    df["rainfall_rollsum4"] = (
        df.groupby(group_col)["rainfall"]
        .shift(1)
        .rolling(window=4, min_periods=4)
        .sum()
    )
    df["temperature_rollmean4"] = (
        df.groupby(group_col)["temperature"]
        .shift(1)
        .rolling(window=4, min_periods=4)
        .mean()
    )
    df["humidity_rollmean4"] = (
        df.groupby(group_col)["humidity"]
        .shift(1)
        .rolling(window=4, min_periods=4)
        .mean()
    )

    return df


def _add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical sin/cos encoding of week number (proposal S5.1, S5.6)."""
    weeks_per_year = 52.0
    angle = 2 * np.pi * df["week_number"] / weeks_per_year
    df["sin_week"] = np.sin(angle).round(4)
    df["cos_week"] = np.cos(angle).round(4)
    return df


def _create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the t+2 target: dengue cases 2 weeks in the future.

    target_cases_t2 = weekly_dengue_cases at forecast_origin + FORECAST_HORIZON
    """
    df = df.sort_values(["city", "year", "week_number"]).reset_index(drop=True)
    df["target_cases_t2"] = df.groupby("city")["weekly_dengue_cases"].shift(-FORECAST_HORIZON)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict, pd.Series]:
    """Build the 28-feature matrix and t+2 target, apply temporal split.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (28 columns), sorted by city then date
    y : pd.Series     — target (weekly_dengue_cases at t+2)
    city: pd.Series   — city label per row (for per-city evaluation)
    splits : dict     — boolean masks for 'train', 'embargo', 'test'
    dates : pd.Series — forecast-origin date per row (for time-axis plots)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # ── 1. Create lag/rolling features (all lookback only) ─────────────
    df = _add_lag_and_rolling(df, "city")

    # ── 2. Cyclical seasonality ────────────────────────────────────────
    df = _add_seasonality(df)

    # ── 3. Create target (t+2 → shift by −2 within city) ──────────────
    df = _create_target(df)

    # ── 4. Drop rows that lack required lag history (first 4 weeks/city) ─
    idx_to_drop = df.groupby("city").head(4).index
    df_clean = df.drop(idx_to_drop).reset_index(drop=True)

    # ── 5. Drop rows with NaN in features or target ─────────────────────
    available_features = [f for f in ALL_FEATURES if f in df_clean.columns]
    missing_features = set(ALL_FEATURES) - set(available_features)
    if missing_features:
        raise ValueError(f"Not all 28 features could be built. Missing: {missing_features}")
    df_clean = df_clean.dropna(subset=available_features + ["target_cases_t2"])

    # ── 6. Assemble X, y ────────────────────────────────────────────────
    X = df_clean[available_features].copy()
    y = df_clean["target_cases_t2"].copy()
    city = df_clean["city"].copy()
    dates = df_clean["date"].copy()

    # ── 7. Temporal split with embargo ─────────────────────────────────
    year = dates.dt.year
    week = dates.dt.isocalendar().week

    train_mask = year <= TRAIN_END_YEAR
    embargo_weeks = (year == TEST_START_YEAR) & (week <= EMBARGO_WEEKS)
    test_mask = (year >= TEST_START_YEAR) & ~embargo_weeks

    splits = {
        "train": train_mask.values,
        "embargo": embargo_weeks.values,
        "test": test_mask.values,
    }

    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  Features: {len(available_features)} (expected {len(ALL_FEATURES)})")
    print(f"  Train:      {train_mask.sum()} rows  (2016–{TRAIN_END_YEAR})")
    print(f"  Embargo:    {embargo_weeks.sum()} rows  (skipped)")
    print(f"  Test ({TEST_START_YEAR}–2021): {test_mask.sum()} rows")

    return X, y, city, splits, dates


def get_case_only_features() -> list[str]:
    """Return the case-history-only feature list for the ablation baseline."""
    return CASE_ONLY_FEATURES.copy()


def get_full_features() -> list[str]:
    """Return the full 28-feature list."""
    return ALL_FEATURES.copy()


# =====================================================================
# NOTE [3a] TRAINING (model definitions).
#   Shows: persistence (predict=cases_lag1), ResidualXGBRegressor
#   (y_hat = max(0, persistence + shrinkage*XGB_residual)), case-only vs full sets,
#   temporal forward-chaining folds (train dates strictly before val dates).
# Copied from: src/models/models.py
# =====================================================================

class ResidualXGBRegressor:
    """XGBoost regressor forecasting trajectory innovations on top of persistence.

    Forecast formulation:
        y_hat_{t+2} = max(0, y_persistence + shrinkage * Delta_hat_{t+2})
    where:
        - y_persistence is cases_lag1 (the most recent confirmed observation)
        - Delta_hat_{t+2} is the forecasted change estimated by XGBoost
        - shrinkage scales the innovation to avoid overfitting outbreak spikes.

    This solves the regime-shift extrapolation boundary of tree-based models
    while enabling meteorological features to accurately guide trajectory adjustments.
    """

    def __init__(
        self,
        feature_set: list[str],
        shrinkage: float = SHRINKAGE,
        **xgb_params,
    ):
        self.feature_set = feature_set
        self.shrinkage = shrinkage
        self.xgb_params = xgb_params
        self.model = XGBRegressor(**xgb_params)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        p = X["cases_lag1"].values.astype(float)
        y_vals = y.values.astype(float) if hasattr(y, "values") else np.asarray(y, dtype=float)
        residual = y_vals - p
        self.model.fit(X[self.feature_set].values, residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = X["cases_lag1"].values.astype(float)
        pred_res = self.model.predict(X[self.feature_set].values)
        return np.maximum(0.0, p + self.shrinkage * pred_res)


def _xgb_full():
    return ResidualXGBRegressor(feature_set=ALL_FEATURES, shrinkage=SHRINKAGE, **XGB_PARAMS)


def _xgb_case_only():
    return ResidualXGBRegressor(feature_set=CASE_ONLY_FEATURES, shrinkage=SHRINKAGE, **XGB_PARAMS)


# Model name → (factory, feature_set_name)
MODELS = {
    "persistence": ("persistence", None),       # special; baseline
    "xgb_case_only": (_xgb_case_only, "case_only"),
    "xgb_full": (_xgb_full, "all_28"),
}

FEATURE_SETS = {
    "case_only": CASE_ONLY_FEATURES,
    "all_28": ALL_FEATURES,
}


def build_model(name: str):
    """Instantiate and return a fresh model by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS.keys())}")
    factory, feature_set = MODELS[name]
    if name == "persistence":
        return None  # persistence is handled specially (predict = cases_lag1)
    model = factory()
    return model


def get_feature_set(name: str) -> list[str]:
    """Return the feature list a model uses."""
    factory, fs_name = MODELS[name]
    if fs_name is None:
        return CASE_ONLY_FEATURES  # persistence uses cases_lag1 only
    return FEATURE_SETS[fs_name].copy()


def persistence_predict(X: pd.DataFrame) -> np.ndarray:
    """Persistence baseline: forecast = most recent observed value (cases_lag1)."""
    return X["cases_lag1"].values.astype(float)


def make_temporal_forward_chaining_folds(
    dates: pd.Series, n_splits: int = 3, min_train_ratio: float = 0.5
):
    """Generator of (train_idx, val_idx) using temporal forward chaining across dates.

    Ensures that for all cities simultaneously, the training period precedes
    the validation period chronologically.
    """
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    min_train_dates = int(n_dates * min_train_ratio)
    val_step = (n_dates - min_train_dates) // n_splits

    if val_step <= 0:
        raise ValueError(f"Too few unique dates ({n_dates}) for {n_splits} folds.")

    for i in range(n_splits):
        t_end = min_train_dates + i * val_step
        v_end = t_end + val_step if i < n_splits - 1 else n_dates
        tr_dates = set(unique_dates[:t_end])
        va_dates = set(unique_dates[t_end:v_end])

        train_idx = np.where(dates.isin(tr_dates))[0]
        val_idx = np.where(dates.isin(va_dates))[0]

        if len(val_idx) == 0:
            break
        yield train_idx, val_idx


# =====================================================================
# NOTE [3b] TRAINING (fit + tune).
#   Shows: grid over shrinkage/lr/n_estimators via forward-chaining CV on
#   train only (_forward_chain_cv), then fit on all train rows and predict
#   train+test (train_model), loop over persistence/xgb_case_only/xgb_full.
# Copied from: src/models/train.py
# =====================================================================

def _evaluate_y(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _forward_chain_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: list[str],
    dates_train: pd.Series | None = None,
) -> dict:
    """Tuning via temporal forward-chaining CV on training data only.

    Evaluates shrinkage and learning hyperparameters on the trajectory innovation
    residual task across multi-city temporal folds. Returns best params.
    """
    if dates_train is not None:
        folds = list(make_temporal_forward_chaining_folds(dates_train, n_splits=N_CV_FOLDS))
    else:
        n_samples = len(X_train)
        min_train_size = n_samples // 2
        test_size = (n_samples - min_train_size) // N_CV_FOLDS
        folds = []
        for i in range(N_CV_FOLDS):
            train_end = min_train_size + i * test_size
            val_end = train_end + test_size if i < N_CV_FOLDS - 1 else n_samples
            folds.append((np.arange(0, train_end), np.arange(train_end, val_end)))

    p_train = X_train["cases_lag1"].values.astype(float)
    y_vals = y_train.values.astype(float)
    res_train = y_vals - p_train

    shrinkage_options = [0.2, 0.3, 0.4]
    lr_options = [0.02, 0.03, 0.05]
    n_est_options = [20, 30, 40]
    results = []

    for shrink in shrinkage_options:
        for lr in lr_options:
            for n_est in n_est_options:
                fold_maes = []
                for train_idx, val_idx in folds:
                    mdl = XGBRegressor(
                        objective="reg:absoluteerror",
                        max_depth=3,
                        learning_rate=lr,
                        n_estimators=n_est,
                        min_child_weight=3,
                        reg_alpha=5.0,
                        reg_lambda=5.0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    )
                    mdl.fit(X_train.iloc[train_idx][feature_set].values, res_train[train_idx])
                    pred_res = mdl.predict(X_train.iloc[val_idx][feature_set].values)
                    pred_val = np.maximum(0.0, p_train[val_idx] + shrink * pred_res)
                    fold_maes.append(mean_absolute_error(y_vals[val_idx], pred_val))

                avg_mae = np.mean(fold_maes)
                results.append({
                    "shrinkage": shrink,
                    "learning_rate": lr,
                    "n_estimators": n_est,
                    "avg_val_mae": avg_mae,
                })

    results_df = pd.DataFrame(results).sort_values("avg_val_mae")
    best = results_df.iloc[0]
    best_params = {
        "objective": "reg:absoluteerror",
        "max_depth": 3,
        "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]),
        "shrinkage": float(best["shrinkage"]),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 5.0,
        "reg_alpha": 5.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    print(f"  CV tuning complete. Best: depth=3, lr={best_params['learning_rate']}, "
          f"n_est={best_params['n_estimators']}, shrinkage={best_params['shrinkage']}, "
          f"CV MAE={best['avg_val_mae']:.2f}")
    return best_params


def train_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: dict,
    dates: pd.Series | None = None,
    tuning_params: dict | None = None,
) -> dict:
    """Train a single model on the training split, evaluate on test.

    Residual XGBoost models: hyperparameters tuned via temporal forward-chaining CV
    on training data only (2016–2019), then the full tuned model is trained on all
    training rows and evaluated on the held-out test set (2020–2021).
    """
    train_mask = splits["train"]
    test_mask = splits["test"]

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    dates_train = dates[train_mask] if dates is not None else None

    # Handle persistence baseline
    if name == "persistence":
        pred_train = persistence_predict(X_train)
        pred_test = persistence_predict(X_test)
        model = None
        feature_set = CASE_ONLY_FEATURES
        params = {}
    else:
        factory, fs_name = MODELS[name]
        if fs_name == "case_only":
            feature_set = CASE_ONLY_FEATURES.copy()
        else:
            feature_set = ALL_FEATURES.copy()

        if name.startswith("xgb"):
            # Tune on training data via temporal forward-chaining CV
            if tuning_params is None:
                tuning_params = _forward_chain_cv(
                    X_train, y_train, feature_set, dates_train=dates_train
                )
            params = tuning_params.copy()
            shrinkage = params.pop("shrinkage", SHRINKAGE)
            model = ResidualXGBRegressor(
                feature_set=feature_set, shrinkage=shrinkage, **params
            )
        else:
            raise ValueError(f"Unknown model name: {name}")

        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

    train_metrics = _evaluate_y(y_train.values, pred_train)
    test_metrics = _evaluate_y(y_test.values, pred_test)

    return {
        "name": name,
        "model": model,
        "params": params,
        "feature_set": feature_set,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "predictions": {
            "train": pred_train,
            "test": pred_test,
        },
    }


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    splits: dict,
    dates: pd.Series | None = None,
) -> list[dict]:
    """Train all models and return results in order.

    Order: persistence → xgb_case_only → xgb_full
    """
    print("=" * 60)
    print("TRAINING MODELS (leakage-free temporal design)")
    print(f"  Train: 2016–2019  |  Test: 2020–2021  |  Embargo: 2 weeks")
    print("=" * 60)

    results = []
    for name in ["persistence", "xgb_case_only", "xgb_full"]:
        print(f"\n  Training: {name}")
        result = train_model(name, X, y, splits, dates=dates)
        results.append(result)

        tm = result["train_metrics"]
        rm = result["test_metrics"]
        print(f"    Train MAE={tm['MAE']:.1f}, RMSE={tm['RMSE']:.1f}, R²={tm['R2']:.3f}")
        print(f"    Test  MAE={rm['MAE']:.1f}, RMSE={rm['RMSE']:.1f}, R²={rm['R2']:.3f}")

    return results


# =====================================================================
# NOTE [4] TESTING (evaluation on held-out test set).
#   Shows: pooled + per-city MAE/RMSE/R2, city-stratified 4-week block-bootstrap
#   95% CIs, weather ablation (full minus case-only) and vs-persistence ratios.
# Copied from: src/evaluation/evaluate.py
# =====================================================================

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


# =====================================================================
# NOTE [Workflow driver] — runs the stages in order, same as pipeline.py steps 1-7:
#   PRE-PROCESS (load_real_panel -> validate_data -> build features)
#   -> SPLITTING (splits dict) -> TRAINING (train_all_models) -> TESTING (evaluate_all_models).
# =====================================================================

def main():
    # PRE-PROCESS
    df = load_real_panel()
    validate_data(df)
    # SPLITTING (inside build_feature_matrix)
    X, y, city, splits, dates = build_feature_matrix(df)
    # TRAINING
    results = train_all_models(X, y, splits, dates=dates)
    # TESTING
    report = evaluate_all_models(results, splits, X, y, city)
    return report


if __name__ == "__main__":
    main()
