"""
Feature engineering module.

Builds the 28-predictor feature matrix and the t+2 target as described
in proposal §5.6 and §5.5. Every feature uses only information available
at or before the forecast origin — no future case counts or weather leak.

Usage:
    from src.features.build_features import build_feature_matrix
    X, y, city, splits, dates = build_feature_matrix(df)
"""

import numpy as np
import pandas as pd

from config import (
    ALL_FEATURES,
    CASE_ONLY_FEATURES,
    FORECAST_HORIZON,
    TRAIN_END_YEAR,
    TEST_START_YEAR,
    EMBARGO_WEEKS,
)


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
    """Cyclical sin/cos encoding of week number (proposal §5.1, §5.6)."""
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
