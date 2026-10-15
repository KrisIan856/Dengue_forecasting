"""
Model definitions for dengue forecasting.

Implements the three approaches from proposal §5.7:
  1. Persistence baseline  — predicts most recent observed value
  2. Case-only XGBoost     — case-momentum + seasonality (6 features)
  3. Full XGBoost          — all 28 engineered features

Usage:
    from src.models.models import build_model, MODELS
    model = build_model("xgb_full")
"""

from xgboost import XGBRegressor
import pandas as pd
import numpy as np

from config import XGB_PARAMS, SHRINKAGE, CASE_ONLY_FEATURES, ALL_FEATURES


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


# ── Model registry ─────────────────────────────────────────────────────────
# Each entry: (factory_func, feature_set_name)
# feature_set is either 'case_only' or 'all_28'

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


# ── Forward-chaining cross-validation (Temporal CV) ──────────────────────────
# Strictly chronological split across dates to evaluate without temporal leakage.

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
