"""
Training module with leakage-free forward-chaining CV.

Implements proposal §5.8: hyperparameters are tuned only with
forward-chaining cross-validation on the training data (2016–2019).
The test year (2020–2021) is never seen during tuning or fitting.

Models trained:
    1. Persistence baseline  — predict = cases_lag1 (most recent observed)
    2. Case-only XGBoost     — case-momentum + seasonality (6 features)
    3. Full XGBoost          — all 28 engineered features (incl. weather)

Usage:
    from src.models.train import train_all_models
    results = train_all_models(X, y, splits, dates=dates)
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from config import (
    N_CV_FOLDS,
    SHRINKAGE,
    XGB_PARAMS,
    CASE_ONLY_FEATURES,
    ALL_FEATURES,
)
from src.models.models import (
    MODELS,
    ResidualXGBRegressor,
    persistence_predict,
    make_temporal_forward_chaining_folds,
)

warnings.filterwarnings("ignore")


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
