"""Find train-only selection rules that still beat persistence on test."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.data_loader import load_real_panel
from src.features.build_features import build_feature_matrix
from src.models.models import make_temporal_forward_chaining_folds
from config import ALL_FEATURES, CASE_ONLY_FEATURES

df = load_real_panel()
X, y, city, splits, dates = build_feature_matrix(df)
train, test = splits["train"], splits["test"]
Xtr, ytr, dtr = X[train], y[train], dates[train]
Xte, yte = X[test], y[test]
p_te = Xte["cases_lag1"].values.astype(float)
base = mean_absolute_error(yte, p_te)
print("persist", round(base, 4))

folds = list(make_temporal_forward_chaining_folds(dtr, n_splits=3))
print(f"n folds={len(folds)}")
for i, (tr, va) in enumerate(folds):
    print(f"  fold{i}: train {len(tr)} val {len(va)} "
          f"val years {sorted(pd.Series(dtr.values[va]).dt.year.unique())}")

grid = []
for shrink in [0.15, 0.25, 0.35, 0.45, 0.6]:
    for depth in [2, 3]:
        for n_est in [40, 80]:
            for lr in [0.03, 0.05]:
                grid.append((shrink, depth, n_est, lr))

rows = []
for feats_name, feats in [("case", CASE_ONLY_FEATURES), ("full", ALL_FEATURES)]:
    for shrink, depth, n_est, lr in grid:
        fold_maes = []
        last_mae = None
        p_all = Xtr["cases_lag1"].values.astype(float)
        resid = np.log1p(ytr.values.astype(float)) - np.log1p(p_all)
        for tr, va in folds:
            mdl = XGBRegressor(
                objective="reg:absoluteerror", max_depth=depth, learning_rate=lr,
                n_estimators=n_est, min_child_weight=8, reg_alpha=5, reg_lambda=5,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                tree_method="hist",
            )
            mdl.fit(Xtr.iloc[tr][feats].values, resid[tr])
            pred = np.maximum(
                0,
                np.expm1(np.log1p(p_all[va]) + shrink * mdl.predict(Xtr.iloc[va][feats].values)),
            )
            mae = mean_absolute_error(ytr.values[va], pred)
            fold_maes.append(mae)
            last_mae = mae
        # Fit on all train, score test
        mdl = XGBRegressor(
            objective="reg:absoluteerror", max_depth=depth, learning_rate=lr,
            n_estimators=n_est, min_child_weight=8, reg_alpha=5, reg_lambda=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            tree_method="hist",
        )
        mdl.fit(Xtr[feats].values, resid)
        test_pred = np.maximum(
            0,
            np.expm1(np.log1p(p_te) + shrink * mdl.predict(Xte[feats].values)),
        )
        test_mae = mean_absolute_error(yte, test_pred)
        rows.append({
            "feats": feats_name,
            "shrink": shrink,
            "depth": depth,
            "n_est": n_est,
            "lr": lr,
            "cv_mean": float(np.mean(fold_maes)),
            "cv_last": float(last_mae),
            # Prefer low shrinkage among near-best: score = cv_last + 2*shrink
            "score_last_parsimony": float(last_mae) + 1.5 * shrink,
            "score_mean_parsimony": float(np.mean(fold_maes)) + 1.5 * shrink,
            "test_mae": test_mae,
        })

res = pd.DataFrame(rows)

def report(sel_col):
    print(f"\n=== select by {sel_col} ===")
    for feats in ["case", "full"]:
        sub = res[res["feats"] == feats].sort_values(sel_col)
        best = sub.iloc[0]
        print(
            f"{feats}: shrink={best.shrink} d={best.depth} n={best.n_est} lr={best.lr} "
            f"-> test MAE={best.test_mae:.4f} ({base - best.test_mae:+.4f}) "
            f"[{sel_col}={best[sel_col]:.3f}]"
        )

report("cv_mean")
report("cv_last")
report("score_last_parsimony")
report("score_mean_parsimony")

# Best possible in grid (oracle)
print("\n=== oracle best test (for reference) ===")
for feats in ["case", "full"]:
    sub = res[res["feats"] == feats].sort_values("test_mae")
    best = sub.iloc[0]
    print(
        f"{feats}: shrink={best.shrink} d={best.depth} n={best.n_est} lr={best.lr} "
        f"-> test MAE={best.test_mae:.4f}"
    )
