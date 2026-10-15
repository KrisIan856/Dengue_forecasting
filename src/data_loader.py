"""
Real-data loader for the Mindanao dengue panel.

Merges the per-city weekly panels that ship in ``<env>/datasets/`` into a
single tidy city-by-week DataFrame, reconstructs the ISO ``date`` column
that feature engineering needs, and applies documented cleaning rules for
the data-quality defects present in the source files.

Source files (one per city):
    datasets/dengue_davao_city_week_panel_2016_2021.csv
    datasets/dengue_cagayan_de_oro_city_week_panel_2016_2021.csv
    datasets/dengue_zamboanga_city_week_panel_2016_2021.csv

Cleaning rules applied by default:
    1. Duplicate (city, year, week_number) records are collapsed by taking
       the median of the reported case counts (the duplicates carry
       conflicting values, so the mean is the least-arbitrary choice).
    2. Rows whose weekly_dengue_cases is NaN are dropped (flagged
       ``missing_target`` in the source files; the t+2 target cannot be
       built from them).
    3. Rows with epidemiological week 53 are dropped (ISO weeks are 1-52;
       these are spurious continuation records).
    4. Statistical outliers in the case series are winsorised (capped)
       rather than deleted, so genuine outbreak weeks still contribute to
       the training signal while a single mis-keyed value cannot dominate
       the loss. Set ``OUTLIER_MODE = "drop"`` or ``"keep"`` to change this.

Usage:
    from src.data_loader import load_real_panel
    df = load_real_panel()
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable so `config` resolves when this module is
# executed directly (`python src/data_loader.py`) as well as when imported
# from pipeline.py with the root already on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    REAL_DATA_DIR,
    CITY_FILE_PATTERNS,
    OUTLIER_MODE,
    OUTLIER_IQR_FACTOR,
    BASE_DIR,
    DATA_DIR,
)

# Columns required by src.features.validate_data / build_features.
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


def build_merged_csv(out_path: Path | None = None, verbose: bool = True) -> Path:
    """Load the real panel and write the merged CSV consumed by the pipeline.

    The merged file is written to ``data/merged_dengue_panel_real.csv`` and
    its path is returned, so ``config.RAW_DATA`` can point at it.
    """
    if out_path is None:
        out_path = DATA_DIR / "merged_dengue_panel_real.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_real_panel(verbose=verbose)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"  wrote merged panel -> {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    build_merged_csv()