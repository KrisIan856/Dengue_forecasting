"""
Data validation module.

Implements the checks described in proposal §5.4:
  1. Schema completeness - all expected columns present.
  2. Missing-week check - every city-year should have 52 weeks.
  3. Plausibility review - case counts and weather within sane bounds.
  4. Data-quality report - printed and saved as JSON.

Usage:
    from src.features.validate_data import validate_data
    report = validate_data(df)
"""

import json
import pandas as pd
from typing import Any

from config import OUTPUT_DIR, MAX_MISSING_PCT


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

    # Temperature: 15?45 ?C is sane for the Philippines
    temp_oob = df_flagged[(df_flagged["temperature"] < 15) | (df_flagged["temperature"] > 45)]
    if len(temp_oob) > 0:
        issues.append(f"Temperature out of [15,45]?C: {len(temp_oob)} rows")

    # Humidity: 20?100%
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


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
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


if __name__ == "__main__":
    from config import RAW_DATA
    df = pd.read_csv(RAW_DATA)
    validate_data(df)
