"""
Generate synthetic dengue + meteorological data for the Mindanao city panel.

Since the real DOH weekly surveillance counts and Project CCHAIN climate
variables are not bundled in this repository, this module creates a
realistic synthetic panel that mirrors the structure described in the
proposal (786 city-weeks across 3 cities for 2016–2021).

Run directly to produce data/synthetic_dengue_data.csv:
    python data/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR, CITIES, FORECAST_HORIZON

np.random.seed(42)

# ── Epidemiological parameters per city ────────────────────────────────────
# These control the seasonal pattern, baseline level, and noise of dengue
# cases so each city has a distinct but realistic profile.
CITY_PARAMS = {
    "Davao City": {
        "peak_month": 9,      # September peak (SW monsoon)
        "baseline": 120,      # weekly baseline
        "amplitude": 180,     # seasonal amplitude
        "noise_sd": 40,       # observation noise
        "weather_amp": 1.0,   # weather seasonality multiplier
    },
    "Cagayan de Oro City": {
        "peak_month": 10,
        "baseline": 80,
        "amplitude": 120,
        "noise_sd": 30,
        "weather_amp": 0.9,
    },
    "Zamboanga City": {
        "peak_month": 8,
        "baseline": 100,
        "amplitude": 150,
        "noise_sd": 35,
        "weather_amp": 1.1,
    },
}


def _epidemiological_week(iso_year: int, week: int) -> pd.Timestamp:
    """Approximate the start date of an ISO epidemiological week."""
    return pd.Timestamp.fromisocalendar(iso_year, week, 1)


def _seasonal_factor(week: int, peak_month: int) -> float:
    """A smooth seasonal bump centred on peak_month (1–12)."""
    week_of_year = week
    # Convert peak month to an approximate week
    peak_week = int(peak_month * 4.345)
    # Normalised distance in weeks (period = 52)
    dist = min(
        abs(week_of_year - peak_week),
        abs(week_of_year - peak_week - 52),
        abs(week_of_year - peak_week + 52),
    )
    # Bell-shaped seasonal multiplier
    return np.exp(-((dist / 6.0) ** 2))


def _weather_for_week(week: int, amp: float, city_seed: int) -> dict:
    """Generate plausible weekly rainfall, temperature, humidity."""
    rng = np.random.default_rng(city_seed + week)
    # Rainfall: seasonal, peak around mid-year / near dengue peak
    rain_base = 20 + 80 * np.exp(-((min(
        abs(week - 32), abs(week - 32 - 52), abs(week - 32 + 52)
    ) / 8.0) ** 2)) * amp
    rainfall = max(0, rng.normal(rain_base, 15))
    # Temperature: relatively stable, slight seasonal variation
    tmean = rng.normal(28.5 + 2 * np.sin(2 * np.pi * (week - 60) / 52), 1.0)
    tmin = tmean - rng.normal(3, 0.5)
    tmax = tmean + rng.normal(3, 0.5)
    # Humidity: counter-seasonal to temperature
    humidity = 75 + 10 * np.cos(2 * np.pi * (week - 30) / 52) + rng.normal(0, 5)
    humidity = np.clip(humidity, 50, 95)
    return {
        "rainfall": round(rainfall, 1),
        "temperature": round(tmean, 1),
        "tmin": round(tmin, 1),
        "tmax": round(tmax, 1),
        "humidity": round(humidity, 1),
    }


def generate_panel(start_year: int = 2016, end_year: int = 2021) -> pd.DataFrame:
    """Build the full city-by-week panel."""
    rows = []
    for city in CITIES:
        p = CITY_PARAMS[city]
        seed = hash(city) % (2**32)
        for year in range(start_year, end_year + 1):
            for week in range(1, 53):
                # Skip incomplete weeks near year-end for 2021 (only need through Dec)
                if year == end_year and week > 52:
                    continue

                weather = _weather_for_week(week, p["weather_amp"], seed)
                # Seasonal dengue driver
                seasonal = _seasonal_factor(week, p["peak_month"])
                # Autoregressive component: carry over from previous week
                # (handled below via noise on the seasonal signal)

                # Base cases with seasonal peak
                cases = p["baseline"] + p["amplitude"] * seasonal
                # Add weather-driven component (lagged effect)
                cases += 0.3 * weather["rainfall"] * seasonal
                # Temperature effect (within viability range)
                temp_effect = -0.5 * (weather["temperature"] - 28.5) ** 2 + 5
                cases += temp_effect * seasonal
                # Add noise
                cases = max(0, np.random.normal(cases, p["noise_sd"]))

                # COVID dip in 2020 (reduced reporting / reduced transmission)
                if year == 2020:
                    cases *= 0.7
                    # Add extra noise during pandemic
                    cases = max(0, np.random.normal(cases, p["noise_sd"] * 1.5))

                rows.append({
                    "city": city,
                    "year": year,
                    "week_number": week,
                    "date": _epidemiological_week(year, week),
                    "weekly_dengue_cases": round(cases),
                    **weather,
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df


def add_derived_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Add bioclimatic indicators described in the proposal.

    heat_index, diurnal_temp_range are computed per-row.
    spi3, pnp, ndvi are added as synthetic weekly fields.
    """
    # Heat index (Rothfusz regression, simplified for warm climates)
    T = df["temperature"]
    R = df["humidity"]
    df["heat_index"] = round(
        -8.784694 + 1.61139411 * T + 2.33854883889 * R
        - 0.14611605 * T * R - 0.002598188 * T**2
        - 0.00330699 * R**2 + 0.000211784 * T * R**2
        + 0.000004032 * T**2 * R, 2
    )

    # Diurnal temperature range
    df["diurnal_temp_range"] = round(df["tmax"] - df["tmin"], 2)

    # Synthetic NDVI (vegetation greenness, higher in wet season)
    np.random.seed(77)
    seasonal_green = 0.4 + 0.3 * np.exp(-((np.minimum(
        np.abs(df["week_number"] - 32),
        np.abs(df["week_number"] - 32 - 52),
        np.abs(df["week_number"] - 32 + 52),
    )) / 10.0)**2)
    df["ndvi"] = np.clip(seasonal_green + np.random.normal(0, 0.05, len(df)), 0.1, 0.9).round(3)

    # SPI-3 (Standardized Precipitation Index, 3-month) — synthetic
    # Computed as a rolling z-score of rainfall over a 12-week window
    df["spi3"] = (
        df.groupby("city")["rainfall"]
        .transform(lambda x: (x.rolling(12, min_periods=1).mean() - x.rolling(12, min_periods=1).mean().mean())
                   / (x.rolling(12, min_periods=1).std() + 1e-6))
        .round(3)
    )

    # PNP (Percent of Normal Precipitation) — synthetic
    long_term_rain = df.groupby("city")["rainfall"].transform(
        lambda x: x.rolling(52, min_periods=1).mean() + 1e-6
    )
    df["pnp"] = (df["rainfall"] / long_term_rain * 100).clip(0, 300).round(1)

    return df


if __name__ == "__main__":
    print("Generating synthetic dengue + climate panel (2016–2021)...")
    df = generate_panel()
    df = add_derived_weather(df)

    # Ensure all expected raw columns exist before feature engineering
    expected_cols = {
        "city", "year", "week_number", "date", "weekly_dengue_cases",
        "rainfall", "temperature", "tmin", "tmax", "humidity",
        "heat_index", "diurnal_temp_range", "ndvi", "spi3", "pnp",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    out_path = DATA_DIR / "synthetic_dengue_data.csv"
    df.to_csv(out_path, index=False)
    n_cities = df["city"].nunique()
    n_weeks = df["week_number"].nunique()
    n_years = df["year"].nunique()
    print(f"  Cities: {n_cities}, Years: {n_years}, Weeks/yr: ~{n_weeks}")
    print(f"  Total rows: {len(df)}")
    print(f"  Written to: {out_path}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nColumns: {list(df.columns)}")
