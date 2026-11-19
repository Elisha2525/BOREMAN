"""
BOREMAN STEP 20 - Environmental Sensor Validation
-------------------------------------------------
Input:
  outputs/step19_data_quality_control/02_environmental_candidate_dataset.csv

This script performs descriptive statistical and time-series validation.
It does NOT delete records, impute values, or declare statistical outliers
to be confirmed sensor errors.

All time-based analysis uses Africa/Windhoek (Namibia local time).
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILE = Path("outputs/step19_data_quality_control/02_environmental_candidate_dataset.csv")
OUTPUT_DIR = Path("outputs/step20_environmental_validation")
SENSORS = ["tds", "humidity", "Temperature"]
TIME_COLUMN = "timestamp_namibia"
EXPECTED_ROWS = 51437


def main():
    print("\nBOREMAN STEP 20 STARTED")
    print("-" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
            "Run this script from the project root folder."
        )

    df = pd.read_csv(INPUT_FILE)
    original_count = len(df)
    print(f"Loaded candidate observations: {original_count:,}")

    required = [TIME_COLUMN, *SENSORS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse the already-converted Namibia timestamps and retain all source fields.
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce", utc=True)
    df[TIME_COLUMN] = df[TIME_COLUMN].dt.tz_convert("Africa/Windhoek")

    for col in SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stable ordering for time-series calculations; no records are removed.
    df = df.sort_values(TIME_COLUMN, kind="stable", na_position="last").reset_index(drop=True)
    df["step20_row_number"] = np.arange(1, len(df) + 1)

    # 1. Descriptive sensor statistics.
    summary_rows = []
    for col in SENSORS:
        s = df[col].dropna()
        if len(s):
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_count = int(((s < lower) | (s > upper)).sum())
            summary_rows.append({
                "sensor": col,
                "non_missing_count": int(s.count()),
                "missing_count": int(df[col].isna().sum()),
                "minimum": s.min(),
                "p01": s.quantile(0.01),
                "p05": s.quantile(0.05),
                "p25": q1,
                "median": s.median(),
                "mean": s.mean(),
                "p75": q3,
                "p95": s.quantile(0.95),
                "p99": s.quantile(0.99),
                "maximum": s.max(),
                "standard_deviation": s.std(),
                "iqr_lower_review_threshold": lower,
                "iqr_upper_review_threshold": upper,
                "iqr_outlier_count": iqr_count,
                "iqr_outlier_percent": (iqr_count / len(s) * 100) if len(s) else np.nan,
            })
        else:
            summary_rows.append({"sensor": col, "non_missing_count": 0})
    sensor_summary = pd.DataFrame(summary_rows)
    sensor_summary.to_csv(OUTPUT_DIR / "01_sensor_descriptive_summary.csv", index=False)

    # 2. Add review flags for IQR outliers. These are statistical flags only.
    for col in SENSORS:
        s = df[col]
        valid = s.dropna()
        if len(valid):
            q1, q3 = valid.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df[f"flag_{col}_iqr_review"] = s.notna() & ((s < lower) | (s > upper))
        else:
            df[f"flag_{col}_iqr_review"] = False

    flag_cols = [f"flag_{c}_iqr_review" for c in SENSORS]
    df["flag_any_iqr_review"] = df[flag_cols].any(axis=1)
    df.to_csv(OUTPUT_DIR / "02_all_candidates_with_statistical_flags.csv", index=False)

    outliers = df[df["flag_any_iqr_review"]].copy()
    outliers.to_csv(OUTPUT_DIR / "03_statistical_outlier_review_records.csv", index=False)

    # 3. Daily and hourly summaries, based strictly on Namibia local time.
    timed = df[df[TIME_COLUMN].notna()].copy()
    timed["namibia_date"] = timed[TIME_COLUMN].dt.strftime("%Y-%m-%d")
    timed["namibia_hour"] = timed[TIME_COLUMN].dt.strftime("%Y-%m-%d %H:00:00")

    daily = timed.groupby("namibia_date").agg(
        observation_count=("source_row_number", "size"),
        first_observation=(TIME_COLUMN, "min"),
        last_observation=(TIME_COLUMN, "max"),
        **{f"{c}_mean": (c, "mean") for c in SENSORS},
        **{f"{c}_median": (c, "median") for c in SENSORS},
        **{f"{c}_minimum": (c, "min") for c in SENSORS},
        **{f"{c}_maximum": (c, "max") for c in SENSORS},
    ).reset_index()
    daily.to_csv(OUTPUT_DIR / "04_daily_namibia_summary.csv", index=False)

    hourly = timed.groupby("namibia_hour").agg(
        observation_count=("source_row_number", "size"),
        **{f"{c}_mean": (c, "mean") for c in SENSORS},
        **{f"{c}_median": (c, "median") for c in SENSORS},
        **{f"{c}_minimum": (c, "min") for c in SENSORS},
        **{f"{c}_maximum": (c, "max") for c in SENSORS},
    ).reset_index()
    hourly.to_csv(OUTPUT_DIR / "05_hourly_namibia_summary.csv", index=False)

    # 4. Correlation matrix: descriptive association, not causation.
    corr = timed[SENSORS].corr(method="pearson")
    corr.to_csv(OUTPUT_DIR / "06_sensor_correlation_matrix.csv")

    # 5. Sampling interval audit, using actual Namibia timestamps.
    timed = timed.sort_values(TIME_COLUMN).copy()
    timed["interval_minutes"] = timed[TIME_COLUMN].diff().dt.total_seconds() / 60
    intervals = timed["interval_minutes"].dropna()
    if len(intervals):
        interval_summary = pd.DataFrame([{
            "interval_count": len(intervals),
            "minimum_minutes": intervals.min(),
            "p05_minutes": intervals.quantile(0.05),
            "median_minutes": intervals.median(),
            "mean_minutes": intervals.mean(),
            "p95_minutes": intervals.quantile(0.95),
            "maximum_minutes": intervals.max(),
            "intervals_over_5_minutes": int((intervals > 5).sum()),
            "intervals_over_60_minutes": int((intervals > 60).sum()),
        }])
    else:
        interval_summary = pd.DataFrame()
    interval_summary.to_csv(OUTPUT_DIR / "07_sampling_interval_summary.csv", index=False)

    # 6. Compact run report.
    report = [
        "BOREMAN STEP 20 - ENVIRONMENTAL SENSOR VALIDATION",
        "=" * 60,
        f"Input file: {INPUT_FILE}",
        f"Output directory: {OUTPUT_DIR}",
        f"Candidate rows loaded: {original_count:,}",
        f"Rows with valid Namibia timestamps: {df[TIME_COLUMN].notna().sum():,}",
        f"Rows with at least one IQR review flag: {int(df['flag_any_iqr_review'].sum()):,}",
        "",
        "Timezone used for all time-based summaries: Africa/Windhoek",
        "Statistical outliers are review flags, not confirmed sensor errors.",
        "No records were deleted or imputed.",
        "No physical sensor limits were assumed.",
        "",
        "SENSOR SUMMARY",
        sensor_summary.to_string(index=False),
        "",
        "CORRELATION MATRIX",
        corr.to_string(),
        "",
        "SAMPLING INTERVAL SUMMARY (minutes)",
        interval_summary.to_string(index=False) if len(interval_summary) else "No intervals available.",
        "",
        "Interpretation reminder:",
        "- IQR flags identify values for review, not automatic rejection.",
        "- Correlation does not establish causation.",
        "- Constant plateaus and zero TDS remain in the candidate data.",
        "- Confirm physical operating ranges only after sensor specifications are supplied.",
    ]
    (OUTPUT_DIR / "08_step20_validation_report.txt").write_text("\n".join(report), encoding="utf-8")

    # Preservation checks.
    if len(df) != original_count:
        raise AssertionError("Row count changed during Step 20.")
    if EXPECTED_ROWS and original_count != EXPECTED_ROWS:
        print(f"WARNING: Expected {EXPECTED_ROWS:,} rows, found {original_count:,}.")
    if df["source_row_number"].duplicated().any():
        raise AssertionError("Duplicate source_row_number values found.")

    print("\nSTEP 20 VALIDATION")
    print("-" * 60)
    print(f"Rows preserved: {len(df):,}")
    print(f"Rows flagged for IQR review: {int(df['flag_any_iqr_review'].sum()):,}")
    print(f"Daily summary rows: {len(daily):,}")
    print(f"Hourly summary rows: {len(hourly):,}")
    print("All row-preservation checks passed.")
    print("\nBOREMAN STEP 20 COMPLETED")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Please share 01_sensor_descriptive_summary.csv and 08_step20_validation_report.txt for review.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
