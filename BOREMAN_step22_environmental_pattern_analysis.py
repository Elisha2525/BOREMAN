"""
BOREMAN STEP 22 - ENVIRONMENTAL PATTERN ANALYSIS

Input:
  outputs/step21_temporal_stability/08_all_candidates_with_temporal_flags.csv

Purpose:
  - Describe daily and hourly TDS, humidity and temperature patterns.
  - Compare fixed-clock daytime (06:00-18:00) and nighttime (18:00-06:00)
    using Namibia local time.
  - Compare plateau-review and non-plateau candidate observations.
  - Compare all candidates with the non-plateau subset.
  - Preserve every input record and all existing review flags.

Important:
  - Day/night is a fixed clock-time grouping, NOT an astronomical sunrise/sunset
    calculation.
  - Statistical review flags are not confirmed sensor faults.
  - This step is descriptive. It does not delete, impute, or correct observations.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

INPUT_FILE = Path("outputs/step21_temporal_stability/08_all_candidates_with_temporal_flags.csv")
OUTPUT_DIR = Path("outputs/step22_environmental_pattern_analysis")

TIME_COL = "timestamp_namibia"
SENSORS = ["tds", "humidity", "Temperature"]
EXPECTED_ROWS = 51437


def stop(message):
    raise ValueError(message)


def load_and_validate():
    if not INPUT_FILE.exists():
        stop(
            f"Input file not found: {INPUT_FILE}\n"
            "Run this script from the project root folder."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "source_row_number", TIME_COL, *SENSORS,
        "flag_constant_plateau", "flag_any_iqr_review",
        "flag_any_abrupt_change_review", "flag_duplicate_timestamp_review",
        "eligible_for_environmental_analysis"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        stop(f"Required columns missing: {missing}")

    if df["source_row_number"].isna().any():
        stop("source_row_number contains missing values.")
    if df["source_row_number"].duplicated().any():
        stop("source_row_number is not unique; stopping to protect traceability.")

    if len(df) != EXPECTED_ROWS:
        print(
            f"WARNING: Expected {EXPECTED_ROWS:,} records based on the reviewed "
            f"Step 21 file; found {len(df):,}. Continuing only if validations pass."
        )

    for col in SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # All timestamps were confirmed to be Namibia-local (+02:00) in Step 21.
    # Parse with utc=True for safe handling of the explicit offset, then convert.
    parsed = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    if parsed.isna().any():
        stop(f"{parsed.isna().sum()} invalid {TIME_COL} values found.")
    df["_time"] = parsed.dt.tz_convert("Africa/Windhoek")

    if df[SENSORS].isna().any().any():
        stop("Missing sensor values found in the candidate input; investigate first.")

    if not df["eligible_for_environmental_analysis"].astype(bool).all():
        stop("Input contains records not marked eligible; investigate before analysis.")

    # Keep the input's pre-existing flags and classifications unchanged.
    df["analysis_group"] = np.where(
        df["flag_constant_plateau"].astype(bool),
        "PLATEAU_REVIEW",
        "NON_PLATEAU_CANDIDATE"
    )

    # Fixed clock grouping in Namibia time; not a solar daylight estimate.
    hour = df["_time"].dt.hour
    df["clock_period"] = np.where(
        (hour >= 6) & (hour < 18),
        "DAYTIME_06_00_TO_18_00",
        "NIGHTTIME_18_00_TO_06_00"
    )

    df["namibia_date"] = df["_time"].dt.strftime("%Y-%m-%d")
    df["namibia_hour"] = df["_time"].dt.strftime("%Y-%m-%d %H:00")

    return df


def summarize(grouped, group_cols):
    agg = grouped[SENSORS].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    agg.columns = [f"{sensor}_{stat}" for sensor, stat in agg.columns]
    return agg.reset_index()


def correlation_table(df, label):
    corr = df[SENSORS].corr(method="pearson")
    rows = []
    for a, b in [
        ("tds", "humidity"),
        ("tds", "Temperature"),
        ("humidity", "Temperature")
    ]:
        rows.append({
            "dataset": label,
            "sensor_1": a,
            "sensor_2": b,
            "pearson_correlation": corr.loc[a, b],
            "records_used": int(df[[a, b]].dropna().shape[0]),
            "interpretation_note": "Association only; correlation does not establish causation."
        })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_validate()
    original_count = len(df)
    original_source_ids = set(df["source_row_number"].tolist())

    # Sort only for output readability; no observations are removed.
    df = df.sort_values(["_time", "source_row_number"]).reset_index(drop=True)

    # 1. Daily summaries across all environmental candidates.
    daily = summarize(df.groupby("namibia_date", sort=True), ["namibia_date"])
    daily.to_csv(OUTPUT_DIR / "01_daily_environmental_summary.csv", index=False)

    # 2. Hour-of-day summaries by Namibia date-hour.
    hourly = summarize(df.groupby("namibia_hour", sort=True), ["namibia_hour"])
    hourly.to_csv(OUTPUT_DIR / "02_hourly_environmental_summary.csv", index=False)

    # 3. Fixed clock daytime vs nighttime summaries.
    day_night = summarize(
        df.groupby("clock_period", sort=True),
        ["clock_period"]
    )
    day_night.to_csv(OUTPUT_DIR / "03_day_night_comparison.csv", index=False)

    # 4. Plateau-review vs non-plateau candidate comparison.
    plateau_comparison = summarize(
        df.groupby("analysis_group", sort=True),
        ["analysis_group"]
    )
    plateau_comparison.to_csv(
        OUTPUT_DIR / "04_plateau_vs_non_plateau_summary.csv", index=False
    )

    # 5. Daily summaries split by plateau status.
    daily_group = summarize(
        df.groupby(["namibia_date", "analysis_group"], sort=True),
        ["namibia_date", "analysis_group"]
    )
    daily_group.to_csv(
        OUTPUT_DIR / "05_daily_plateau_comparison.csv", index=False
    )

    # 6. Day/night summaries split by plateau status.
    day_night_group = summarize(
        df.groupby(["clock_period", "analysis_group"], sort=True),
        ["clock_period", "analysis_group"]
    )
    day_night_group.to_csv(
        OUTPUT_DIR / "06_day_night_plateau_comparison.csv", index=False
    )

    # 7. Pearson correlations: all candidates vs non-plateau subset.
    corr_rows = []
    corr_rows.extend(correlation_table(df, "ALL_ENVIRONMENTAL_CANDIDATES"))
    non_plateau = df[df["analysis_group"] == "NON_PLATEAU_CANDIDATE"]
    corr_rows.extend(correlation_table(non_plateau, "NON_PLATEAU_CANDIDATES"))
    pd.DataFrame(corr_rows).to_csv(
        OUTPUT_DIR / "07_sensor_correlation_comparison.csv", index=False
    )

    # 8. Record-level analysis view. Original columns/flags retained; helper
    # columns are added for traceability and later plotting/modelling decisions.
    record_view = df.drop(columns=["_time"]).copy()
    record_view.to_csv(
        OUTPUT_DIR / "08_records_with_environmental_groups.csv", index=False
    )

    # 9. Counts by date and analysis group, including review-flag counts.
    count_rows = []
    for (date, group_name), part in df.groupby(
        ["namibia_date", "analysis_group"], sort=True
    ):
        count_rows.append({
            "namibia_date": date,
            "analysis_group": group_name,
            "record_count": len(part),
            "iqr_review_records": int(part["flag_any_iqr_review"].astype(bool).sum()),
            "abrupt_change_review_records": int(
                part["flag_any_abrupt_change_review"].astype(bool).sum()
            ),
            "duplicate_timestamp_review_records": int(
                part["flag_duplicate_timestamp_review"].astype(bool).sum()
            ),
            "zero_tds_records": int((part["tds"] == 0).sum())
        })
    pd.DataFrame(count_rows).to_csv(
        OUTPUT_DIR / "09_daily_quality_flag_counts.csv", index=False
    )

    # Preservation checks.
    if len(df) != original_count:
        stop("Record count changed during Step 22.")
    if set(df["source_row_number"].tolist()) != original_source_ids:
        stop("Source record identity changed during Step 22.")
    if df["source_row_number"].duplicated().any():
        stop("Duplicate source_row_number detected after processing.")
    if int(df["flag_constant_plateau"].astype(bool).sum()) != 30430:
        print(
            "WARNING: Plateau flag count differs from reviewed Step 21 count "
            "(30,430). Verify whether the input has changed."
        )

    report = []
    report.append("BOREMAN STEP 22 - ENVIRONMENTAL PATTERN ANALYSIS")
    report.append("=" * 58)
    report.append(f"Input file: {INPUT_FILE}")
    report.append(f"Output directory: {OUTPUT_DIR}")
    report.append(f"Input records: {original_count:,}")
    report.append(f"Output record-view records: {len(record_view):,}")
    report.append(f"Unique source records: {df['source_row_number'].nunique():,}")
    report.append(
        f"Namibia local date range: {df['namibia_date'].min()} to "
        f"{df['namibia_date'].max()}"
    )
    report.append(f"Plateau-review records: {(df['analysis_group'] == 'PLATEAU_REVIEW').sum():,}")
    report.append(
        f"Non-plateau candidate records: "
        f"{(df['analysis_group'] == 'NON_PLATEAU_CANDIDATE').sum():,}"
    )
    report.append(f"IQR review records: {df['flag_any_iqr_review'].astype(bool).sum():,}")
    report.append(
        f"Abrupt-change review records: "
        f"{df['flag_any_abrupt_change_review'].astype(bool).sum():,}"
    )
    report.append(
        f"Duplicate-timestamp review records: "
        f"{df['flag_duplicate_timestamp_review'].astype(bool).sum():,}"
    )
    report.append("")
    report.append("TIME GROUPING")
    report.append("Daytime: 06:00 inclusive to 18:00 exclusive, Namibia local clock.")
    report.append("Nighttime: 18:00 inclusive to 06:00 exclusive, Namibia local clock.")
    report.append("This is a fixed clock-time grouping, not astronomical daylight.")
    report.append("")
    report.append("METHOD LIMITATIONS")
    report.append("- Descriptive statistics only; no deletion or imputation.")
    report.append("- Plateau-review records remain included and separately labelled.")
    report.append("- IQR and abrupt-change flags are review indicators, not confirmed faults.")
    report.append("- Pearson correlation describes linear association, not causation.")
    report.append("- Unequal sampling intervals mean record counts are not time-weighted.")
    report.append("- No interpolation across gaps and no claim of continuous coverage.")
    report.append("")
    report.append("PRESERVATION CHECKS")
    report.append("PASS: all source records preserved.")
    report.append("PASS: source_row_number remains unique.")
    report.append("PASS: all timestamps parsed and grouped in Africa/Windhoek time.")
    report.append("PASS: no sensor values deleted or imputed.")
    report.append("")
    report.append("OUTPUT FILES")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        report.append(f"- {path.name}")
    (OUTPUT_DIR / "10_step22_environmental_pattern_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nSTEP 22 COMPLETED")
    print(f"Input records: {original_count:,}")
    print(f"Records preserved: {len(record_view):,}")
    print(f"Daily summaries: {len(daily):,}")
    print(f"Hourly summaries: {len(hourly):,}")
    print(f"Plateau-review records: {(df['analysis_group'] == 'PLATEAU_REVIEW').sum():,}")
    print(f"Non-plateau records: {(df['analysis_group'] == 'NON_PLATEAU_CANDIDATE').sum():,}")
    print("All preservation checks passed.")
    print(f"Outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSTEP 22 STOPPED: {exc}")
        sys.exit(1)
