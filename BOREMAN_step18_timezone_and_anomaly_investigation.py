
"""
BOREMAN - Step 18: Timezone Normalization and Sensor Anomaly Investigation

Source:
    outputs/live_timeseries/01_original_export.csv

Timezone assumption:
    Imported timestamps represent IST (Asia/Kolkata).
    Actual sensor recording timezone is Namibia (Africa/Windhoek).

Important:
    - Preserve original observations.
    - Do not impute or delete sensor readings.
    - Confirmed testing observations are classified separately.
    - Other constant-value periods are investigated, not automatically rejected.
    - All analysis timestamps are converted to Namibia local time.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

INPUT_FILE = Path("outputs/live_timeseries/01_original_export.csv")

OUTPUT_DIR = Path("outputs") / "step18_timezone_anomaly_investigation"

SOURCE_TIMEZONE = "Asia/Kolkata"
TARGET_TIMEZONE = "Africa/Windhoek"

SENSOR_COLUMNS = ["tds", "humidity", "Temperature"]

# Minimum consecutive identical sensor observations
# before reporting a constant-value plateau.
PLATEAU_MIN_RECORDS = 30

# Sampling gaps are assessed against local sampling cadence.
MIN_GAP_THRESHOLD_MINUTES = 5
GAP_MULTIPLIER = 3
ROLLING_INTERVAL_COUNT = 20


# ---------------------------------------------------------
# 2. MAIN FUNCTION
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}"
        )

    print("\nBOREMAN STEP 18 STARTED")
    print("-" * 55)

    # -----------------------------------------------------
    # 3. LOAD ORIGINAL DATA
    # -----------------------------------------------------

    data = pd.read_csv(INPUT_FILE)

    required = ["Timestamp"] + SENSOR_COLUMNS

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Preserve original source row reference.
    data.insert(
        0,
        "source_row_number",
        range(2, len(data) + 2)
    )

    # Preserve raw timestamp and sensor values.
    data["timestamp_original"] = data["Timestamp"]

    for column in SENSOR_COLUMNS:
        data[f"{column}_original"] = data[column]

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # 4. PARSE IST TIMESTAMPS
    # -----------------------------------------------------

    parsed = pd.to_datetime(
        data["Timestamp"],
        format="mixed",
        errors="coerce"
    )

    # Input timestamps are assumed to be naive IST values.
    # Do not localize timestamps that are already timezone-aware.
    if parsed.dt.tz is None:
        timestamp_ist = parsed.dt.tz_localize(
            SOURCE_TIMEZONE,
            ambiguous="NaT",
            nonexistent="NaT"
        )
    else:
        timestamp_ist = parsed.dt.tz_convert(
            SOURCE_TIMEZONE
        )

    data["timestamp_ist"] = timestamp_ist

    data["timestamp_namibia"] = (
        data["timestamp_ist"].dt.tz_convert(TARGET_TIMEZONE)
    )

    data["flag_invalid_timestamp"] = (
        data["timestamp_namibia"].isna()
    )

    print(f"Original observations: {len(data):,}")

    print(
        "Invalid timestamps:",
        int(data["flag_invalid_timestamp"].sum())
    )

    # -----------------------------------------------------
    # 5. IDENTIFY CONFIRMED TESTING OBSERVATIONS
    # -----------------------------------------------------

    # The user confirmed the repeated combination below
    # represents sensor-testing observations.

    data["flag_confirmed_testing"] = (
        data["tds"].eq(45)
        & data["humidity"].eq(45)
        & data["Temperature"].eq(70)
        & data["timestamp_namibia"].notna()
    )

    testing = data[
        data["flag_confirmed_testing"]
    ].copy()

    print(
        "Confirmed testing records:",
        len(testing)
    )

    testing.to_csv(
        OUTPUT_DIR / "01_confirmed_testing_records.csv",
        index=False
    )

    # -----------------------------------------------------

    # -----------------------------------------------------
    # 6. IDENTIFY CONSTANT-VALUE PLATEAUS
    # -----------------------------------------------------

    valid = data[
        data["timestamp_namibia"].notna()
    ].copy()

    valid = valid.sort_values(
        "timestamp_namibia",
        kind="stable"
    ).copy()

    # Calculate timestamp intervals to detect discontinuities.
    valid["interval_minutes"] = (
        valid["timestamp_namibia"]
        .diff()
        .dt.total_seconds() / 60
    )

    # Calculate the local sampling cadence using the
    # previous 20 intervals.
    valid["previous_20_interval_median"] = (
        valid["interval_minutes"]
        .shift(1)
        .rolling(
            ROLLING_INTERVAL_COUNT,
            min_periods=3
        )
        .median()
    )

    global_median = valid["interval_minutes"].median()

    valid["expected_interval_minutes"] = (
        valid["previous_20_interval_median"]
        .fillna(global_median)
    )

    valid["gap_threshold_minutes"] = (
        valid["expected_interval_minutes"]
        .mul(GAP_MULTIPLIER)
        .clip(lower=MIN_GAP_THRESHOLD_MINUTES)
    )

    # A continuous interval must not exceed the local
    # sampling-gap threshold.
    continuous_interval = (
        valid["interval_minutes"]
        .le(valid["gap_threshold_minutes"])
        .fillna(False)
    )

    # Confirm that all three sensor values are available.
    sensor_values_present = (
        valid[SENSOR_COLUMNS].notna().all(axis=1)
    )

    # Confirmed testing observations must not be included
    # in environmental plateau detection.
    not_testing = ~valid["flag_confirmed_testing"]

    # Sensor values must match the previous observation.
    same_sensor_values = (
        valid[SENSOR_COLUMNS]
        .eq(valid[SENSOR_COLUMNS].shift())
        .all(axis=1)
    )

    previous_not_testing = (
        not_testing.shift(1).fillna(False)
    )

    previous_sensor_values_present = (
        sensor_values_present.shift(1).fillna(False)
    )

    # Continue a plateau only when:
    # 1. All three sensor values are identical.
    # 2. Current and previous observations are not testing.
    # 3. Current and previous sensor values are valid.
    # 4. There is no significant timestamp gap.
    same_as_previous = (
        same_sensor_values
        & not_testing
        & previous_not_testing
        & sensor_values_present
        & previous_sensor_values_present
        & continuous_interval
    )

    # A new segment begins whenever continuity is broken.
    valid["plateau_run_id"] = (
        (~same_as_previous).cumsum()
    )

    plateau_segments = []
    plateau_source_rows = set()

    for segment_id, group in valid.groupby(
        "plateau_run_id",
        sort=False
    ):

        if len(group) < PLATEAU_MIN_RECORDS:
            continue

        # Never include confirmed testing observations.
        if group["flag_confirmed_testing"].any():
            continue

        # All three sensor values must be present.
        if group[SENSOR_COLUMNS].isna().any().any():
            continue

        plateau_source_rows.update(
            group["source_row_number"].tolist()
        )

        intervals = (
            group["timestamp_namibia"]
            .diff()
            .dropna()
            .dt.total_seconds()
        )

        plateau_segments.append({
            "segment_id": int(segment_id),
            "start_namibia": group["timestamp_namibia"].iloc[0],
            "end_namibia": group["timestamp_namibia"].iloc[-1],
            "record_count": len(group),
            "duration_hours": round(
                (
                    group["timestamp_namibia"].iloc[-1]
                    - group["timestamp_namibia"].iloc[0]
                ).total_seconds() / 3600,
                2
            ),
            "tds": group["tds"].iloc[0],
            "humidity": group["humidity"].iloc[0],
            "Temperature": group["Temperature"].iloc[0],
            "median_interval_seconds": round(
                intervals.median(), 2
            ) if len(intervals) else None
        })

    plateau_report = pd.DataFrame(plateau_segments)

    if not plateau_report.empty:
        plateau_report = plateau_report.sort_values(
            "start_namibia"
        )

    data["flag_constant_plateau"] = (
        data["source_row_number"].isin(plateau_source_rows)
    )

    plateau_records = data[
        data["flag_constant_plateau"]
    ].copy()

    plateau_report.to_csv(
        OUTPUT_DIR / "02_constant_plateau_segments.csv",
        index=False
    )

    print("Constant plateau segments:", len(plateau_report))
    print("Records in plateau segments:", len(plateau_records))


    # -----------------------------------------------------
    # 7. ZERO TDS AND OTHER SENSOR FLAGS
    # -----------------------------------------------------

    data["flag_tds_zero"] = data["tds"].eq(0)

    data["flag_tds_negative"] = data["tds"].lt(0)

    # Assumes humidity is expressed as relative humidity (%).
    data["flag_humidity_outside_0_100"] = (
        (data["humidity"] < 0)
        | (data["humidity"] > 100)
    )

    zero_tds = data[
        data["flag_tds_zero"]
    ].copy()

    zero_tds.to_csv(
        OUTPUT_DIR / "03_zero_tds_records.csv",
        index=False
    )

    print("Zero TDS records:", len(zero_tds))

    # -----------------------------------------------------
    # 8. TIME GAP ANALYSIS
    # -----------------------------------------------------

    valid = valid.sort_values(
        "timestamp_namibia",
        kind="stable"
    ).copy()

    valid["interval_minutes"] = (
        valid["timestamp_namibia"]
        .diff()
        .dt.total_seconds() / 60
    )

    valid["previous_20_interval_median"] = (
        valid["interval_minutes"]
        .shift(1)
        .rolling(
            ROLLING_INTERVAL_COUNT,
            min_periods=3
        )
        .median()
    )

    global_median = valid["interval_minutes"].median()

    valid["expected_interval_minutes"] = (
        valid["previous_20_interval_median"]
        .fillna(global_median)
    )

    valid["gap_threshold_minutes"] = (
        valid["expected_interval_minutes"]
        .mul(GAP_MULTIPLIER)
        .clip(lower=MIN_GAP_THRESHOLD_MINUTES)
    )

    gaps = valid[
        valid["interval_minutes"]
        > valid["gap_threshold_minutes"]
    ].copy()

    gaps["gap_start_namibia"] = (
        valid["timestamp_namibia"].shift(1).loc[gaps.index]
    )

    gaps["gap_end_namibia"] = (
        gaps["timestamp_namibia"]
    )

    gaps["gap_duration_hours"] = (
        gaps["interval_minutes"] / 60
    )

    gaps.to_csv(
        OUTPUT_DIR / "04_time_gap_analysis.csv",
        index=False
    )

    print("Significant time gaps:", len(gaps))
    print(
        "Median sampling interval (minutes):",
        round(global_median, 3)
    )

    # -----------------------------------------------------
    # 9. CLASSIFY ALL OBSERVATIONS
    # -----------------------------------------------------

    data["data_classification"] = "VALID_CANDIDATE"

    data.loc[
        data["flag_tds_zero"],
        "data_classification"
    ] = "SUSPECT_ZERO_TDS"

    data.loc[
        data["flag_constant_plateau"],
        "data_classification"
    ] = "SUSPECT_CONSTANT_PLATEAU"

    data.loc[
        data["flag_confirmed_testing"],
        "data_classification"
    ] = "CONFIRMED_TEST"

    data.loc[
        data["flag_invalid_timestamp"],
        "data_classification"
    ] = "INVALID_TIMESTAMP"

    # Preserve all observations in the classified output.
    classified = data.sort_values(
        "timestamp_namibia",
        na_position="last",
        kind="stable"
    ).copy()

    classified.to_csv(
        OUTPUT_DIR / "05_classified_observations.csv",
        index=False
    )

    # -----------------------------------------------------
    # 10. ANOMALY SUMMARY
    # -----------------------------------------------------

    summary = {
        "total_original_observations": len(data),
        "valid_timestamp_records": int(
            data["timestamp_namibia"].notna().sum()
        ),
        "invalid_timestamp_records": int(
            data["flag_invalid_timestamp"].sum()
        ),
        "confirmed_testing_records": int(
            data["flag_confirmed_testing"].sum()
        ),
        "constant_plateau_segments": len(plateau_report),
        "constant_plateau_records": int(
            data["flag_constant_plateau"].sum()
        ),
        "zero_tds_records": int(
            data["flag_tds_zero"].sum()
        ),
        "negative_tds_records": int(
            data["flag_tds_negative"].sum()
        ),
        "humidity_outside_0_100_records": int(
            data["flag_humidity_outside_0_100"].sum()
        ),
        "significant_time_gaps": len(gaps),
        "median_sampling_interval_minutes": round(
            global_median, 3
        ),
        "timezone_source": SOURCE_TIMEZONE,
        "timezone_target": TARGET_TIMEZONE
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    )

    summary_df.to_csv(
        OUTPUT_DIR / "06_anomaly_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # 11. TEXT REPORT
    # -----------------------------------------------------

    report_path = OUTPUT_DIR / "07_step18_investigation_report.txt"

    with open(report_path, "w", encoding="utf-8") as report:

        report.write("BOREMAN STEP 18 INVESTIGATION REPORT\n")
        report.write("=" * 55 + "\n\n")

        for key, value in summary.items():
            report.write(f"{key}: {value}\n")

        report.write("\nDATA CLASSIFICATION\n")
        report.write("-" * 55 + "\n")

        counts = (
            classified["data_classification"]
            .value_counts()
        )

        for category, count in counts.items():
            report.write(f"{category}: {count}\n")

        report.write("\nIMPORTANT NOTES\n")
        report.write("-" * 55 + "\n")

        report.write(
            "All original observations are preserved.\n"
        )

        report.write(
            "Confirmed testing observations are classified separately.\n"
        )

        report.write(
            "Constant-value plateaus are flagged for investigation, "
            "not automatically rejected.\n"
        )

        report.write(
            "Zero TDS readings are retained.\n"
        )

        report.write(
            "No missing-value imputation or resampling was performed.\n"
        )

        report.write(
            "All analysis timestamps use Africa/Windhoek timezone.\n"
        )

    # -----------------------------------------------------
    # 12. COMPLETION
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("BOREMAN STEP 18 COMPLETED")
    print("=" * 55)

    print("\nOutput directory:")
    print(OUTPUT_DIR.resolve())

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.glob("*")):
        if file.is_file():
            print(f"{file.name} | {file.stat().st_size:,} bytes")

    print("\nPlease review the investigation reports before Step 19.")


# ---------------------------------------------------------
# 13. EXECUTE SCRIPT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
