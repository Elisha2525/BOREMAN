
"""
BOREMAN - Step 19: Data Quality Control and Dataset Preparation

Input:
    outputs/step18_timezone_anomaly_investigation/
    05_classified_observations.csv

Principles:
    - Preserve every original observation.
    - Exclude confirmed testing records from environmental candidates.
    - Exclude invalid timestamps and invalid humidity from candidates.
    - Retain zero TDS and constant plateaus with warning flags.
    - Do not impute, resample, or interpolate data.
    - Maintain a record-level audit trail.
    - Use Namibia local timestamps.
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

INPUT_FILE = (
    Path("outputs")
    / "step18_timezone_anomaly_investigation"
    / "05_classified_observations.csv"
)

OUTPUT_DIR = (
    Path("outputs")
    / "step19_data_quality_control"
)

SENSOR_COLUMNS = ["tds", "humidity", "Temperature"]
TARGET_TIMEZONE = "Africa/Windhoek"

REQUIRED_COLUMNS = [
    "source_row_number",
    "timestamp_namibia",
    "flag_invalid_timestamp",
    "flag_confirmed_testing",
    "flag_constant_plateau",
    "flag_tds_zero",
    "flag_tds_negative",
    "flag_humidity_outside_0_100",
]


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def read_flag(series):
    """Safely convert CSV boolean-like values into True/False."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def add_reason(existing, condition, reason):
    """Add a quality-control reason without losing earlier reasons."""
    return existing.where(
        ~condition,
        existing.where(existing.eq(""), existing + "; ")
        + reason
    )


# ---------------------------------------------------------
# 3. MAIN FUNCTION
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Step 18 input file not found: {INPUT_FILE.resolve()}"
        )

    print("\nBOREMAN STEP 19 STARTED")
    print("-" * 60)

    # -----------------------------------------------------
    # 4. LOAD STEP 18 CLASSIFIED OBSERVATIONS
    # -----------------------------------------------------

    data = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing from Step 18 output: "
            f"{missing_columns}"
        )

    for column in SENSOR_COLUMNS:
        if column not in data.columns:
            raise ValueError(
                f"Missing sensor column: {column}"
            )

    original_count = len(data)

    print(f"Loaded observations: {original_count:,}")

    # -----------------------------------------------------
    # 5. NORMALIZE QUALITY FLAGS
    # -----------------------------------------------------

    flag_columns = [
        "flag_invalid_timestamp",
        "flag_confirmed_testing",
        "flag_constant_plateau",
        "flag_tds_zero",
        "flag_tds_negative",
        "flag_humidity_outside_0_100",
    ]

    for column in flag_columns:
        data[column] = read_flag(data[column])

    # Independently check timestamp availability.
    data["timestamp_namibia"] = (
    pd.to_datetime(
        data["timestamp_namibia"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(TARGET_TIMEZONE)
)

    data["flag_invalid_timestamp"] = (
        data["flag_invalid_timestamp"]
        | data["timestamp_namibia"].isna()
    )

    # Check whether any of the three sensor values are missing.
    data["flag_missing_sensor_value"] = (
        data[SENSOR_COLUMNS].isna().any(axis=1)
    )

    # Preserve original record count and unique source references.
    if data["source_row_number"].duplicated().any():
        raise ValueError(
            "Duplicate source_row_number values detected. "
            "Please investigate before proceeding."
        )

    # -----------------------------------------------------
    # 6. DETERMINE ENVIRONMENTAL CANDIDATE ELIGIBILITY
    # -----------------------------------------------------

    data["eligible_for_environmental_analysis"] = (
        ~data["flag_confirmed_testing"]
        & ~data["flag_invalid_timestamp"]
        & ~data["flag_humidity_outside_0_100"]
        & ~data["flag_missing_sensor_value"]
    )

    # Zero TDS and constant plateaus are deliberately NOT
    # excluded from this eligibility rule.

    # -----------------------------------------------------
    # 7. CREATE QUALITY-CONTROL REASONS
    # -----------------------------------------------------

    data["quality_control_reasons"] = ""

    data["quality_control_reasons"] = add_reason(
        data["quality_control_reasons"],
        data["flag_confirmed_testing"],
        "CONFIRMED_SENSOR_TESTING"
    )

    data["quality_control_reasons"] = add_reason(
        data["quality_control_reasons"],
        data["flag_invalid_timestamp"],
        "INVALID_TIMESTAMP"
    )

    data["quality_control_reasons"] = add_reason(
        data["quality_control_reasons"],
        data["flag_humidity_outside_0_100"],
        "HUMIDITY_OUTSIDE_0_100"
    )

    data["quality_control_reasons"] = add_reason(
        data["quality_control_reasons"],
        data["flag_missing_sensor_value"],
        "MISSING_SENSOR_VALUE"
    )

    data["quality_control_reasons"] = add_reason(
        data["quality_control_reasons"],
        data["flag_constant_plateau"],
        "SUSPECT_CONSTANT_PLATEAU_RETAINED"
    )

    data["quality_control_reasons"] = add_reason(
        data["quality_control_reasons"],
        data["flag_tds_zero"],
        "ZERO_TDS_RETAINED"
    )

    data["quality_control_reasons"] = add_reason(
        data["quality_control_reasons"],
        data["flag_tds_negative"],
        "NEGATIVE_TDS"
    )

    data.loc[
        data["quality_control_reasons"].eq(""),
        "quality_control_reasons"
    ] = "NO_CURRENT_QUALITY_FLAGS"

    # -----------------------------------------------------
    # 8. ASSIGN FINAL QUALITY-CONTROL STATUS
    # -----------------------------------------------------

    data["quality_control_status"] = "CANDIDATE_NO_FLAGS"

    data.loc[
        data["flag_constant_plateau"]
        | data["flag_tds_zero"]
        | data["flag_tds_negative"],
        "quality_control_status"
    ] = "CANDIDATE_WITH_WARNINGS"

    data.loc[
        data["flag_missing_sensor_value"],
        "quality_control_status"
    ] = "EXCLUDED_MISSING_SENSOR_VALUE"

    data.loc[
        data["flag_humidity_outside_0_100"],
        "quality_control_status"
    ] = "EXCLUDED_INVALID_HUMIDITY"

    data.loc[
        data["flag_invalid_timestamp"],
        "quality_control_status"
    ] = "EXCLUDED_INVALID_TIMESTAMP"

    data.loc[
        data["flag_confirmed_testing"],
        "quality_control_status"
    ] = "EXCLUDED_CONFIRMED_TEST"

    # -----------------------------------------------------
    # 9. PREPARE SEPARATE DATASETS
    # -----------------------------------------------------

    # Dataset A: All observations with quality flags.
    all_observations = data.copy()

    # Dataset B: Environmental candidates.
    # Zero TDS and constant plateaus remain included.
    environmental_candidates = data[
        data["eligible_for_environmental_analysis"]
    ].copy()

    # Dataset C: Confirmed testing records.
    confirmed_testing = data[
        data["flag_confirmed_testing"]
    ].copy()

    # Dataset D: Suspect observations.
    suspect_condition = (
        data["flag_constant_plateau"]
        | data["flag_tds_zero"]
        | data["flag_tds_negative"]
        | data["flag_humidity_outside_0_100"]
        | data["flag_missing_sensor_value"]
    )

    suspect_observations = data[
        suspect_condition
        & ~data["flag_confirmed_testing"]
        & ~data["flag_invalid_timestamp"]
    ].copy()

    # Dataset E: Invalid timestamp records.
    invalid_timestamp_records = data[
        data["flag_invalid_timestamp"]
    ].copy()

    # Dataset F: Record-level audit trail.
    audit_columns = [
        "source_row_number",
        "timestamp_original",
        "timestamp_namibia",
        "tds",
        "humidity",
        "Temperature",
        "flag_confirmed_testing",
        "flag_invalid_timestamp",
        "flag_constant_plateau",
        "flag_tds_zero",
        "flag_tds_negative",
        "flag_humidity_outside_0_100",
        "flag_missing_sensor_value",
        "eligible_for_environmental_analysis",
        "quality_control_status",
        "quality_control_reasons",
    ]

    audit_columns = [
        column for column in audit_columns
        if column in data.columns
    ]

    audit_trail = data[audit_columns].copy()

    # -----------------------------------------------------
    # 10. SAVE OUTPUT FILES
    # -----------------------------------------------------

    outputs = {
        "01_all_observations_with_quality_flags.csv":
            all_observations,

        "02_environmental_candidate_dataset.csv":
            environmental_candidates,

        "03_confirmed_testing_excluded.csv":
            confirmed_testing,

        "04_suspect_observations.csv":
            suspect_observations,

        "05_invalid_timestamp_records.csv":
            invalid_timestamp_records,

        "07_record_level_audit_trail.csv":
            audit_trail,
    }

    for filename, dataframe in outputs.items():
        dataframe.to_csv(
            OUTPUT_DIR / filename,
            index=False
        )

    # -----------------------------------------------------
    # 11. DATA QUALITY SUMMARY
    # -----------------------------------------------------

    summary = [
        ("total_original_observations", original_count),
        ("all_observations_output", len(all_observations)),
        ("environmental_candidate_records",
         len(environmental_candidates)),
        ("confirmed_testing_excluded", len(confirmed_testing)),
        ("suspect_observations", len(suspect_observations)),
        ("invalid_timestamp_records",
         len(invalid_timestamp_records)),
        ("constant_plateau_records",
         int(data["flag_constant_plateau"].sum())),
        ("zero_tds_records",
         int(data["flag_tds_zero"].sum())),
        ("negative_tds_records",
         int(data["flag_tds_negative"].sum())),
        ("invalid_humidity_records",
         int(data["flag_humidity_outside_0_100"].sum())),
        ("missing_sensor_value_records",
         int(data["flag_missing_sensor_value"].sum())),
        ("eligible_records_with_warnings",
         int(
             (
                 environmental_candidates[
                     "quality_control_status"
                 ] == "CANDIDATE_WITH_WARNINGS"
             ).sum()
         )),
        ("records_with_no_current_quality_flags",
         int(
             (
                 data["quality_control_status"]
                 == "CANDIDATE_NO_FLAGS"
             ).sum()
         )),
    ]

    summary_df = pd.DataFrame(
        summary,
        columns=["metric", "value"]
    )

    summary_df.to_csv(
        OUTPUT_DIR / "06_data_quality_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # 12. VALIDATION CHECKS
    # -----------------------------------------------------

    assert len(all_observations) == original_count

    assert (
        len(confirmed_testing)
        == int(data["flag_confirmed_testing"].sum())
    )

    assert (
        len(invalid_timestamp_records)
        == int(data["flag_invalid_timestamp"].sum())
    )

    assert (
        len(audit_trail) == original_count
    )

    assert (
        environmental_candidates[
            "eligible_for_environmental_analysis"
        ].all()
    )

    assert (
        not environmental_candidates[
            "flag_confirmed_testing"
        ].any()
    )

    assert (
        not environmental_candidates[
            "flag_invalid_timestamp"
        ].any()
    )

    assert (
        not environmental_candidates[
            "flag_humidity_outside_0_100"
        ].any()
    )

    # Confirm that zero TDS and plateau records have not
    # been automatically excluded.
    print("\nQUALITY CONTROL VALIDATION")
    print("-" * 60)

    print(
        "Zero TDS retained in candidates:",
        int(
            environmental_candidates["flag_tds_zero"].sum()
        )
    )

    print(
        "Constant plateau records retained in candidates:",
        int(
            environmental_candidates[
                "flag_constant_plateau"
            ].sum()
        )
    )

    print("All record-preservation checks passed.")

    # -----------------------------------------------------
    # 13. COMPLETION REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("BOREMAN STEP 19 COMPLETED")
    print("=" * 60)

    print(f"Original observations: {original_count:,}")
    print(
        "Environmental candidates:",
        f"{len(environmental_candidates):,}"
    )
    print(
        "Confirmed testing excluded:",
        f"{len(confirmed_testing):,}"
    )
    print(
        "Suspect observations:",
        f"{len(suspect_observations):,}"
    )
    print(
        "Invalid timestamps:",
        f"{len(invalid_timestamp_records):,}"
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR.resolve())

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.glob("*.csv")):
        print(
            f"{file.name} | "
            f"{file.stat().st_size:,} bytes"
        )

    print("\nPlease review Step 19 outputs before proceeding.")


# ---------------------------------------------------------
# 14. EXECUTE SCRIPT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
