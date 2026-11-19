"""
BOREMAN STEP 23 - MODELLING READINESS ASSESSMENT

Input:
  outputs/step22_environmental_pattern_analysis/08_records_with_environmental_groups.csv

Purpose:
  - Separate primary, secondary and plateau-review observations.
  - Quantify quality flags and coverage by Namibia local date.
  - Propose a chronological (not random) train/validation/test split for
    PRIMARY modelling candidates only.
  - Identify limitations that must be resolved before model training.

This step does NOT train a model, delete observations, impute values, or
declare sensor readings physically correct.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

INPUT_FILE = Path(
    "outputs/step22_environmental_pattern_analysis/"
    "08_records_with_environmental_groups.csv"
)
OUTPUT_DIR = Path("outputs/step23_modelling_readiness")

TIME_COL = "timestamp_namibia"
SENSORS = ["tds", "humidity", "Temperature"]

FLAG_COLUMNS = [
    "flag_constant_plateau",
    "flag_tds_zero",
    "flag_tds_negative",
    "flag_humidity_outside_0_100",
    "flag_any_iqr_review",
    "flag_any_abrupt_change_review",
    "flag_duplicate_timestamp_review",
]

def stop(message):
    raise ValueError(message)


def as_bool(series):
    """Safely parse CSV booleans without treating the string 'False' as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(False)
        .astype(bool)
    )


def load_data():
    if not INPUT_FILE.exists():
        stop(
            f"Input file not found: {INPUT_FILE}\n"
            "Run this script from the project root after completing Step 22."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "source_row_number", TIME_COL, *SENSORS,
        "analysis_group", "namibia_date",
        "eligible_for_environmental_analysis",
        *FLAG_COLUMNS,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        stop(f"Required columns missing: {missing}")

    if df["source_row_number"].isna().any():
        stop("Missing source_row_number values.")
    if df["source_row_number"].duplicated().any():
        stop("source_row_number is not unique.")

    for col in SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[SENSORS].isna().any().any():
        stop("Missing sensor values found. Stop and investigate before readiness assessment.")

    parsed = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    if parsed.isna().any():
        stop("Invalid Namibia timestamps found.")
    df["_time"] = parsed.dt.tz_convert("Africa/Windhoek")
    df["_date"] = df["_time"].dt.date.astype(str)

    for col in FLAG_COLUMNS:
        df[col] = as_bool(df[col])

    df["eligible_for_environmental_analysis"] = as_bool(
        df["eligible_for_environmental_analysis"]
    )
    if not df["eligible_for_environmental_analysis"].all():
        stop("At least one row is not eligible_for_environmental_analysis.")

    # Confirm that Step 22 grouping matches the source timestamp.
    if not (df["namibia_date"].astype(str) == df["_date"]).all():
        stop("namibia_date does not match timestamp_namibia after timezone parsing.")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    input_count = len(df)
    source_ids = set(df["source_row_number"].tolist())

    # Plateau status is an independent review category.
    plateau = df["flag_constant_plateau"]
    review_flags = [
        "flag_tds_zero",
        "flag_tds_negative",
        "flag_humidity_outside_0_100",
        "flag_any_iqr_review",
        "flag_any_abrupt_change_review",
        "flag_duplicate_timestamp_review",
    ]
    df["flag_any_non_plateau_review"] = df[review_flags].any(axis=1)

    df["modelling_readiness_group"] = np.select(
        [
            plateau,
            (~plateau) & (~df["flag_any_non_plateau_review"]),
            (~plateau) & df["flag_any_non_plateau_review"],
        ],
        [
            "PLATEAU_REVIEW_EXCLUDE_FROM_PRIMARY",
            "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS",
            "SECONDARY_CANDIDATE_REVIEW_FLAGS_PRESENT",
        ],
        default="UNCLASSIFIED_REVIEW_REQUIRED",
    )

    # Keep all records; this is an assessment label, not a deletion filter.
    df = df.sort_values(["_time", "source_row_number"]).reset_index(drop=True)

    # Chronological split recommendation is calculated on primary candidates
    # by distinct local dates. Same-date records cannot cross split boundaries.
    df["proposed_temporal_split"] = "NOT_ASSIGNED"
    primary_mask = (
        df["modelling_readiness_group"]
        == "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS"
    )
    primary_dates = sorted(df.loc[primary_mask, "_date"].unique())

    split_info = []
    if len(primary_dates) >= 3:
        n_dates = len(primary_dates)
        train_end = max(1, int(np.floor(n_dates * 0.70)))
        val_end = max(train_end + 1, int(np.floor(n_dates * 0.85)))
        val_end = min(val_end, n_dates - 1)

        train_dates = primary_dates[:train_end]
        val_dates = primary_dates[train_end:val_end]
        test_dates = primary_dates[val_end:]

        date_to_split = {}
        date_to_split.update({d: "TRAIN" for d in train_dates})
        date_to_split.update({d: "VALIDATION" for d in val_dates})
        date_to_split.update({d: "TEST" for d in test_dates})

        df.loc[primary_mask, "proposed_temporal_split"] = (
            df.loc[primary_mask, "_date"].map(date_to_split).values
        )

        for split_name, dates in [
            ("TRAIN", train_dates),
            ("VALIDATION", val_dates),
            ("TEST", test_dates),
        ]:
            split_rows = df[
                primary_mask & df["_date"].isin(dates)
            ]
            split_info.append({
                "split": split_name,
                "date_count": len(dates),
                "start_date": dates[0] if dates else "",
                "end_date": dates[-1] if dates else "",
                "record_count": len(split_rows),
            })
    else:
        split_info.append({
            "split": "NOT_PROPOSED",
            "date_count": len(primary_dates),
            "start_date": primary_dates[0] if primary_dates else "",
            "end_date": primary_dates[-1] if primary_dates else "",
            "record_count": int(primary_mask.sum()),
        })

    # Record-level output retains all original fields and adds readiness labels.
    record_output = df.drop(columns=["_time", "_date"])
    record_output.to_csv(
        OUTPUT_DIR / "01_all_records_with_readiness_labels.csv", index=False
    )

    # Group-level sensor profile, explicitly descriptive.
    sensor_summary = (
        df.groupby("modelling_readiness_group")[SENSORS]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )
    sensor_summary.columns = [
        f"{sensor}_{stat}" for sensor, stat in sensor_summary.columns
    ]
    sensor_summary.reset_index().to_csv(
        OUTPUT_DIR / "02_readiness_group_sensor_summary.csv", index=False
    )

    # Counts and flag prevalence by readiness group.
    count_rows = []
    for group_name, part in df.groupby("modelling_readiness_group", sort=True):
        row = {
            "modelling_readiness_group": group_name,
            "record_count": len(part),
            "distinct_namibia_dates": part["_date"].nunique(),
            "start_date": part["_date"].min(),
            "end_date": part["_date"].max(),
        }
        for flag in FLAG_COLUMNS:
            row[f"{flag}_count"] = int(part[flag].sum())
        count_rows.append(row)
    pd.DataFrame(count_rows).to_csv(
        OUTPUT_DIR / "03_readiness_and_flag_counts.csv", index=False
    )

    # Daily coverage and readiness-group counts.
    daily_rows = []
    for date, part in df.groupby("_date", sort=True):
        row = {
            "namibia_date": date,
            "total_records": len(part),
            "primary_candidate_records": int(
                (part["modelling_readiness_group"]
                 == "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS").sum()
            ),
            "secondary_candidate_records": int(
                (part["modelling_readiness_group"]
                 == "SECONDARY_CANDIDATE_REVIEW_FLAGS_PRESENT").sum()
            ),
            "plateau_review_records": int(
                (part["modelling_readiness_group"]
                 == "PLATEAU_REVIEW_EXCLUDE_FROM_PRIMARY").sum()
            ),
        }
        daily_rows.append(row)
    pd.DataFrame(daily_rows).to_csv(
        OUTPUT_DIR / "04_daily_modelling_coverage.csv", index=False
    )

    # Chronological split dates and counts.
    pd.DataFrame(split_info).to_csv(
        OUTPUT_DIR / "05_proposed_chronological_split.csv", index=False
    )

    # Explicit readiness checklist; items requiring external evidence are not
    # inferred from sensor-only records.
    checks = [
        {
            "readiness_item": "Source record traceability",
            "status": "PASS",
            "evidence_or_action": "Unique source_row_number retained for every input record."
        },
        {
            "readiness_item": "Namibia local timestamps",
            "status": "PASS",
            "evidence_or_action": "Timestamps parsed in Africa/Windhoek; date grouping cross-checked."
        },
        {
            "readiness_item": "Plateau observations separated",
            "status": "PASS",
            "evidence_or_action": "Plateau-review observations labelled separately; not deleted."
        },
        {
            "readiness_item": "Review flags retained",
            "status": "PASS",
            "evidence_or_action": "Zero, negative, humidity-range, IQR, abrupt-change and duplicate flags retained."
        },
        {
            "readiness_item": "Chronological split proposal",
            "status": "PROPOSED" if len(primary_dates) >= 3 else "NOT_AVAILABLE",
            "evidence_or_action": "Date-block split only; requires review before model use."
        },
        {
            "readiness_item": "Ground-truth target labels",
            "status": "NOT_PROVIDED",
            "evidence_or_action": "Sensor-only data does not establish a supervised target or verified water-quality class."
        },
        {
            "readiness_item": "Independent laboratory/reference measurements",
            "status": "NOT_PROVIDED",
            "evidence_or_action": "Obtain calibrated reference or laboratory measurements before claiming predictive accuracy."
        },
        {
            "readiness_item": "Sensor calibration and field maintenance log",
            "status": "NOT_PROVIDED",
            "evidence_or_action": "Confirm sensor model, calibration dates, maintenance, installation and field-testing periods."
        },
        {
            "readiness_item": "External validation period/site",
            "status": "NOT_PROVIDED",
            "evidence_or_action": "A separate time period or location is needed to test generalisation."
        },
    ]
    pd.DataFrame(checks).to_csv(
        OUTPUT_DIR / "06_model_readiness_checklist.csv", index=False
    )

    # Preservation and classification assertions.
    if len(df) != input_count:
        stop("Record count changed.")
    if set(df["source_row_number"].tolist()) != source_ids:
        stop("Source record identities changed.")
    if df["source_row_number"].duplicated().any():
        stop("Duplicate source_row_number found.")
    if (df["modelling_readiness_group"] == "UNCLASSIFIED_REVIEW_REQUIRED").any():
        stop("Some records were not assigned a readiness group.")
    if df["proposed_temporal_split"].isin(["TRAIN", "VALIDATION", "TEST"]).any():
        split_dates = df[df["proposed_temporal_split"] != "NOT_ASSIGNED"].groupby(
            "_date"
        )["proposed_temporal_split"].nunique()
        if (split_dates > 1).any():
            stop("Date leakage detected: a Namibia date appears in multiple splits.")

    group_counts = df["modelling_readiness_group"].value_counts().to_dict()
    report = [
        "BOREMAN STEP 23 - MODELLING READINESS ASSESSMENT",
        "=" * 60,
        f"Input file: {INPUT_FILE}",
        f"Input records: {input_count:,}",
        f"Records preserved: {len(df):,}",
        f"Unique source records: {df['source_row_number'].nunique():,}",
        f"Namibia date range: {df['_date'].min()} to {df['_date'].max()}",
        f"Distinct Namibia dates: {df['_date'].nunique()}",
        "",
        "READINESS GROUP COUNTS",
    ]
    for name, count in sorted(group_counts.items()):
        report.append(f"- {name}: {count:,}")
    report.extend([
        "",
        "PROPOSED CHRONOLOGICAL SPLIT",
        f"Primary candidate dates available: {len(primary_dates)}",
        "Split is based on chronological date blocks, not random row splitting.",
        "Records from the same Namibia date remain in the same split.",
        "This is a proposal only and must be reviewed before training.",
        "",
        "MODELLING LIMITATIONS",
        "- This dataset contains sensor observations, not verified target labels.",
        "- No supervised water-quality classification/regression target is established here.",
        "- Independent laboratory/reference measurements are not present in this input.",
        "- Sensor calibration and maintenance evidence must be collected.",
        "- Plateau-review records are not part of primary modelling candidates.",
        "- Secondary candidates remain available for sensitivity analysis.",
        "- No records were deleted, imputed or corrected.",
        "- A temporal split does not replace external/site-level validation.",
        "",
        "PRESERVATION CHECKS",
        "PASS: all input observations preserved.",
        "PASS: source_row_number remains unique.",
        "PASS: all records assigned to a readiness group.",
        "PASS: no Namibia date crosses proposed split boundaries.",
        "",
        "OUTPUT FILES",
        "- 01_all_records_with_readiness_labels.csv",
        "- 02_readiness_group_sensor_summary.csv",
        "- 03_readiness_and_flag_counts.csv",
        "- 04_daily_modelling_coverage.csv",
        "- 05_proposed_chronological_split.csv",
        "- 06_model_readiness_checklist.csv",
        "- 07_step23_modelling_readiness_report.txt",
    ])
    (OUTPUT_DIR / "07_step23_modelling_readiness_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nSTEP 23 COMPLETED")
    print(f"Records preserved: {len(df):,}")
    for name, count in sorted(group_counts.items()):
        print(f"{name}: {count:,}")
    print(f"Primary candidate dates: {len(primary_dates)}")
    print("All preservation and date-split checks passed.")
    print(f"Outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSTEP 23 STOPPED: {exc}")
        sys.exit(1)
