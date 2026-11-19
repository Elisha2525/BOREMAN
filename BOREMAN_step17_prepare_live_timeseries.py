"""
BOREMAN - Step 17: Prepare live sensor time-series data
Input:  conductivity.xlsx (exported from Google Sheets)
Output: outputs/live_timeseries/

This script preserves the original values. It does NOT impute missing readings,
remove suspicious values, or resample the series.
"""

from pathlib import Path
import pandas as pd

# Run this script from the BOREMAN project root.
INPUT_FILE = Path("conductivity.xlsx")
OUTPUT_DIR = Path("outputs") / "live_timeseries"

REQUIRED_COLUMNS = ["Timestamp", "tds", "humidity", "Temperature"]
SENSOR_COLUMNS = ["tds", "humidity", "Temperature"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.resolve()}\n"
            "Place conductivity.xlsx in the BOREMAN project root."
        )

    original = pd.read_excel(INPUT_FILE)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in original.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Preserve the imported sheet exactly as read.
    original.to_csv(OUTPUT_DIR / "01_original_export.csv", index=False)

    data = original[REQUIRED_COLUMNS].copy()
    data["source_row_number"] = range(2, len(data) + 2)

    # format='mixed' handles the mixed timestamp formats in the export.
    data["timestamp_parsed"] = pd.to_datetime(
        data["Timestamp"], format="mixed", errors="coerce"
    )

    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Flags are diagnostic only. No flagged observation is silently deleted.
    data["flag_invalid_timestamp"] = data["timestamp_parsed"].isna()
    data["flag_duplicate_timestamp"] = (
        data["timestamp_parsed"].notna()
        & data.duplicated("timestamp_parsed", keep=False)
    )
    data["flag_tds_zero"] = data["tds"].eq(0)
    data["flag_tds_45_temp_70_block"] = (
        data["tds"].eq(45) & data["Temperature"].eq(70)
    )
    data["flag_humidity_202"] = data["humidity"].eq(202)
    data["flag_missing_sensor_value"] = data[SENSOR_COLUMNS].isna().any(axis=1)

    # Invalid timestamp rows are retained separately for review.
    invalid_timestamp = data[data["flag_invalid_timestamp"]].copy()
    invalid_timestamp.to_csv(
        OUTPUT_DIR / "02_invalid_timestamp_rows.csv", index=False
    )

    # Sort valid timestamp rows chronologically. Keep duplicate rows here.
    valid = data[data["timestamp_parsed"].notna()].copy()
    valid = valid.sort_values("timestamp_parsed", kind="stable")
    valid.to_csv(OUTPUT_DIR / "03_timestamp_sorted_all_rows.csv", index=False)

    # Exact duplicate timestamp + identical sensor values: keep one copy for
    # a modelling candidate, but retain all rows in the previous file.
    dedup = valid.drop_duplicates(
        subset=["timestamp_parsed"] + SENSOR_COLUMNS, keep="first"
    ).copy()
    dedup.to_csv(
        OUTPUT_DIR / "04_exact_duplicate_removed_candidate.csv", index=False
    )

    # All diagnostic flags in one file.
    flagged = data[
        data[
            [
                "flag_invalid_timestamp",
                "flag_duplicate_timestamp",
                "flag_tds_zero",
                "flag_tds_45_temp_70_block",
                "flag_humidity_202",
                "flag_missing_sensor_value",
            ]
        ].any(axis=1)
    ].copy()
    flagged.to_csv(OUTPUT_DIR / "05_flagged_observations.csv", index=False)
if __name__ == "__main__":
    main()
