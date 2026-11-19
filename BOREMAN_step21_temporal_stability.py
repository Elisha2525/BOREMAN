"""
BOREMAN STEP 21 - Temporal Behaviour and Sensor Stability
---------------------------------------------------------
Input:
  outputs/step20_environmental_validation/02_all_candidates_with_statistical_flags.csv

Purpose:
  - Review duplicate timestamps.
  - Detect abrupt sensor-to-sensor changes using robust first-difference thresholds.
  - Identify repeated multi-sensor plateaus, without treating them as confirmed faults.
  - Summarize daily/hourly sensor stability and coverage.
  - Preserve every input observation.

No records are deleted or imputed. Statistical events are review candidates only.
All time-based analysis uses Africa/Windhoek (Namibia local time).
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILE = Path("outputs/step20_environmental_validation/02_all_candidates_with_statistical_flags.csv")
OUTPUT_DIR = Path("outputs/step21_temporal_stability")
TIME_COL = "timestamp_namibia"
SENSORS = ["tds", "humidity", "Temperature"]
MIN_PLATEAU_RECORDS = 30


def robust_threshold(values, multiplier=6.0):
    """Return median +/- multiplier * 1.4826*MAD for non-null values."""
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan, np.nan, np.nan, np.nan
    med = x.median()
    mad = (x - med).abs().median()
    sigma = 1.4826 * mad
    if sigma == 0 or not np.isfinite(sigma):
        return med, mad, np.nan, np.nan
    return med, mad, med - multiplier * sigma, med + multiplier * sigma


def main():
    print("\nBOREMAN STEP 21 STARTED")
    print("-" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\n"
            "Run this script from the project root folder."
        )

    df = pd.read_csv(INPUT_FILE)
    input_count = len(df)
    required = [TIME_COL, *SENSORS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "source_row_number" not in df.columns:
        raise ValueError("source_row_number is required for traceability.")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True).dt.tz_convert("Africa/Windhoek")
    for c in SENSORS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([TIME_COL, "source_row_number"], kind="stable", na_position="last").reset_index(drop=True)
    df["step21_row_number"] = np.arange(1, len(df) + 1)

    # A. Duplicate timestamp groups. Keep all rows.
    valid_time = df[df[TIME_COL].notna()].copy()
    duplicate_mask = valid_time.duplicated(subset=[TIME_COL], keep=False)
    duplicate_rows = valid_time[duplicate_mask].copy()
    if not duplicate_rows.empty:
        duplicate_rows["duplicate_timestamp_group_size"] = (
            duplicate_rows.groupby(TIME_COL)[TIME_COL].transform("size")
        )
        # Determine whether all sensor values are identical within each
        # duplicate-timestamp group. Use group-level aggregation (not
        # DataFrameGroupBy.transform, whose callable receives individual
        # Series in some pandas versions).
        same_values_by_timestamp = (
            duplicate_rows.groupby(TIME_COL)[SENSORS]
            .nunique(dropna=False)
            .le(1)
            .all(axis=1)
        )
        duplicate_rows["same_sensor_values_within_timestamp_group"] = (
            duplicate_rows[TIME_COL].map(same_values_by_timestamp).astype(bool)
        )
    duplicate_rows.to_csv(OUTPUT_DIR / "01_duplicate_timestamp_review.csv", index=False)

    # B. Consecutive changes.
    # The previous implementation included the many zero differences from
    # long constant plateaus. That made the global MAD equal to zero and
    # produced NaN thresholds, so no abrupt changes could ever be flagged.
    #
    # Corrected approach:
    # - estimate thresholds from NON-ZERO changes only;
    # - use intervals > 0 and <= 5 minutes for the baseline;
    # - exclude duplicate timestamps and long gaps from event flagging;
    # - use median +/- 6 * 1.4826 * MAD;
    # - use an IQR / 99.5th-percentile fallback if MAD is zero.
    # Flags remain review candidates, not confirmed faults.
    timed = valid_time.sort_values([TIME_COL, "source_row_number"]).copy()
    timed["interval_minutes"] = timed[TIME_COL].diff().dt.total_seconds() / 60
    event_summary = []
    change_records = timed[["source_row_number", TIME_COL, "interval_minutes", *SENSORS]].copy()

    baseline_interval = (
        change_records["interval_minutes"].gt(0)
        & change_records["interval_minutes"].le(5)
    )

    for sensor in SENSORS:
        diff_col = f"{sensor}_change_from_previous"
        flag_col = f"flag_{sensor}_abrupt_change_review"
        change_records[diff_col] = timed[sensor].diff()

        baseline = change_records.loc[
            baseline_interval & change_records[diff_col].notna()
            & change_records[diff_col].ne(0), diff_col
        ].dropna()

        if baseline.empty:
            med = mad = low = high = np.nan
            method = "No non-zero baseline changes available"
            flag = pd.Series(False, index=change_records.index)
        else:
            med = float(baseline.median())
            mad = float((baseline - med).abs().median())
            sigma = 1.4826 * mad

            if sigma > 0 and np.isfinite(sigma):
                method = "Non-zero changes, intervals 0-5 min; median +/- 6 * 1.4826 * MAD"
            else:
                q1, q3 = baseline.quantile([0.25, 0.75])
                iqr = float(q3 - q1)
                sigma = iqr / 1.349 if iqr > 0 else float(
                    (baseline - med).abs().quantile(0.995)
                ) / 3.0
                method = "Non-zero changes; MAD fallback to IQR scale or 99.5th percentile deviation"

            if sigma > 0 and np.isfinite(sigma):
                low = med - 6.0 * sigma
                high = med + 6.0 * sigma
                eligible_for_flag = (
                    change_records[diff_col].notna()
                    & change_records["interval_minutes"].gt(0)
                    & change_records["interval_minutes"].le(5)
                )
                flag = eligible_for_flag & (
                    (change_records[diff_col] < low)
                    | (change_records[diff_col] > high)
                )
            else:
                low = high = np.nan
                flag = pd.Series(False, index=change_records.index)
                method += "; no usable scale, no flags generated"

        change_records[flag_col] = flag.fillna(False).astype(bool)
        event_summary.append({
            "sensor": sensor,
            "baseline_nonzero_change_count": int(len(baseline)),
            "baseline_difference_median": med,
            "baseline_difference_mad": mad,
            "review_lower_threshold": low,
            "review_upper_threshold": high,
            "abrupt_change_review_count": int(change_records[flag_col].sum()),
            "threshold_method": method,
            "flagging_interval_rule": "Only intervals > 0 and <= 5 minutes",
        })

    abrupt_cols = [f"flag_{s}_abrupt_change_review" for s in SENSORS]
    change_records["flag_any_abrupt_change_review"] = change_records[abrupt_cols].any(axis=1)
    change_records.to_csv(OUTPUT_DIR / "02_consecutive_sensor_changes.csv", index=False)
    change_records[change_records["flag_any_abrupt_change_review"]].to_csv(
        OUTPUT_DIR / "03_abrupt_change_review_records.csv", index=False
    )
    pd.DataFrame(event_summary).to_csv(OUTPUT_DIR / "04_abrupt_change_threshold_summary.csv", index=False)

    # C. Consecutive identical values across all three sensors.
    # Break runs at large timestamp gaps; estimate cadence from previous 20 intervals.
    plateau_df = timed[["source_row_number", TIME_COL, "interval_minutes", *SENSORS]].copy()
    plateau_df["local_median_interval_minutes"] = (
        plateau_df["interval_minutes"].shift(1).rolling(20, min_periods=3).median()
    )
    plateau_df["gap_threshold_minutes"] = np.maximum(
        3 * plateau_df["local_median_interval_minutes"].fillna(1.0), 5.0
    )

    same_as_previous = pd.Series(False, index=plateau_df.index)
    if len(plateau_df) > 1:
        same_values = (
            plateau_df[SENSORS].eq(plateau_df[SENSORS].shift(1)).all(axis=1)
            & plateau_df[SENSORS].notna().all(axis=1)
            & plateau_df[SENSORS].shift(1).notna().all(axis=1)
        )
        within_gap = plateau_df["interval_minutes"].le(plateau_df["gap_threshold_minutes"])
        same_as_previous = same_values & within_gap

    run_ids = []
    run_id = 0
    for idx, same in enumerate(same_as_previous.tolist()):
        if idx == 0 or not same:
            run_id += 1
        run_ids.append(run_id)
    plateau_df["run_id"] = run_ids

    plateau_segments = []
    for rid, group in plateau_df.groupby("run_id", sort=False):
        if len(group) >= MIN_PLATEAU_RECORDS:
            plateau_segments.append({
                "plateau_segment_id": int(rid),
                "record_count": int(len(group)),
                "start_namibia": group[TIME_COL].min(),
                "end_namibia": group[TIME_COL].max(),
                "duration_hours": (
                    group[TIME_COL].max() - group[TIME_COL].min()
                ).total_seconds() / 3600,
                "tds_value": group["tds"].iloc[0],
                "humidity_value": group["humidity"].iloc[0],
                "Temperature_value": group["Temperature"].iloc[0],
                "median_interval_minutes": group["interval_minutes"].median(),
                "source_row_start": group["source_row_number"].iloc[0],
                "source_row_end": group["source_row_number"].iloc[-1],
                "classification": "SUSPECT_CONSTANT_PLATEAU_REVIEW",
            })
    plateau_segments_df = pd.DataFrame(plateau_segments)
    plateau_segments_df.to_csv(OUTPUT_DIR / "05_constant_plateau_segments.csv", index=False)

    # D. Daily/hourly stability summaries.
    timed["namibia_date"] = timed[TIME_COL].dt.strftime("%Y-%m-%d")
    timed["namibia_hour"] = timed[TIME_COL].dt.strftime("%Y-%m-%d %H:00:00")

    agg = {}
    for sensor in SENSORS:
        agg[f"{sensor}_mean"] = (sensor, "mean")
        agg[f"{sensor}_median"] = (sensor, "median")
        agg[f"{sensor}_std"] = (sensor, "std")
        agg[f"{sensor}_minimum"] = (sensor, "min")
        agg[f"{sensor}_maximum"] = (sensor, "max")
        agg[f"{sensor}_unique_values"] = (sensor, "nunique")

    daily = timed.groupby("namibia_date").agg(
        observation_count=("source_row_number", "size"), **agg
    ).reset_index()
    hourly = timed.groupby("namibia_hour").agg(
        observation_count=("source_row_number", "size"), **agg
    ).reset_index()
    daily.to_csv(OUTPUT_DIR / "06_daily_stability_summary.csv", index=False)
    hourly.to_csv(OUTPUT_DIR / "07_hourly_stability_summary.csv", index=False)

    # E. Record-level event flags, merged back by unique source row number.
    event_flags = change_records[["source_row_number", *abrupt_cols, "flag_any_abrupt_change_review"]]
    df = df.merge(event_flags, on="source_row_number", how="left", validate="one_to_one")
    for c in abrupt_cols + ["flag_any_abrupt_change_review"]:
        df[c] = df[c].fillna(False).astype(bool)

    # Mark membership in duplicate timestamp groups.
    duplicate_source_rows = set(duplicate_rows["source_row_number"].tolist()) if len(duplicate_rows) else set()
    df["flag_duplicate_timestamp_review"] = df["source_row_number"].isin(duplicate_source_rows)
    df.to_csv(OUTPUT_DIR / "08_all_candidates_with_temporal_flags.csv", index=False)

    # F. Report.
    interval = timed["interval_minutes"].dropna()
    report = [
        "BOREMAN STEP 21 - TEMPORAL BEHAVIOUR AND SENSOR STABILITY",
        "=" * 68,
        f"Input: {INPUT_FILE}",
        f"Input records: {input_count:,}",
        f"Valid Namibia timestamps: {df[TIME_COL].notna().sum():,}",
        f"Duplicate timestamp records: {len(duplicate_rows):,}",
        f"Records flagged for abrupt changes: {int(change_records['flag_any_abrupt_change_review'].sum()):,}",
        f"Constant plateau segments (minimum {MIN_PLATEAU_RECORDS} records): {len(plateau_segments_df):,}",
        f"Records preserved: {len(df):,}",
        "",
        "Sampling intervals (minutes):",
        f"Count: {len(interval):,}",
        f"Median: {interval.median() if len(interval) else np.nan}",
        f"Maximum: {interval.max() if len(interval) else np.nan}",
        "",
        "Abrupt change thresholds:",
        pd.DataFrame(event_summary).to_string(index=False),
        "",
        "Interpretation:",
        "- Duplicate timestamps and abrupt changes are review flags, not automatic errors.",
        "- Abrupt-change thresholds use non-zero consecutive changes with intervals > 0 and <= 5 minutes.",
        "- Constant plateaus are excluded from the abrupt-change baseline so they cannot force MAD to zero.",
        "- A constant plateau is not proof of sensor malfunction.",
        "- No records were deleted or imputed.",
        "- All time-based summaries use Africa/Windhoek.",
    ]
    (OUTPUT_DIR / "09_step21_temporal_stability_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    # Assertions.
    if len(df) != input_count:
        raise AssertionError("Step 21 changed the input row count.")
    if df["source_row_number"].duplicated().any():
        raise AssertionError("Duplicate source_row_number found.")
    if df["source_row_number"].isna().any():
        raise AssertionError("Missing source_row_number found.")

    print("\nSTEP 21 VALIDATION")
    print("-" * 60)
    print(f"Rows preserved: {len(df):,}")
    print(f"Duplicate timestamp records: {len(duplicate_rows):,}")
    print(f"Abrupt-change review records: {int(change_records['flag_any_abrupt_change_review'].sum()):,}")
    print(f"Constant plateau segments: {len(plateau_segments_df):,}")
    print("All row-preservation checks passed.")
    print("\nBOREMAN STEP 21 COMPLETED")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Please upload 01_duplicate_timestamp_review.csv,")
    print("04_abrupt_change_threshold_summary.csv,")
    print("05_constant_plateau_segments.csv, and")
    print("09_step21_temporal_stability_report.txt for review.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
