"""
BOREMAN Step 25B - AI Feature Engineering
Creates separate primary-baseline and review feature datasets.

Input:
outputs/step23_modelling_readiness/01_all_records_with_readiness_labels.csv

Safety:
- Does not modify the source dataset.
- Does not delete or impute observations.
- Keeps existing quality flags and proposed temporal split.
- Uses timestamp_namibia as the canonical time source.
- Creates features separately within each modelling-readiness group.
- Breaks feature history at duplicate/non-positive intervals and gaps > 5 minutes.
- Rolling windows are observation-count windows, not clock-time windows.
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILE = Path(
    r"outputs\step23_modelling_readiness\01_all_records_with_readiness_labels.csv"
)
OUTPUT_DIR = Path(r"outputs\step25B_ai_feature_engineering")

TIME_COL = "timestamp_namibia"
SENSORS = ["tds", "humidity", "Temperature"]
GROUP_COL = "modelling_readiness_group"
PRIMARY_GROUP = "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS"
MAX_CONTIGUOUS_GAP_MINUTES = 5.0
LAGS = [1, 3, 5]
ROLLING_WINDOWS = [5, 15, 30]


def add_features_to_group(group_df):
    """Create causal features within one readiness group, resetting at gaps."""
    group_df = group_df.sort_values(
        [TIME_COL, "source_row_number"], kind="stable"
    ).copy()

    # UTC parsing preserves the instant; conversion restores Namibia local time.
    ts_utc = pd.to_datetime(group_df[TIME_COL], errors="coerce", utc=True)
    group_df["_timestamp_local"] = ts_utc.dt.tz_convert("Africa/Windhoek")
    group_df["_time_delta_minutes"] = (
        group_df["_timestamp_local"].diff().dt.total_seconds() / 60.0
    )

    # A new segment begins at the first row, duplicate/non-positive intervals,
    # invalid timestamps, or intervals exceeding the conservative 5-minute cap.
    new_segment = (
        group_df["_time_delta_minutes"].isna()
        | (group_df["_time_delta_minutes"] <= 0)
        | (group_df["_time_delta_minutes"] > MAX_CONTIGUOUS_GAP_MINUTES)
    )
    group_df["_segment_id"] = new_segment.cumsum()

    feature_frames = []
    for _, segment in group_df.groupby("_segment_id", sort=False):
        segment = segment.copy()

        for sensor in SENSORS:
            values = pd.to_numeric(segment[sensor], errors="coerce")

            segment[f"{sensor}_current"] = values
            segment[f"{sensor}_change_from_previous"] = values.diff()
            segment[f"{sensor}_rate_per_minute"] = (
                values.diff() / segment["_time_delta_minutes"].replace(0, np.nan)
            )

            for lag in LAGS:
                segment[f"{sensor}_lag_{lag}"] = values.shift(lag)

            for window in ROLLING_WINDOWS:
                prior_values = values.shift(1)
                segment[f"{sensor}_rolling_mean_previous_{window}"] = (
                    prior_values.rolling(window=window, min_periods=2).mean()
                )
                segment[f"{sensor}_rolling_std_previous_{window}"] = (
                    prior_values.rolling(window=window, min_periods=2).std()
                )
                segment[f"{sensor}_deviation_from_previous_{window}_mean"] = (
                    values
                    - prior_values.rolling(window=window, min_periods=2).mean()
                )

        local_time = segment["_timestamp_local"]
        hour = local_time.dt.hour + local_time.dt.minute / 60.0
        weekday = local_time.dt.dayofweek

        segment["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        segment["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        segment["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
        segment["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
        segment["namibia_date_feature"] = local_time.dt.strftime("%Y-%m-%d")
        segment["namibia_hour_feature"] = local_time.dt.hour
        segment["sampling_interval_minutes"] = segment["_time_delta_minutes"]
        segment["feature_segment_id"] = segment["_segment_id"]

        feature_frames.append(segment)

    if not feature_frames:
        return group_df.iloc[0:0].copy()

    return pd.concat(feature_frames, axis=0).sort_index()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run this script from the BOREMAN project root."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [
        "source_row_number", TIME_COL, GROUP_COL, *SENSORS,
        "proposed_temporal_split"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from input: {missing}")

    original_rows = len(df)
    original_source_ids = df["source_row_number"].copy()

    # Validate canonical timestamps; do not fall back to ambiguous timestamp fields.
    parsed = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    invalid_count = int(parsed.isna().sum())
    if invalid_count:
        raise ValueError(
            f"{invalid_count} canonical Namibia timestamps could not be parsed. "
            "Stop and investigate rather than dropping these records."
        )

    # Verify the stored timestamps carry the expected Namibia UTC offset.
    local = parsed.dt.tz_convert("Africa/Windhoek")
    df[TIME_COL] = local.map(lambda x: x.isoformat())

    # Numeric conversion is validation only; fail if sensor values become missing.
    for sensor in SENSORS:
        converted = pd.to_numeric(df[sensor], errors="coerce")
        if converted.isna().any():
            raise ValueError(
                f"{sensor} contains missing/non-numeric values after conversion. "
                "No rows were removed."
            )
        df[sensor] = converted

    if df[GROUP_COL].isna().any():
        raise ValueError("Missing modelling readiness group values detected.")

    # Preserve source columns and add engineered features group by group.
    outputs = []
    for group_name, group_df in df.groupby(GROUP_COL, sort=False):
        engineered = add_features_to_group(group_df)
        outputs.append(engineered)

    all_features = pd.concat(outputs, axis=0).sort_values(
        "source_row_number", kind="stable"
    )

    # Drop internal working timestamp; keep canonical timestamp_namibia.
    all_features = all_features.drop(
        columns=["_timestamp_local", "_time_delta_minutes", "_segment_id"],
        errors="ignore"
    )

    # Split primary baseline from all other review groups.
    primary = all_features[
        all_features[GROUP_COL] == PRIMARY_GROUP
    ].copy()
    review = all_features[
        all_features[GROUP_COL] != PRIMARY_GROUP
    ].copy()

    # Feature dictionary documents generated feature families.
    feature_rows = []
    for sensor in SENSORS:
        feature_rows.extend([
            {
                "feature": f"{sensor}_current",
                "description": f"Current observed {sensor} value",
                "family": "current_measurement",
                "causal": "Yes",
            },
            {
                "feature": f"{sensor}_change_from_previous",
                "description": f"Difference from previous observation in same segment",
                "family": "change",
                "causal": "Yes",
            },
            {
                "feature": f"{sensor}_rate_per_minute",
                "description": f"Change divided by elapsed minutes",
                "family": "rate_of_change",
                "causal": "Yes",
            },
        ])
        for lag in LAGS:
            feature_rows.append({
                "feature": f"{sensor}_lag_{lag}",
                "description": f"{sensor} value {lag} observations earlier in same segment",
                "family": "lag",
                "causal": "Yes",
            })
        for window in ROLLING_WINDOWS:
            for suffix, description in [
                ("mean", "mean of preceding observations, excluding current row"),
                ("std", "standard deviation of preceding observations, excluding current row"),
                ("deviation_from_mean", "current value minus preceding-observation mean"),
            ]:
                feature_rows.append({
                    "feature": (
                        f"{sensor}_rolling_{suffix}_previous_{window}"
                        if suffix in ["mean", "std"]
                        else f"{sensor}_deviation_from_previous_{window}_mean"
                    ),
                    "description": f"{sensor}: {description}; window={window} observations",
                    "family": "rolling_statistics",
                    "causal": "Yes",
                })

    feature_rows.extend([
        {"feature": "hour_sin", "description": "Cyclical Namibia local hour sine", "family": "time", "causal": "Yes"},
        {"feature": "hour_cos", "description": "Cyclical Namibia local hour cosine", "family": "time", "causal": "Yes"},
        {"feature": "weekday_sin", "description": "Cyclical weekday sine", "family": "time", "causal": "Yes"},
        {"feature": "weekday_cos", "description": "Cyclical weekday cosine", "family": "time", "causal": "Yes"},
        {"feature": "sampling_interval_minutes", "description": "Interval from previous observation in same segment", "family": "sampling", "causal": "Yes"},
        {"feature": "feature_segment_id", "description": "Sequence segment identifier; resets at gaps/duplicates", "family": "sequence_control", "causal": "Yes"},
    ])
    feature_dictionary = pd.DataFrame(feature_rows).drop_duplicates(
        subset=["feature"]
    )

    # Save outputs.
    all_features.to_csv(
        OUTPUT_DIR / "01_all_records_with_engineered_features.csv",
        index=False
    )
    primary.to_csv(
        OUTPUT_DIR / "02_primary_candidate_baseline_features.csv",
        index=False
    )
    review.to_csv(
        OUTPUT_DIR / "03_secondary_and_plateau_review_features.csv",
        index=False
    )
    feature_dictionary.to_csv(
        OUTPUT_DIR / "04_feature_dictionary.csv",
        index=False
    )

    # Split coverage summary for primary candidates only.
    split_summary = (
        primary.groupby("proposed_temporal_split", dropna=False)
        .agg(
            record_count=("source_row_number", "count"),
            first_date=("namibia_date_feature", "min"),
            last_date=("namibia_date_feature", "max"),
        )
        .reset_index()
    )
    split_summary["percentage"] = (
        split_summary["record_count"] / len(primary) * 100
    ).round(4)
    split_summary.to_csv(
        OUTPUT_DIR / "05_primary_split_feature_coverage.csv",
        index=False
    )

    # Integrity assertions.
    if len(all_features) != original_rows:
        raise AssertionError("Row count changed during feature engineering.")
    if set(all_features["source_row_number"].tolist()) != set(
        original_source_ids.tolist()
    ):
        raise AssertionError("Source row identity check failed.")
    if all_features["source_row_number"].duplicated().any():
        raise AssertionError("Duplicate source_row_number detected.")
    if len(primary) + len(review) != original_rows:
        raise AssertionError("Primary and review datasets do not reconcile.")

    report = [
        "BOREMAN STEP 25B - AI FEATURE ENGINEERING REPORT",
        "=" * 60,
        f"Input records: {original_rows:,}",
        f"Output records: {len(all_features):,}",
        f"Primary baseline records: {len(primary):,}",
        f"Secondary/plateau review records: {len(review):,}",
        f"Canonical timestamp: {TIME_COL}",
        "Timezone: Africa/Windhoek",
        f"Invalid canonical timestamps: {invalid_count}",
        f"Maximum contiguous gap: {MAX_CONTIGUOUS_GAP_MINUTES} minutes",
        f"Lag observation counts: {LAGS}",
        f"Rolling observation windows: {ROLLING_WINDOWS}",
        "Rolling windows exclude the current observation.",
        "Feature histories are isolated by modelling-readiness group.",
        "Feature histories reset at duplicate/non-positive intervals and gaps > 5 minutes.",
        "No records deleted or imputed.",
        "No model trained.",
        "Source row identity and count checks: PASSED",
        "",
        "IMPORTANT INTERPRETATION:",
        "- Rolling windows count observations, not fixed minutes.",
        "- Sampling cadence is not perfectly uniform.",
        "- Primary candidates are a baseline-development subset, not confirmed normal ground truth.",
        "- Existing review flags are retained and must not be treated as verified fault labels.",
        "- Proposed temporal split labels are retained unchanged.",
    ]
    (OUTPUT_DIR / "06_step25B_feature_engineering_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n========== BOREMAN STEP 25B COMPLETE ==========")
    print(f"Input records: {original_rows:,}")
    print(f"All engineered records: {len(all_features):,}")
    print(f"Primary baseline records: {len(primary):,}")
    print(f"Secondary/plateau review records: {len(review):,}")
    print("Source row identity and count checks: PASSED")
    print("No records deleted or imputed. No model trained.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("===============================================\n")


if __name__ == "__main__":
    main()
