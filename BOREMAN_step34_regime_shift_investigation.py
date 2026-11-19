"""
BOREMAN STEP 34 - REGIME-SHIFT INVESTIGATION

Investigates why model performance changes across the three chronological
walk-forward folds. Uses TRAIN-period supervised samples only and joins
source quality flags from Step 25B by origin source row number.

No model training. No final VALIDATION or TEST data is used.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUPERVISED_FILE = ROOT / "outputs" / "step27_time_series_forecasting" / "01_forecast_supervised_dataset.csv"
ENGINEERED_FILE = ROOT / "outputs" / "step25B_ai_feature_engineering" / "01_all_records_with_engineered_features.csv"
OUT_DIR = ROOT / "outputs" / "step34_regime_shift_investigation"

TARGETS = ["tds", "humidity", "Temperature"]
SENSOR_COLS = {
    "tds": "origin_tds",
    "humidity": "origin_humidity",
    "Temperature": "origin_Temperature",
}
FLAG_COLS = [
    "flag_constant_plateau",
    "flag_tds_zero",
    "flag_any_iqr_review",
    "flag_any_abrupt_change_review",
    "flag_duplicate_timestamp_review",
    "flag_confirmed_testing",
]

def safe_bool(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def describe_sensor(frame, split_label, group_label, sensor, col):
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "period_group": group_label,
        "split": split_label,
        "sensor": sensor,
        "n": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "median": values.median(),
        "q05": values.quantile(0.05),
        "q25": values.quantile(0.25),
        "q75": values.quantile(0.75),
        "q95": values.quantile(0.95),
        "minimum": values.min(),
        "maximum": values.max(),
    }

def main():
    print("=" * 72)
    print("BOREMAN STEP 34 - REGIME-SHIFT INVESTIGATION")
    print("=" * 72)

    for file in [SUPERVISED_FILE, ENGINEERED_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"Required input not found: {file}")

    print("\nReading TRAIN supervised samples and Step 25B quality flags...")
    sup = pd.read_csv(SUPERVISED_FILE)
    eng = pd.read_csv(ENGINEERED_FILE)

    required_sup = [
        "origin_source_row_number", "target_source_row_number",
        "origin_timestamp_namibia", "target_timestamp_namibia",
        "requested_horizon_minutes", "proposed_temporal_split",
    ] + list(SENSOR_COLS.values()) + [f"actual_future_{t}" for t in TARGETS]
    missing = [c for c in required_sup if c not in sup.columns]
    if missing:
        raise ValueError(f"Missing supervised columns: {missing}")
    required_eng = ["source_row_number", "timestamp_namibia", "tds", "humidity", "Temperature"]
    missing = [c for c in required_eng if c not in eng.columns]
    if missing:
        raise ValueError(f"Missing Step 25B columns: {missing}")

    sup = sup[sup["proposed_temporal_split"].eq("TRAIN")].copy()
    sup["origin_dt"] = pd.to_datetime(sup["origin_timestamp_namibia"], utc=True, errors="coerce")
    sup["target_dt"] = pd.to_datetime(sup["target_timestamp_namibia"], utc=True, errors="coerce")
    sup = sup.dropna(subset=["origin_dt", "target_dt"])
    sup["origin_date"] = sup["origin_dt"].dt.date

    dates = sorted(sup["origin_date"].dropna().unique())
    if len(dates) < 11:
        raise ValueError(f"Expected 11 TRAIN dates for the three folds; found {len(dates)}")

    # Exactly match Step 32's three expanding validation windows.
    fold_specs = [(1, 5, 7), (2, 7, 9), (3, 9, 11)]
    fold_frames = []
    for fold_no, train_end, val_end in fold_specs:
        valid_dates = set(dates[train_end:val_end])
        fold = sup[sup["origin_date"].isin(valid_dates)].copy()
        fold["walk_forward_fold"] = f"FOLD_{fold_no}"
        fold_frames.append(fold)
    fold_data = pd.concat(fold_frames, ignore_index=True)
    fold_data["requested_horizon_minutes"] = pd.to_numeric(
        fold_data["requested_horizon_minutes"], errors="coerce"
    )
    # One row per unique origin is used for origin-level summaries so that
    # origins with more eligible horizons are not unintentionally overweighted.
    fold_origins = (
        fold_data.sort_values(["walk_forward_fold", "origin_dt"])
        .drop_duplicates(["walk_forward_fold", "origin_source_row_number"])
        .copy()
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Origin distributions use unique origins; future-target distributions
    # are kept separate by requested horizon.
    distribution_rows = []
    for fold_name, group in fold_origins.groupby("walk_forward_fold"):
        for sensor in TARGETS:
            row = describe_sensor(group, "TRAIN", fold_name, sensor, SENSOR_COLS[sensor])
            if row:
                row["measurement_position"] = "origin"
                row["horizon_minutes"] = np.nan
                distribution_rows.append(row)
    for (fold_name, horizon), group in fold_data.groupby(
        ["walk_forward_fold", "requested_horizon_minutes"]
    ):
        for sensor in TARGETS:
            row = describe_sensor(group, "TRAIN", fold_name, sensor, f"actual_future_{sensor}")
            if row:
                row["measurement_position"] = "future_target"
                row["horizon_minutes"] = int(horizon)
                distribution_rows.append(row)
    distributions = pd.DataFrame(distribution_rows)
    distributions.to_csv(OUT_DIR / "01_sensor_distributions_by_fold.csv", index=False)

    # Correlations calculated separately for each fold using origin readings.
    correlation_rows = []
    for fold_name, group in fold_origins.groupby("walk_forward_fold"):
        frame = group[list(SENSOR_COLS.values())].rename(
            columns={v: k for k, v in SENSOR_COLS.items()}
        )
        corr = frame.corr(method="pearson")
        for a in TARGETS:
            for b in TARGETS:
                correlation_rows.append({
                    "fold": fold_name, "sensor_a": a, "sensor_b": b,
                    "pearson_correlation": corr.loc[a, b],
                    "sample_count": int(frame[[a, b]].dropna().shape[0]),
                })
    pd.DataFrame(correlation_rows).to_csv(
        OUT_DIR / "02_sensor_correlations_by_fold.csv", index=False
    )

    # Standardized mean differences between fold 2/3 and fold 1.
    feature_cols = [
        "tds_current", "humidity_current", "Temperature_current",
        "tds_rate_per_minute", "humidity_rate_per_minute", "Temperature_rate_per_minute",
        "tds_rolling_std_previous_5", "humidity_rolling_std_previous_5",
        "Temperature_rolling_std_previous_5", "sampling_interval_minutes",
    ]
    missing_features = [c for c in feature_cols if c not in fold_data.columns]
    if missing_features:
        raise ValueError(f"Missing engineered features in supervised data: {missing_features}")
    drift_rows = []
    base = fold_origins[fold_origins["walk_forward_fold"].eq("FOLD_1")]
    for feature in feature_cols:
        base_values = pd.to_numeric(base[feature], errors="coerce").dropna()
        base_mean = base_values.mean()
        base_sd = base_values.std()
        for fold_name in ["FOLD_2", "FOLD_3"]:
            values = pd.to_numeric(
                fold_origins.loc[fold_origins["walk_forward_fold"].eq(fold_name), feature],
                errors="coerce"
            ).dropna()
            mean = values.mean()
            sd = values.std()
            pooled_sd = np.sqrt((base_sd ** 2 + sd ** 2) / 2) if pd.notna(sd) else np.nan
            smd = (mean - base_mean) / pooled_sd if pd.notna(pooled_sd) and pooled_sd > 0 else np.nan
            drift_rows.append({
                "feature": feature, "reference_fold": "FOLD_1",
                "comparison_fold": fold_name,
                "reference_n": len(base_values), "comparison_n": len(values),
                "reference_mean": base_mean, "comparison_mean": mean,
                "reference_std": base_sd, "comparison_std": sd,
                "standardized_mean_difference": smd,
            })
    pd.DataFrame(drift_rows).to_csv(
        OUT_DIR / "03_feature_drift_vs_fold1.csv", index=False
    )

    # Join source quality flags to the origin observation. A source row can
    # appear repeatedly at different forecast horizons, so report unique origins.
    available_flags = [c for c in FLAG_COLS if c in eng.columns]
    if not available_flags:
        raise ValueError("No expected quality flag columns found in Step 25B dataset.")
    flag_source = eng[["source_row_number"] + available_flags].drop_duplicates("source_row_number")
    for col in available_flags:
        flag_source[col] = safe_bool(flag_source[col])
    origins = fold_origins[
        ["origin_source_row_number", "walk_forward_fold"]
    ].drop_duplicates()
    origins = origins.merge(
        flag_source, left_on="origin_source_row_number",
        right_on="source_row_number", how="left", validate="many_to_one"
    )
    flag_rows = []
    for fold_name, group in origins.groupby("walk_forward_fold"):
        for flag in available_flags:
            flag_rows.append({
                "fold": fold_name,
                "quality_flag": flag,
                "unique_origin_records": int(group["origin_source_row_number"].nunique()),
                "flagged_origin_records": int(group.loc[group[flag].fillna(False), "origin_source_row_number"].nunique()),
                "flagged_percentage": float(group[flag].fillna(False).mean() * 100),
            })
    pd.DataFrame(flag_rows).to_csv(
        OUT_DIR / "04_quality_flag_composition_by_fold.csv", index=False
    )

    # Daily target statistics within each fold, to identify gradual vs abrupt shifts.
    daily_rows = []
    fold_dates = fold_origins[["origin_date", "walk_forward_fold"]].drop_duplicates()
    for (fold_name, date), group in fold_origins.groupby(["walk_forward_fold", "origin_date"]):
        row = {
            "fold": fold_name,
            "namibia_date": str(date),
            "supervised_rows": len(group),
            "unique_origin_records": group["origin_source_row_number"].nunique(),
        }
        for sensor in TARGETS:
            vals = pd.to_numeric(group[SENSOR_COLS[sensor]], errors="coerce").dropna()
            row[f"{sensor}_origin_mean"] = vals.mean()
            row[f"{sensor}_origin_median"] = vals.median()
            row[f"{sensor}_origin_std"] = vals.std()
        daily_rows.append(row)
    pd.DataFrame(daily_rows).sort_values(["fold", "namibia_date"]).to_csv(
        OUT_DIR / "05_daily_sensor_regime_summary.csv", index=False
    )

    # Summarise fold membership and coverage.
    coverage = []
    for fold_name, group in fold_origins.groupby("walk_forward_fold"):
        horizon_group = fold_data[fold_data["walk_forward_fold"].eq(fold_name)]
        coverage.append({
            "fold": fold_name,
            "start_date": str(group["origin_date"].min()),
            "end_date": str(group["origin_date"].max()),
            "unique_origin_records": group["origin_source_row_number"].nunique(),
            "supervised_rows_across_horizons": len(horizon_group),
            "unique_target_records": horizon_group["target_source_row_number"].nunique(),
            "horizons_present": ",".join(map(str, sorted(horizon_group["requested_horizon_minutes"].dropna().unique()))),
        })
    pd.DataFrame(coverage).to_csv(
        OUT_DIR / "06_fold_coverage.csv", index=False
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure all tables are saved after directory creation (directory is created
    # here too late for earlier writes; create before writing in the final run).
    report = [
        "BOREMAN STEP 34 - REGIME-SHIFT INVESTIGATION REPORT",
        "=" * 72,
        f"TRAIN supervised rows read: {len(sup):,}",
        f"Distinct TRAIN origin dates: {len(dates)}",
        f"Walk-forward fold rows: {len(fold_data):,}",
        "",
        "METHOD",
        "- Uses TRAIN-period supervised records only.",
        "- Fold validation dates exactly follow Step 32: dates 6-7, 8-9 and 10-11.",
        "- Compares sensor distributions, correlations, engineered-feature drift,",
        "  source quality flags and daily sensor summaries across folds.",
        "- Quality flags are joined by origin source row number.",
        "- Repeated forecast horizons are reduced to unique origin records for flag counts.",
        "- No model retraining and no final TEST data use.",
        "",
        "INTERPRETATION",
        "- Distribution differences are descriptive and do not establish causation.",
        "- Correlation differences do not establish physical or causal relationships.",
        "- Quality flags indicate review conditions, not confirmed sensor faults.",
        "- The folds are overlapping expanding-window validation periods, not independent datasets.",
        "- This analysis concerns sensor readings, not water safety.",
        "",
        "OUTPUTS",
        "- 01_sensor_distributions_by_fold.csv",
        "- 02_sensor_correlations_by_fold.csv",
        "- 03_feature_drift_vs_fold1.csv",
        "- 04_quality_flag_composition_by_fold.csv",
        "- 05_daily_sensor_regime_summary.csv",
        "- 06_fold_coverage.csv",
    ]
    (OUT_DIR / "07_step34_regime_shift_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nSTEP 34 COMPLETED")
    print(f"TRAIN supervised rows: {len(sup):,}")
    print(f"Fold rows analysed: {len(fold_data):,}")
    print(f"Output folder: {OUT_DIR}")
    print("No model training; final VALIDATION and TEST data were not used.")

if __name__ == "__main__":
    main()
