from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
INPUT = ROOT / "outputs" / "step25B_ai_feature_engineering" / "01_all_records_with_engineered_features.csv"
OUT = ROOT / "outputs" / "step35_environmental_regime_timeofday_v2"

EXPECTED_FOLDS = {
    "FOLD_1": ["2026-10-09", "2026-10-10"],
    "FOLD_2": ["2026-10-11", "2026-10-12"],
    "FOLD_3": ["2026-10-13", "2026-10-14"],
}

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    required = [
        "source_row_number", "timestamp_namibia", "tds", "humidity",
        "Temperature", "modelling_readiness_group", "proposed_temporal_split"
    ]
    df = pd.read_csv(INPUT, low_memory=False)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    input_rows = len(df)
    split_values = sorted(df["proposed_temporal_split"].dropna().astype(str).unique())
    group_values = sorted(df["modelling_readiness_group"].dropna().astype(str).unique())

    split_norm = df["proposed_temporal_split"].astype(str).str.strip().str.upper()
    group_norm = df["modelling_readiness_group"].astype(str).str.strip().str.upper()

    # Strictly restrict to TRAIN and primary candidate records.
    train_mask = split_norm.eq("TRAIN")
    primary_mask = group_norm.str.contains("PRIMARY", na=False)
    filtered = df[train_mask & primary_mask].copy()

    if filtered.empty:
        raise ValueError(
            "No rows matched proposed_temporal_split == 'TRAIN' and "
            "modelling_readiness_group containing 'PRIMARY'. "
            f"Split values: {split_values}; readiness values: {group_values}"
        )

    filtered["_ts"] = pd.to_datetime(filtered["timestamp_namibia"], errors="coerce", utc=True)
    filtered = filtered[filtered["_ts"].notna()].copy()
    filtered["_ts"] = filtered["_ts"].dt.tz_convert("Africa/Windhoek")

    for c in ["tds", "humidity", "Temperature"]:
        filtered[c] = pd.to_numeric(filtered[c], errors="coerce")

    # One row per source observation; preserve source file unchanged.
    before_dedup = len(filtered)
    filtered = filtered.drop_duplicates(subset=["source_row_number"], keep="first").copy()
    duplicates_removed = before_dedup - len(filtered)

    filtered["_date"] = filtered["_ts"].dt.strftime("%Y-%m-%d")
    filtered["_hour"] = filtered["_ts"].dt.hour
    filtered["_time_band"] = np.where(
        (filtered["_hour"] >= 6) & (filtered["_hour"] < 18),
        "DAY_06_18", "NIGHT_18_06"
    )

    actual_dates = sorted(filtered["_date"].dropna().unique().tolist())
    expected_dates = sorted({d for dates in EXPECTED_FOLDS.values() for d in dates})

    # Confirm the Step 32 training window exactly; do not silently infer new folds.
    if actual_dates != sorted([
        "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
        "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11",
        "2026-10-12", "2026-10-13", "2026-10-14"
    ]):
        raise ValueError(
            "Filtered primary TRAIN dates do not match the 11-date Step 32 TRAIN window. "
            f"Found {len(actual_dates)} dates: {actual_dates}. "
            "Stop here; do not reinterpret the folds."
        )

    fold_map = {d: fold for fold, dates in EXPECTED_FOLDS.items() for d in dates}
    fold_df = filtered[filtered["_date"].isin(expected_dates)].copy()
    fold_df["_fold"] = fold_df["_date"].map(fold_map)

    # All summaries are based on unique source observations, not repeated horizons.
    daily = (
        fold_df.groupby(["_fold", "_date"])
        .agg(
            records=("source_row_number", "nunique"),
            tds_mean=("tds", "mean"), tds_median=("tds", "median"), tds_std=("tds", "std"),
            humidity_mean=("humidity", "mean"), humidity_median=("humidity", "median"), humidity_std=("humidity", "std"),
            temperature_mean=("Temperature", "mean"), temperature_median=("Temperature", "median"), temperature_std=("Temperature", "std")
        ).reset_index()
    )

    hourly = (
        fold_df.groupby(["_fold", "_hour"])
        .agg(
            records=("source_row_number", "nunique"),
            tds_mean=("tds", "mean"), tds_std=("tds", "std"),
            humidity_mean=("humidity", "mean"), humidity_std=("humidity", "std"),
            temperature_mean=("Temperature", "mean"), temperature_std=("Temperature", "std")
        ).reset_index()
    )

    daynight = (
        fold_df.groupby(["_fold", "_time_band"])
        .agg(
            records=("source_row_number", "nunique"),
            tds_mean=("tds", "mean"), tds_median=("tds", "median"), tds_std=("tds", "std"),
            humidity_mean=("humidity", "mean"), humidity_median=("humidity", "median"), humidity_std=("humidity", "std"),
            temperature_mean=("Temperature", "mean"), temperature_median=("Temperature", "median"), temperature_std=("Temperature", "std")
        ).reset_index()
    )

    corr_rows = []
    for (fold, band), g in fold_df.groupby(["_fold", "_time_band"]):
        corr_rows.append({
            "fold": fold, "time_band": band, "records": g["source_row_number"].nunique(),
            "tds_humidity_corr": g[["tds", "humidity"]].corr().iloc[0, 1],
            "tds_temperature_corr": g[["tds", "Temperature"]].corr().iloc[0, 1],
            "humidity_temperature_corr": g[["humidity", "Temperature"]].corr().iloc[0, 1],
        })
    correlations = pd.DataFrame(corr_rows)

    variability = (
        fold_df.groupby(["_fold", "_time_band"])
        .agg(
            records=("source_row_number", "nunique"),
            tds_min=("tds", "min"), tds_max=("tds", "max"),
            tds_iqr=("tds", lambda s: s.quantile(.75) - s.quantile(.25)),
            humidity_min=("humidity", "min"), humidity_max=("humidity", "max"),
            humidity_iqr=("humidity", lambda s: s.quantile(.75) - s.quantile(.25)),
            temperature_min=("Temperature", "min"), temperature_max=("Temperature", "max"),
            temperature_iqr=("Temperature", lambda s: s.quantile(.75) - s.quantile(.25))
        ).reset_index()
    )

    coverage = (
        fold_df.groupby("_fold")
        .agg(
            records=("source_row_number", "nunique"),
            distinct_dates=("_date", "nunique"),
            first_timestamp=("_ts", "min"),
            last_timestamp=("_ts", "max")
        ).reset_index().rename(columns={"_fold": "fold"})
    )
    coverage["first_timestamp"] = coverage["first_timestamp"].astype(str)
    coverage["last_timestamp"] = coverage["last_timestamp"].astype(str)

    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "01_daily_regime_summary.csv", index=False)
    hourly.to_csv(OUT / "02_hourly_regime_summary.csv", index=False)
    daynight.to_csv(OUT / "03_day_night_comparison_by_fold.csv", index=False)
    correlations.to_csv(OUT / "04_correlations_by_fold_and_timeband.csv", index=False)
    variability.to_csv(OUT / "05_sensor_variability_by_fold_and_timeband.csv", index=False)
    coverage.to_csv(OUT / "06_fold_time_coverage.csv", index=False)

    report = [
        "BOREMAN STEP 35 V2 - ENVIRONMENTAL REGIME AND TIME-OF-DAY INVESTIGATION",
        "=" * 78,
        f"Input rows read: {input_rows}",
        f"Rows matching TRAIN + PRIMARY before timestamp/dedup checks: {before_dedup}",
        f"Unique primary TRAIN observations after deduplication: {len(filtered)}",
        f"Duplicate source rows removed for analysis only: {duplicates_removed}",
        f"Distinct primary TRAIN dates: {len(actual_dates)}",
        "",
        "EXACT STEP 32 FOLD VALIDATION WINDOWS",
    ]
    for fold, dates in EXPECTED_FOLDS.items():
        report.append(f"- {fold}: {', '.join(dates)}")
    report += [
        "",
        "METHOD",
        "- Explicit filter: proposed_temporal_split == TRAIN.",
        "- Explicit filter: modelling_readiness_group contains PRIMARY.",
        "- Exact Step 32 validation dates are hard-coded and checked against the 11-date TRAIN window.",
        "- Canonical timestamp_namibia converted to Africa/Windhoek.",
        "- DAY is fixed Namibia clock time 06:00-18:00; NIGHT is 18:00-06:00.",
        "- One source observation counted once; no repeated forecast horizons.",
        "- Record-weighted summaries; no interpolation, imputation or model training.",
        "- Original source data is not modified.",
        "",
        "LIMITATIONS",
        "- Day/night windows are fixed clock periods, not astronomical sunrise/sunset.",
        "- Results are descriptive, not causal; flags and readings do not establish sensor faults.",
        "- Unequal sampling can affect record-weighted summaries.",
        "- Sensor readings alone cannot establish water safety.",
        "",
        "OUTPUTS",
        "- 01_daily_regime_summary.csv",
        "- 02_hourly_regime_summary.csv",
        "- 03_day_night_comparison_by_fold.csv",
        "- 04_correlations_by_fold_and_timeband.csv",
        "- 05_sensor_variability_by_fold_and_timeband.csv",
        "- 06_fold_time_coverage.csv",
        "- 07_step35_environmental_regime_report.txt",
    ]
    (OUT / "07_step35_environmental_regime_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("BOREMAN STEP 35 V2 - ENVIRONMENTAL REGIME AND TIME-OF-DAY INVESTIGATION")
    print("=" * 76)
    print(f"Input rows: {input_rows:,}")
    print(f"TRAIN + PRIMARY rows before timestamp/dedup checks: {before_dedup:,}")
    print(f"Unique primary TRAIN observations: {len(filtered):,}")
    print(f"Duplicate source rows removed for analysis: {duplicates_removed:,}")
    print(f"Fold rows: {fold_df.groupby('_fold')['source_row_number'].nunique().to_dict()}")
    print(f"Output folder: {OUT.resolve()}")
    print("Exact Step 32 fold dates verified. No model training; no VALIDATION/TEST rows used.")

if __name__ == "__main__":
    main()
