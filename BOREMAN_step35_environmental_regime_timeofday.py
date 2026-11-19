from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path.cwd()
INPUT = ROOT / "outputs" / "step25B_ai_feature_engineering" / "01_all_records_with_engineered_features.csv"
OUT = ROOT / "outputs" / "step35_environmental_regime_timeofday"
OUT.mkdir(parents=True, exist_ok=True)

def find_col(df, candidates, required=True):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise ValueError(f"Required column not found. Tried: {candidates}. Available columns: {list(df.columns)}")
    return None

def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce")

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)
    ts_col = find_col(df, ["timestamp_namibia"])
    tds_col = find_col(df, ["tds"])
    hum_col = find_col(df, ["humidity"])
    temp_col = find_col(df, ["Temperature", "temperature"])
    group_col = find_col(df, ["readiness_group"], required=False)
    split_col = find_col(df, ["split", "data_split"], required=False)
    row_col = find_col(df, ["source_row_number", "source_row", "source_row_num"], required=False)

    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    # Source timestamps are already represented as Namibia-local offsets; convert to Namibia time.
    df["_ts"] = df["_ts"].dt.tz_convert("Africa/Windhoek")
    df = df[df["_ts"].notna()].copy()
    for col in [tds_col, hum_col, temp_col]:
        df[col] = numeric(df, col)

    if group_col:
        primary_mask = df[group_col].astype(str).str.lower().str.contains("primary")
        df = df[primary_mask].copy()
    if split_col:
        train_mask = df[split_col].astype(str).str.lower().str.contains("train")
        df = df[train_mask].copy()

    # Retain one record per original observation. Prefer source row number when available.
    dedup_key = row_col if row_col else ts_col
    before = len(df)
    df = df.drop_duplicates(subset=[dedup_key], keep="first").copy()
    deduplicated = before - len(df)

    df["_date"] = df["_ts"].dt.strftime("%Y-%m-%d")
    df["_hour"] = df["_ts"].dt.hour
    df["_time_band"] = np.where((df["_hour"] >= 6) & (df["_hour"] < 18), "DAY_06_18", "NIGHT_18_06")

    dates = sorted(df["_date"].dropna().unique())
    if len(dates) < 11:
        raise ValueError(f"Expected at least 11 distinct TRAIN dates to reproduce Step 32 folds; found {len(dates)}: {dates}")

    fold_map = {}
    windows = [("FOLD_1", dates[5:7]), ("FOLD_2", dates[7:9]), ("FOLD_3", dates[9:11])]
    for fold, fold_dates in windows:
        for d in fold_dates:
            fold_map[d] = fold
    df["_fold"] = df["_date"].map(fold_map)
    df = df[df["_fold"].notna()].copy()

    # Daily and hourly summaries are record-weighted, not time-weighted.
    daily = (df.groupby(["_fold", "_date"], dropna=False)
             .agg(records=(tds_col, "size"),
                  tds_mean=(tds_col, "mean"), tds_median=(tds_col, "median"), tds_std=(tds_col, "std"),
                  humidity_mean=(hum_col, "mean"), humidity_median=(hum_col, "median"), humidity_std=(hum_col, "std"),
                  temperature_mean=(temp_col, "mean"), temperature_median=(temp_col, "median"), temperature_std=(temp_col, "std"))
             .reset_index())

    hourly = (df.groupby(["_fold", "_hour"], dropna=False)
              .agg(records=(tds_col, "size"),
                   tds_mean=(tds_col, "mean"), tds_std=(tds_col, "std"),
                   humidity_mean=(hum_col, "mean"), humidity_std=(hum_col, "std"),
                   temperature_mean=(temp_col, "mean"), temperature_std=(temp_col, "std"))
              .reset_index())

    daynight = (df.groupby(["_fold", "_time_band"], dropna=False)
                .agg(records=(tds_col, "size"),
                     tds_mean=(tds_col, "mean"), tds_median=(tds_col, "median"), tds_std=(tds_col, "std"),
                     humidity_mean=(hum_col, "mean"), humidity_median=(hum_col, "median"), humidity_std=(hum_col, "std"),
                     temperature_mean=(temp_col, "mean"), temperature_median=(temp_col, "median"), temperature_std=(temp_col, "std"))
                .reset_index())

    corr_rows = []
    for (fold, band), g in df.groupby(["_fold", "_time_band"]):
        corr_rows.append({
            "fold": fold, "time_band": band, "records": len(g),
            "tds_humidity_corr": g[[tds_col, hum_col]].corr().iloc[0, 1],
            "tds_temperature_corr": g[[tds_col, temp_col]].corr().iloc[0, 1],
            "humidity_temperature_corr": g[[hum_col, temp_col]].corr().iloc[0, 1],
        })
    correlations = pd.DataFrame(corr_rows)

    # Variability comparison helps separate shifts in average level from shifts in spread.
    variability = (df.groupby(["_fold", "_time_band"])
                   .agg(records=(tds_col, "size"),
                        tds_min=(tds_col, "min"), tds_max=(tds_col, "max"), tds_iqr=(tds_col, lambda s: s.quantile(.75)-s.quantile(.25)),
                        humidity_min=(hum_col, "min"), humidity_max=(hum_col, "max"), humidity_iqr=(hum_col, lambda s: s.quantile(.75)-s.quantile(.25)),
                        temperature_min=(temp_col, "min"), temperature_max=(temp_col, "max"), temperature_iqr=(temp_col, lambda s: s.quantile(.75)-s.quantile(.25)))
                   .reset_index())

    coverage = (df.groupby("_fold")
                .agg(records=(tds_col, "size"), distinct_dates=("_date", "nunique"),
                     first_timestamp=("_ts", "min"), last_timestamp=("_ts", "max"))
                .reset_index().rename(columns={"_fold": "fold"}))
    coverage["first_timestamp"] = coverage["first_timestamp"].astype(str)
    coverage["last_timestamp"] = coverage["last_timestamp"].astype(str)

    daily.to_csv(OUT_DIR / "01_daily_regime_summary.csv", index=False)
    hourly.to_csv(OUT_DIR / "02_hourly_regime_summary.csv", index=False)
    daynight.to_csv(OUT_DIR / "03_day_night_comparison_by_fold.csv", index=False)
    correlations.to_csv(OUT_DIR / "04_correlations_by_fold_and_timeband.csv", index=False)
    variability.to_csv(OUT_DIR / "05_sensor_variability_by_fold_and_timeband.csv", index=False)
    coverage.to_csv(OUT_DIR / "06_fold_time_coverage.csv", index=False)

    report = [
        "BOREMAN STEP 35 - ENVIRONMENTAL REGIME AND TIME-OF-DAY INVESTIGATION",
        "=" * 78,
        f"Input rows read: {len(pd.read_csv(INPUT, usecols=[ts_col]))}",
        f"Unique primary TRAIN observations analysed: {len(df)}",
        f"Duplicate rows removed for analysis only: {deduplicated}",
        f"Distinct TRAIN dates available: {len(dates)}",
        "",
        "FOLD WINDOWS (same Step 32 date positions)",
    ]
    for fold, fold_dates in windows:
        report.append(f"- {fold}: {', '.join(fold_dates)}")
    report += [
        "",
        "METHOD",
        "- Uses Step 25B engineered dataset, restricted to primary TRAIN observations where labels are available.",
        "- Reproduces Step 32 fold validation windows using the 6th-11th distinct TRAIN dates.",
        "- Converts canonical timestamps to Africa/Windhoek for all hour/day grouping.",
        "- DAY is fixed Namibia clock time 06:00-18:00; NIGHT is 18:00-06:00.",
        "- Each original observation is counted once; no repeated forecast horizons.",
        "- Summaries are record-weighted, not time-weighted.",
        "- No interpolation, imputation, model training or deletion of source data.",
        "",
        "INTERPRETATION LIMITATIONS",
        "- Day/night bands are fixed clock windows, not astronomical sunrise/sunset.",
        "- Differences are descriptive and do not establish causes.",
        "- Sensor readings alone cannot establish water safety or confirmed sensor faults.",
        "- Unequal sampling and missing time periods may affect group comparisons.",
        "",
        "OUTPUTS",
        "- 01_daily_regime_summary.csv",
        "- 02_hourly_regime_summary.csv",
        "- 03_day_night_comparison_by_fold.csv",
        "- 04_correlations_by_fold_and_timeband.csv",
        "- 05_sensor_variability_by_fold_and_timeband.csv",
        "- 06_fold_time_coverage.csv",
    ]
    (OUT_DIR / "07_step35_environmental_regime_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("BOREMAN STEP 35 - ENVIRONMENTAL REGIME AND TIME-OF-DAY INVESTIGATION")
    print("=" * 72)
    print(f"Unique primary TRAIN observations analysed: {len(df):,}")
    print(f"Duplicate rows removed for analysis only: {deduplicated:,}")
    print(f"Fold rows analysed: {df['_fold'].value_counts().sort_index().to_dict()}")
    print(f"Output folder: {OUT_DIR.resolve()}")
    print("No model training; validation/test periods outside the TRAIN fold windows were not used.")

if __name__ == "__main__":
    OUT_DIR = OUT
    main()
