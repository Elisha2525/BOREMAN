from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
INPUT = ROOT / "outputs" / "step25B_ai_feature_engineering" / "01_all_records_with_engineered_features.csv"
OUT = ROOT / "outputs" / "step36_sensor_response_confounding"

EXPECTED_TRAIN_DATES = [
    "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
    "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11",
    "2026-10-12", "2026-10-13", "2026-10-14"
]

def safe_corr(a, b):
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan, len(pair)
    return pair.iloc[:, 0].corr(pair.iloc[:, 1]), len(pair)

def residualize(series, design):
    mask = series.notna() & np.isfinite(series.to_numpy(dtype=float))
    if mask.sum() <= design.shape[1] + 2:
        return pd.Series(np.nan, index=series.index)
    y = series.loc[mask].to_numpy(dtype=float)
    x = design.loc[mask].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[mask] = residuals
    return out

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)
    required = [
        "source_row_number", "timestamp_namibia", "tds", "humidity",
        "Temperature", "modelling_readiness_group", "proposed_temporal_split",
        "sampling_interval_minutes"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    input_rows = len(df)
    split = df["proposed_temporal_split"].astype(str).str.strip().str.upper()
    group = df["modelling_readiness_group"].astype(str).str.strip().str.upper()
    d = df[split.eq("TRAIN") & group.str.contains("PRIMARY", na=False)].copy()
    if d.empty:
        raise ValueError("No rows matched TRAIN + PRIMARY.")

    d["_ts"] = pd.to_datetime(d["timestamp_namibia"], errors="coerce", utc=True)
    d = d[d["_ts"].notna()].copy()
    d["_ts"] = d["_ts"].dt.tz_convert("Africa/Windhoek")
    for c in ["tds", "humidity", "Temperature", "sampling_interval_minutes"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.drop_duplicates(subset=["source_row_number"], keep="first").copy()
    d["_date"] = d["_ts"].dt.strftime("%Y-%m-%d")
    d["_hour"] = d["_ts"].dt.hour
    d["_time_band"] = np.where(d["_hour"].between(6, 17), "DAY_06_18", "NIGHT_18_06")
    d["_fold"] = d["_date"].map({
        "2026-10-09": "FOLD_1", "2026-10-10": "FOLD_1",
        "2026-10-11": "FOLD_2", "2026-10-12": "FOLD_2",
        "2026-10-13": "FOLD_3", "2026-10-14": "FOLD_3"
    })

    actual_dates = sorted(d["_date"].dropna().unique().tolist())
    if actual_dates != EXPECTED_TRAIN_DATES:
        raise ValueError(
            "TRAIN date window mismatch. Expected the 11 Step 32 dates "
            f"{EXPECTED_TRAIN_DATES}, found {actual_dates}. Stopping."
        )

    # 1) Raw relationships by fold, day/night and overall.
    corr_rows = []
    groups = [("ALL_TRAIN", d)] + [
        (f"{fold}_{band}", g)
        for (fold, band), g in d[d["_fold"].notna()].groupby(["_fold", "_time_band"])
    ] + [
        (fold, g) for fold, g in d[d["_fold"].notna()].groupby("_fold")
    ]
    for label, g in groups:
        for xcol, xname in [("Temperature", "temperature"), ("humidity", "humidity")]:
            r, n = safe_corr(g["tds"], g[xcol])
            corr_rows.append({
                "group": label, "relationship": f"TDS_vs_{xname}",
                "records": n, "pearson_r": r
            })
    raw_corr = pd.DataFrame(corr_rows)

    # 2) Adjusted association: remove date and hour-of-day mean effects from each variable.
    # This is a descriptive residual correlation, not a causal or calibration model.
    design = pd.DataFrame({"intercept": np.ones(len(d))}, index=d.index)
    date_dummies = pd.get_dummies(d["_date"], prefix="date", drop_first=True, dtype=float)
    hour_dummies = pd.get_dummies(d["_hour"].astype("Int64"), prefix="hour", drop_first=True, dtype=float)
    design = pd.concat([design, date_dummies, hour_dummies], axis=1).astype(float)

    residuals = {}
    for col in ["tds", "Temperature", "humidity"]:
        residuals[col] = residualize(d[col], design)
    adjusted_rows = []
    for xcol, xname in [("Temperature", "temperature"), ("humidity", "humidity")]:
        r, n = safe_corr(residuals["tds"], residuals[xcol])
        adjusted_rows.append({
            "relationship": f"TDS_vs_{xname}",
            "records": n,
            "raw_pearson_all_train": safe_corr(d["tds"], d[xcol])[0],
            "date_and_hour_adjusted_residual_correlation": r,
            "adjustment": "Removed date and hour-of-day group means"
        })
    adjusted = pd.DataFrame(adjusted_rows)

    # 3) First-difference relationships for consecutive readings <=5 minutes apart.
    d = d.sort_values("_ts").copy()
    d["_interval_actual_min"] = d.groupby("_date")["_ts"].diff().dt.total_seconds() / 60
    for col in ["tds", "humidity", "Temperature"]:
        d[f"delta_{col}"] = d.groupby("_date")[col].diff()
    valid_pairs = d["_interval_actual_min"].between(0, 5, inclusive="right")
    change_rows = []
    for xcol, xname in [("delta_Temperature", "temperature_change"), ("delta_humidity", "humidity_change")]:
        r, n = safe_corr(d.loc[valid_pairs, "delta_tds"], d.loc[valid_pairs, xcol])
        change_rows.append({
            "relationship": f"TDS_change_vs_{xname}",
            "consecutive_pairs": n,
            "pearson_r": r,
            "maximum_interval_minutes": 5
        })
    change_corr = pd.DataFrame(change_rows)

    # 4) Sampling coverage and intervals.
    interval_summary = (
        d.groupby(["_date"])
        .agg(
            records=("source_row_number", "nunique"),
            median_interval_minutes=("_interval_actual_min", "median"),
            mean_interval_minutes=("_interval_actual_min", "mean"),
            p95_interval_minutes=("_interval_actual_min", lambda s: s.quantile(.95)),
            intervals_over_5min=("_interval_actual_min", lambda s: int((s > 5).sum())),
            intervals_over_60min=("_interval_actual_min", lambda s: int((s > 60).sum()))
        ).reset_index()
    )

    # 5) Fold/time-band descriptive summaries, with record counts.
    summary = (
        d[d["_fold"].notna()]
        .groupby(["_fold", "_time_band"])
        .agg(
            records=("source_row_number", "nunique"),
            tds_mean=("tds", "mean"), tds_median=("tds", "median"), tds_std=("tds", "std"),
            humidity_mean=("humidity", "mean"), humidity_std=("humidity", "std"),
            temperature_mean=("Temperature", "mean"), temperature_std=("Temperature", "std"),
            median_sampling_interval_minutes=("_interval_actual_min", "median")
        ).reset_index()
    )

    # 6) Explicit instrumentation/metadata checklist. Data values are not substitutes.
    checklist = pd.DataFrame([
        ["TDS sensor model and datasheet", "NOT PROVIDED IN DATASET", "Needed to establish temperature compensation method and range"],
        ["Temperature compensation enabled/disabled", "NOT PROVIDED IN DATASET", "Cannot infer compensation setting from correlations"],
        ["TDS sensor temperature coefficient / calibration setting", "NOT PROVIDED IN DATASET", "Needed to interpret compensated TDS/EC output"],
        ["SHT20 physical placement", "NOT PROVIDED IN DATASET", "Cannot tell whether humidity/temperature represent ambient air, enclosure or another environment"],
        ["TDS probe placement and immersion depth", "NOT PROVIDED IN DATASET", "Needed to assess installation and measurement conditions"],
        ["Reference EC/TDS meter readings", "NOT PROVIDED IN DATASET", "Needed for independent agreement / calibration assessment"],
        ["Water temperature measured at probe", "NOT CONFIRMED", "The Temperature column is from SHT20 context; do not assume it is water temperature"],
        ["Calibration and maintenance log", "NOT PROVIDED IN DATASET", "Needed to evaluate drift or maintenance-related changes"],
    ], columns=["required_metadata", "availability_in_current_dataset", "why_it_matters"])

    OUT.mkdir(parents=True, exist_ok=True)
    raw_corr.to_csv(OUT / "01_raw_sensor_correlations_by_fold_and_timeband.csv", index=False)
    adjusted.to_csv(OUT / "02_date_hour_adjusted_associations.csv", index=False)
    change_corr.to_csv(OUT / "03_consecutive_change_correlations.csv", index=False)
    interval_summary.to_csv(OUT / "04_sampling_interval_by_train_date.csv", index=False)
    summary.to_csv(OUT / "05_sensor_summary_by_fold_and_timeband.csv", index=False)
    checklist.to_csv(OUT / "06_sensor_metadata_and_validation_gaps.csv", index=False)

    report = [
        "BOREMAN STEP 36 - SENSOR RESPONSE AND ENVIRONMENTAL CONFOUNDING INVESTIGATION",
        "=" * 82,
        f"Input rows: {input_rows}",
        f"Unique primary TRAIN observations analysed: {len(d)}",
        f"TRAIN dates verified: {', '.join(actual_dates)}",
        "",
        "METHOD",
        "- Strictly filtered proposed_temporal_split == TRAIN.",
        "- Strictly filtered modelling_readiness_group containing PRIMARY.",
        "- Used canonical timestamp_namibia converted to Africa/Windhoek.",
        "- Used each unique source_row_number once.",
        "- Calculated raw TDS-temperature and TDS-humidity correlations by fold and fixed day/night band.",
        "- Calculated residual correlations after removing date and hour-of-day group means.",
        "- Calculated consecutive first-difference correlations only within the same date and intervals <=5 minutes.",
        "- Summarised actual sampling intervals and recorded metadata gaps.",
        "- No model training, imputation, interpolation, calibration claims or source-data modification.",
        "",
        "INTERPRETATION LIMITATIONS",
        "- Correlation and adjusted correlation do not establish causation.",
        "- Date/hour adjustment removes group means only; it does not control every environmental or operational factor.",
        "- SHT20 temperature must not be treated as water temperature without confirmed placement.",
        "- The dataset does not establish whether TDS temperature compensation is enabled or how it is configured.",
        "- TDS readings alone cannot establish water safety or laboratory accuracy.",
        "- First differences are sensitive to measurement noise and sampling intervals.",
        "",
        "OUTPUTS",
        "- 01_raw_sensor_correlations_by_fold_and_timeband.csv",
        "- 02_date_hour_adjusted_associations.csv",
        "- 03_consecutive_change_correlations.csv",
        "- 04_sampling_interval_by_train_date.csv",
        "- 05_sensor_summary_by_fold_and_timeband.csv",
        "- 06_sensor_metadata_and_validation_gaps.csv",
    ]
    (OUT / "07_step36_sensor_response_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("BOREMAN STEP 36 - SENSOR RESPONSE AND ENVIRONMENTAL CONFOUNDING INVESTIGATION")
    print("=" * 78)
    print(f"Input rows: {input_rows:,}")
    print(f"Unique primary TRAIN observations analysed: {len(d):,}")
    print(f"TRAIN dates verified: {len(actual_dates)}")
    print(f"Fold/time-band groups: {len(summary)}")
    print(f"Output folder: {OUT.resolve()}")
    print("No model training; VALIDATION and TEST observations were not used.")

if __name__ == "__main__":
    main()
