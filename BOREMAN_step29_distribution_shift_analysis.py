"""BOREMAN STEP 29 - TRAINING / VALIDATION / TEST DISTRIBUTION-SHIFT ANALYSIS

Read-only diagnostic. Does not train models or modify prior outputs.
Run from the BOREMAN project root:
    python .\\BOREMAN_step29_distribution_shift_analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STEP25B_CANDIDATES = [
    ROOT / "outputs" / "step25B_ai_feature_engineering" / "01_all_records_with_engineered_features.csv",
    ROOT / "outputs" / "step25B_feature_engineering" / "01_all_records_with_engineered_features.csv",
    ROOT / "01_all_records_with_engineered_features.csv",
]
STEP27_DIR_CANDIDATES = [ROOT / "outputs" / "step27_time_series_forecasting", ROOT]
OUT_DIR = ROOT / "outputs" / "step29_distribution_shift_analysis"
SENSORS = ["tds", "humidity", "Temperature"]
SPLIT_ORDER = ["TRAIN", "VALIDATION", "TEST"]


def locate(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{checked}")


def numeric_summary(frame, columns, group_cols, metric_name):
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for col in columns:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({**base, "variable": col, "n": int(values.size),
                         "mean": values.mean(), "std": values.std(),
                         "median": values.median(), "q05": values.quantile(.05),
                         "q25": values.quantile(.25), "q75": values.quantile(.75),
                         "q95": values.quantile(.95), "min": values.min(), "max": values.max(),
                         "analysis_type": metric_name})
    return pd.DataFrame(rows)


def safe_read(path):
    return pd.read_csv(path, low_memory=False)


def main():
    print("=" * 68)
    print("BOREMAN STEP 29 - DISTRIBUTION-SHIFT ANALYSIS")
    print("=" * 68)

    step25b_path = locate(STEP25B_CANDIDATES, "Step 25B engineered dataset")
    step27_dir = next((p for p in STEP27_DIR_CANDIDATES if (p / "01_forecast_supervised_dataset.csv").exists()), None)
    if step27_dir is None:
        raise FileNotFoundError("Could not find outputs\\step27_time_series_forecasting\\01_forecast_supervised_dataset.csv")
    supervised_path = step27_dir / "01_forecast_supervised_dataset.csv"
    predictions_path = step27_dir / "02_forecast_predictions.csv"

    print(f"Reading Step 25B: {step25b_path}")
    base = safe_read(step25b_path)
    print(f"Reading Step 27 supervised dataset: {supervised_path}")
    supervised = safe_read(supervised_path)
    print(f"Reading Step 27 predictions: {predictions_path}")
    predictions = safe_read(predictions_path)

    required_base = {"source_row_number", "proposed_temporal_split", "modelling_readiness_group", *SENSORS}
    required_sup = {"origin_source_row_number", "proposed_temporal_split", "requested_horizon_minutes", *[f"{s}_current" for s in SENSORS]}
    required_pred = {"split", "target", "requested_horizon_minutes", "actual_value", "persistence_prediction", "random_forest_prediction", "random_forest_absolute_error", "origin_source_row_number"}
    for name, frame, required in [("Step25B", base, required_base), ("Step27 supervised", supervised, required_sup), ("Step27 predictions", predictions, required_pred)]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    # Limit distribution comparisons to the same primary-candidate cohort used by Step 27.
    primary = base[base["modelling_readiness_group"].astype(str).str.upper().str.contains("PRIMARY", na=False)].copy()
    if primary.empty:
        raise ValueError("No PRIMARY records found in Step 25B dataset.")
    primary["split"] = primary["proposed_temporal_split"].astype(str).str.upper()
    primary = primary[primary["split"].isin(SPLIT_ORDER)].copy()

    # 1. Raw sensor distributions by chronological split.
    sensor_cols = [c for c in SENSORS if c in primary.columns]
    sensor_dist = numeric_summary(primary, sensor_cols, ["split"], "sensor_value_distribution")
    sensor_dist.to_csv(OUT_DIR / "01_sensor_distribution_by_split.csv", index=False)

    # 2. Engineered-feature distribution shift. Include current sensor values and
    # selected history / variability / time features; exclude row IDs and segment IDs.
    candidate_features = []
    for sensor in SENSORS:
        candidate_features += [
            f"{sensor}_current", f"{sensor}_change_from_previous", f"{sensor}_rate_per_minute",
            f"{sensor}_lag_1", f"{sensor}_lag_3", f"{sensor}_lag_5",
            f"{sensor}_rolling_mean_previous_5", f"{sensor}_rolling_std_previous_5",
            f"{sensor}_deviation_from_previous_5_mean",
        ]
    candidate_features += ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "sampling_interval_minutes"]
    feature_cols = [c for c in candidate_features if c in primary.columns and pd.api.types.is_numeric_dtype(primary[c])]
    feature_shift = numeric_summary(primary, feature_cols, ["split"], "engineered_feature_distribution")
    # Add TRAIN-relative standardized mean difference and median shift for each feature.
    if not feature_shift.empty:
        train_stats = feature_shift[feature_shift["split"] == "TRAIN"].set_index("variable")
        train_stats = train_stats[~train_stats.index.duplicated(keep="first")]
        feature_shift["train_mean"] = feature_shift["variable"].map(train_stats["mean"])
        feature_shift["train_std"] = feature_shift["variable"].map(train_stats["std"])
        feature_shift["standardized_mean_difference_vs_train"] = (
            feature_shift["mean"] - feature_shift["train_mean"]
        ) / feature_shift["train_std"].replace(0, np.nan)
        feature_shift["median_difference_vs_train"] = feature_shift["median"] - feature_shift["variable"].map(train_stats["median"])
    feature_shift.to_csv(OUT_DIR / "02_feature_distribution_shift.csv", index=False)

    # 3. Random Forest signed error (prediction - actual) to identify systematic bias.
    pred = predictions.copy()
    pred["split"] = pred["split"].astype(str).str.upper()
    pred["rf_signed_error"] = pd.to_numeric(pred["random_forest_prediction"], errors="coerce") - pd.to_numeric(pred["actual_value"], errors="coerce")
    pred["persistence_signed_error"] = pd.to_numeric(pred["persistence_prediction"], errors="coerce") - pd.to_numeric(pred["actual_value"], errors="coerce")
    bias_rows = []
    for (split, horizon, target), g in pred.groupby(["split", "requested_horizon_minutes", "target"]):
        e = g["rf_signed_error"].dropna()
        pe = g["persistence_signed_error"].dropna()
        if e.empty:
            continue
        bias_rows.append({"split": split, "horizon_minutes": int(horizon), "target": target, "n": len(e),
                          "rf_mean_signed_error": e.mean(), "rf_median_signed_error": e.median(),
                          "rf_overprediction_pct": (e > 0).mean() * 100,
                          "rf_underprediction_pct": (e < 0).mean() * 100,
                          "rf_mae": e.abs().mean(), "rf_rmse": np.sqrt(np.mean(np.square(e))),
                          "persistence_mean_signed_error": pe.mean(), "persistence_mae": pe.abs().mean()})
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(OUT_DIR / "03_random_forest_bias_by_horizon.csv", index=False)

    # 4. Check whether RF predictions fall outside the TRAIN target-value range.
    # This is a range diagnostic, not a statement that values are physically invalid.
    range_rows = []
    for (horizon, target), train_g in pred[pred["split"] == "TRAIN"].groupby(["requested_horizon_minutes", "target"]):
        train_vals = pd.to_numeric(train_g["actual_value"], errors="coerce").dropna()
        if train_vals.empty:
            continue
        lo, hi = train_vals.min(), train_vals.max()
        for split in ["VALIDATION", "TEST"]:
            g = pred[(pred["split"] == split) & (pred["requested_horizon_minutes"] == horizon) & (pred["target"] == target)]
            vals = pd.to_numeric(g["random_forest_prediction"], errors="coerce").dropna()
            actual = pd.to_numeric(g["actual_value"], errors="coerce").dropna()
            if vals.empty:
                continue
            range_rows.append({"split": split, "horizon_minutes": int(horizon), "target": target,
                               "train_actual_min": lo, "train_actual_max": hi,
                               "rf_prediction_min": vals.min(), "rf_prediction_max": vals.max(),
                               "rf_predictions_below_train_min": int((vals < lo).sum()),
                               "rf_predictions_above_train_max": int((vals > hi).sum()),
                               "rf_predictions_outside_train_range_pct": ((vals < lo) | (vals > hi)).mean() * 100,
                               "actual_min": actual.min(), "actual_max": actual.max(),
                               "actual_values_outside_train_range": int(((actual < lo) | (actual > hi)).sum())})
    pd.DataFrame(range_rows).to_csv(OUT_DIR / "04_prediction_and_target_range_check.csv", index=False)

    # 5. Readiness-group context for each split, using the full Step 25B primary cohort.
    context_rows = []
    for (split, group_name), g in primary.groupby(["split", "modelling_readiness_group"], dropna=False):
        row = {"split": split, "modelling_readiness_group": group_name, "record_count": len(g)}
        for sensor in sensor_cols:
            vals = pd.to_numeric(g[sensor], errors="coerce").dropna()
            row[f"{sensor}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{sensor}_median"] = vals.median() if len(vals) else np.nan
        context_rows.append(row)
    pd.DataFrame(context_rows).to_csv(OUT_DIR / "05_split_readiness_context.csv", index=False)

    # Report
    lines = [
        "BOREMAN STEP 29 - DISTRIBUTION-SHIFT ANALYSIS REPORT",
        "=" * 68,
        f"Step 25B source rows: {len(base):,}",
        f"Primary-candidate rows in chronological splits: {len(primary):,}",
        f"Step 27 supervised samples: {len(supervised):,}",
        f"Step 27 prediction rows: {len(predictions):,}",
        f"Feature variables compared: {len(feature_cols)}",
        "",
        "METHOD",
        "- Compares TRAIN, VALIDATION and TEST within the primary-candidate cohort.",
        "- Reports sensor and engineered-feature distributions, including TRAIN-relative standardized mean differences.",
        "- Signed RF error is prediction minus actual; positive means overprediction.",
        "- Prediction range checks compare RF predictions and actual targets with the TRAIN actual-value range.",
        "- Range exceedance is a diagnostic only; it does not establish physical invalidity.",
        "- No model retraining, imputation, source-row deletion or prior-output modification occurs.",
        "",
        "RANDOM FOREST TEST BIAS SUMMARY",
    ]
    test_bias = bias[bias["split"] == "TEST"] if not bias.empty else pd.DataFrame()
    if test_bias.empty:
        lines.append("No TEST bias rows available.")
    else:
        for _, r in test_bias.iterrows():
            direction = "overpredicts on average" if r["rf_mean_signed_error"] > 0 else "underpredicts on average" if r["rf_mean_signed_error"] < 0 else "has near-zero mean bias"
            lines.append(f"{int(r['horizon_minutes'])} min | {r['target']} | n={int(r['n'])} | mean signed error={r['rf_mean_signed_error']:.4f} | {direction} | RF MAE={r['rf_mae']:.4f} | Persistence MAE={r['persistence_mae']:.4f}")
    lines += ["", "INTERPRETATION CAUTIONS",
              "- Distribution differences do not prove the cause of model underperformance.",
              "- Standardized mean differences are descriptive and can be unstable for near-zero TRAIN standard deviation.",
              "- The chronological TEST period is one holdout period, not external validation.",
              "- Sensor-reading forecasts do not establish water safety.",
              "", "SOURCE PRESERVATION",
              "- Step 25B and Step 27 files were read only.",
              "- No source observations were deleted, imputed or overwritten."]
    (OUT_DIR / "06_step29_distribution_shift_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nSTEP 29 OUTPUT SUMMARY")
    print(f"Primary-candidate records analysed: {len(primary):,}")
    print(f"Engineered features compared: {len(feature_cols)}")
    print(f"Prediction rows analysed: {len(pred):,}")
    print(f"Output folder: {OUT_DIR}")
    print("\nSTEP 29 COMPLETED")
    print("Review generated outputs before proceeding.")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
