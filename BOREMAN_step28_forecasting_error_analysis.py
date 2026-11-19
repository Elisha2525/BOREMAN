#!/usr/bin/env python
"""
BOREMAN STEP 28 - FORECASTING ERROR ANALYSIS

Reads Step 27 predictions and supervised samples.
Does not retrain models or modify source/Step 27 outputs.
All timestamps are interpreted in Namibia local time (Africa/Windhoek).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "outputs" / "step27_time_series_forecasting"
OUTPUT_DIR = ROOT / "outputs" / "step28_forecasting_error_analysis"

PREDICTIONS_FILE = INPUT_DIR / "02_forecast_predictions.csv"
SUPERVISED_FILE = INPUT_DIR / "01_forecast_supervised_dataset.csv"

TARGETS = ["tds", "humidity", "Temperature"]
EXPECTED_HORIZONS = [5, 15, 30, 60]


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Place this script in the project root and ensure Step 27 outputs exist."
        )


def metric_row(frame, group_values):
    result = dict(group_values)
    result["sample_count"] = int(len(frame))
    for model, col in [
        ("Persistence", "persistence_absolute_error"),
        ("RandomForest", "random_forest_absolute_error"),
    ]:
        errors = pd.to_numeric(frame[col], errors="coerce").dropna()
        result[f"{model.lower()}_mae"] = float(errors.mean()) if len(errors) else np.nan
        result[f"{model.lower()}_rmse"] = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan
        result[f"{model.lower()}_median_absolute_error"] = float(errors.median()) if len(errors) else np.nan
        result[f"{model.lower()}_p90_absolute_error"] = float(errors.quantile(0.90)) if len(errors) else np.nan
    p = result.get("persistence_mae", np.nan)
    r = result.get("randomforest_mae", np.nan)
    result["rf_mae_change_vs_persistence_pct"] = (
        float((r - p) / p * 100.0) if pd.notna(p) and p != 0 and pd.notna(r) else np.nan
    )
    result["better_model_by_mae"] = (
        "Persistence" if pd.notna(p) and pd.notna(r) and p < r
        else "RandomForest" if pd.notna(p) and pd.notna(r) and r < p
        else "Tie/Unavailable"
    )
    return result


def grouped_metrics(df, group_cols):
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(metric_row(group, dict(zip(group_cols, keys))))
    return pd.DataFrame(rows)


def main():
    print("=" * 68)
    print("BOREMAN STEP 28 - FORECASTING ERROR ANALYSIS")
    print("=" * 68)

    require_file(PREDICTIONS_FILE)
    require_file(SUPERVISED_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nReading Step 27 prediction and supervised datasets...")
    pred = pd.read_csv(PREDICTIONS_FILE)
    supervised = pd.read_csv(SUPERVISED_FILE)

    required_pred = {
        "origin_source_row_number", "target_source_row_number",
        "origin_timestamp_namibia", "target_timestamp_namibia",
        "requested_horizon_minutes", "actual_horizon_minutes", "split",
        "target", "actual_value", "persistence_prediction",
        "random_forest_prediction", "persistence_absolute_error",
        "random_forest_absolute_error"
    }
    missing = sorted(required_pred - set(pred.columns))
    if missing:
        raise ValueError(f"Step 27 predictions missing required columns: {missing}")

    # Parse ISO timestamps while retaining Namibia local clock values.
    pred["origin_dt"] = pd.to_datetime(pred["origin_timestamp_namibia"], errors="coerce", utc=True).dt.tz_convert("Africa/Windhoek")
    pred["target_dt"] = pd.to_datetime(pred["target_timestamp_namibia"], errors="coerce", utc=True).dt.tz_convert("Africa/Windhoek")
    pred["origin_hour"] = pred["origin_dt"].dt.hour
    pred["origin_date_namibia"] = pred["origin_dt"].dt.strftime("%Y-%m-%d")
    pred["time_of_day"] = np.where(
        (pred["origin_hour"] >= 6) & (pred["origin_hour"] < 18), "DAY_06_00_TO_18_00", "NIGHT_18_00_TO_06_00"
    )
    pred["actual_change_from_persistence"] = pred["actual_value"] - pred["persistence_prediction"]
    pred["absolute_actual_change"] = pred["actual_change_from_persistence"].abs()
    pred["rf_signed_error"] = pred["random_forest_prediction"] - pred["actual_value"]
    pred["persistence_signed_error"] = pred["persistence_prediction"] - pred["actual_value"]
    pred["rf_minus_persistence_abs_error"] = (
        pred["random_forest_absolute_error"] - pred["persistence_absolute_error"]
    )

    # 1. Overall comparison, including all splits.
    overall = grouped_metrics(pred, ["requested_horizon_minutes", "target", "split"])
    overall.to_csv(OUTPUT_DIR / "01_overall_error_comparison.csv", index=False)

    # 2. Hour-of-day analysis using Namibia local origin time.
    hourly = grouped_metrics(pred.dropna(subset=["origin_hour"]), ["requested_horizon_minutes", "target", "split", "origin_hour"])
    hourly.to_csv(OUTPUT_DIR / "02_hourly_error_analysis_namibia.csv", index=False)

    # 3. Fixed clock day/night analysis (not astronomical sunrise/sunset).
    daynight = grouped_metrics(pred, ["requested_horizon_minutes", "target", "split", "time_of_day"])
    daynight.to_csv(OUTPUT_DIR / "03_day_night_error_comparison.csv", index=False)

    # 4. Error by magnitude of observed change from persistence (origin reading).
    change_rows = []
    for (horizon, target, split), group in pred.groupby(
        ["requested_horizon_minutes", "target", "split"], sort=True
    ):
        g = group.copy()
        # Quantile bands are calculated independently within each target/horizon/split.
        try:
            g["observed_change_band"] = pd.qcut(
                g["absolute_actual_change"], q=4,
                labels=["Q1_smallest_change", "Q2", "Q3", "Q4_largest_change"],
                duplicates="drop"
            ).astype(str)
        except ValueError:
            g["observed_change_band"] = "UNCLASSIFIED"
        part = grouped_metrics(g, ["observed_change_band"])
        part.insert(0, "split", split)
        part.insert(0, "target", target)
        part.insert(0, "requested_horizon_minutes", horizon)
        part["mean_absolute_observed_change"] = g.groupby("observed_change_band", observed=True)["absolute_actual_change"].mean().reindex(part["observed_change_band"]).to_numpy()
        change_rows.append(part)
    change_analysis = pd.concat(change_rows, ignore_index=True) if change_rows else pd.DataFrame()
    change_analysis.to_csv(OUTPUT_DIR / "04_error_by_observed_change_magnitude.csv", index=False)

    # 5. Largest RF errors and cases where RF is much worse than persistence.
    top = pred.sort_values(
        ["random_forest_absolute_error", "rf_minus_persistence_abs_error"],
        ascending=[False, False]
    ).head(500).copy()
    keep = [
        "origin_source_row_number", "target_source_row_number",
        "origin_timestamp_namibia", "target_timestamp_namibia",
        "requested_horizon_minutes", "actual_horizon_minutes", "split",
        "target", "actual_value", "persistence_prediction",
        "random_forest_prediction", "actual_change_from_persistence",
        "persistence_absolute_error", "random_forest_absolute_error",
        "rf_minus_persistence_abs_error", "origin_hour", "time_of_day"
    ]
    top[keep].to_csv(OUTPUT_DIR / "05_top_500_random_forest_error_records.csv", index=False)

    # 6. Compact report with defensible interpretation.
    test = overall[overall["split"].astype(str).str.upper() == "TEST"].copy()
    comparisons = len(test)
    rf_wins = int((test["better_model_by_mae"] == "RandomForest").sum())
    persistence_wins = int((test["better_model_by_mae"] == "Persistence").sum())
    lines = [
        "BOREMAN STEP 28 - FORECASTING ERROR ANALYSIS REPORT",
        "=" * 68,
        f"Prediction rows analysed: {len(pred):,}",
        f"Targets: {sorted(pred['target'].dropna().astype(str).unique().tolist())}",
        f"Horizons: {sorted(pred['requested_horizon_minutes'].dropna().astype(int).unique().tolist())}",
        f"Splits: {sorted(pred['split'].dropna().astype(str).unique().tolist())}",
        "",
        "METHOD",
        "- Reuses Step 27 predictions; no model retraining.",
        "- Error summaries include MAE, RMSE, median absolute error and 90th percentile absolute error.",
        "- Time-of-day uses Namibia local origin timestamps.",
        "- DAY is fixed 06:00-18:00; NIGHT is 18:00-06:00, not astronomical daylight.",
        "- Observed-change bands are within-group quartiles of absolute change from the persistence prediction.",
        "",
        "TEST-SET MAE COMPARISON",
        f"Comparisons (horizon x target): {comparisons}",
        f"Persistence better: {persistence_wins}",
        f"Random Forest better: {rf_wins}",
        "",
    ]
    for _, row in test.sort_values(["requested_horizon_minutes", "target"]).iterrows():
        lines.append(
            f"{int(row['requested_horizon_minutes'])} min | {row['target']} | "
            f"n={int(row['sample_count'])} | "
            f"Persistence MAE={row['persistence_mae']:.6f} | "
            f"RandomForest MAE={row['randomforest_mae']:.6f} | "
            f"RF MAE change={row['rf_mae_change_vs_persistence_pct']:.2f}% | "
            f"better={row['better_model_by_mae']}"
        )
    lines += [
        "",
        "INTERPRETATION LIMITATIONS",
        "- Error patterns describe this dataset and holdout period; they do not prove causal effects.",
        "- Hourly and day/night groups can have different sample counts.",
        "- Quartile change bands are descriptive and are not operational thresholds.",
        "- A large error is not automatically a sensor fault or water-quality event.",
        "- Forecasting sensor readings does not establish water safety.",
        "- Results are an initial chronological holdout evaluation, not external validation.",
        "",
        "SOURCE PRESERVATION",
        "- Step 27 inputs and outputs were read only.",
        "- No original observations were deleted, imputed, or overwritten.",
    ]
    (OUTPUT_DIR / "06_step28_error_analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nStep 28 output summary:")
    print(f"Prediction rows analysed: {len(pred):,}")
    print(f"Test comparisons: {comparisons}")
    print(f"Persistence wins by MAE: {persistence_wins}")
    print(f"Random Forest wins by MAE: {rf_wins}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nSTEP 28 COMPLETED")
    print("Review generated outputs before proceeding.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
