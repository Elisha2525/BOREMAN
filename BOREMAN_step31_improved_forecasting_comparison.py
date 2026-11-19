"""
BOREMAN STEP 31 - IMPROVED FORECASTING MODEL COMPARISON

Compares existing persistence and Random Forest results with:
1. Linear drift baseline: current value + recent rate * forecast horizon.
2. Ridge regression: separate model per target and horizon, with alpha selected
   on VALIDATION only.

TRAIN is used for fitting, VALIDATION for alpha selection, TEST for final
holdout evaluation. Existing Step 25B and Step 27 files are read-only.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "outputs"
FEATURE_FILE = INPUT_DIR / "step25B_ai_feature_engineering" / "01_all_records_with_engineered_features.csv"
SUPERVISED_FILE = INPUT_DIR / "step27_time_series_forecasting" / "01_forecast_supervised_dataset.csv"
PREDICTIONS_FILE = INPUT_DIR / "step27_time_series_forecasting" / "02_forecast_predictions.csv"
OUT_DIR = INPUT_DIR / "step31_improved_forecasting_comparison"

HORIZONS = [5, 15, 30, 60]
TARGETS = ["tds", "humidity", "Temperature"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
FEATURES = [
    "tds_current", "humidity_current", "Temperature_current",
    "tds_change_from_previous", "humidity_change_from_previous",
    "Temperature_change_from_previous",
    "tds_rate_per_minute", "humidity_rate_per_minute",
    "Temperature_rate_per_minute",
    "tds_lag_1", "tds_lag_3", "tds_lag_5",
    "humidity_lag_1", "humidity_lag_3", "humidity_lag_5",
    "Temperature_lag_1", "Temperature_lag_3", "Temperature_lag_5",
    "tds_rolling_mean_previous_5", "humidity_rolling_mean_previous_5",
    "Temperature_rolling_mean_previous_5",
    "tds_rolling_std_previous_5", "humidity_rolling_std_previous_5",
    "Temperature_rolling_std_previous_5",
    "tds_deviation_from_previous_5_mean",
    "humidity_deviation_from_previous_5_mean",
    "Temperature_deviation_from_previous_5_mean",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "sampling_interval_minutes",
]
RATE_COLUMN = {
    "tds": "tds_rate_per_minute",
    "humidity": "humidity_rate_per_minute",
    "Temperature": "Temperature_rate_per_minute",
}
CURRENT_COLUMN = {
    "tds": "tds_current",
    "humidity": "humidity_current",
    "Temperature": "Temperature_current",
}

def metric_row(y_true, y_pred, split, horizon, target, model):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "split": split,
        "horizon_minutes": int(horizon),
        "target": target,
        "model": model,
        "sample_count": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "mean_signed_error": float(np.mean(y_pred - y_true)),
    }

def main():
    print("=" * 68)
    print("BOREMAN STEP 31 - IMPROVED FORECASTING MODEL COMPARISON")
    print("=" * 68)
    for path in [SUPERVISED_FILE, PREDICTIONS_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Keep the original Step 27 folder structure under outputs."
            )

    print("\nReading Step 27 supervised dataset and predictions...")
    data = pd.read_csv(SUPERVISED_FILE)
    old_pred = pd.read_csv(PREDICTIONS_FILE)

    required = FEATURES + [
        "requested_horizon_minutes", "proposed_temporal_split",
        "actual_future_tds", "actual_future_humidity", "actual_future_Temperature",
        "origin_tds", "origin_humidity", "origin_Temperature",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing expected Step 27 supervised columns: {missing}")

    # Keep only complete feature vectors; never impute missing source observations.
    data["requested_horizon_minutes"] = pd.to_numeric(data["requested_horizon_minutes"], errors="coerce")
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[data["requested_horizon_minutes"].isin(HORIZONS)].copy()

    predictions = []
    metrics = []
    selections = []

    for horizon in HORIZONS:
        hdata = data[data["requested_horizon_minutes"] == horizon].copy()
        for target in TARGETS:
            target_col = f"actual_future_{target}"
            origin_col = f"origin_{target}"
            rate_col = RATE_COLUMN[target]
            # Same rows for fair model comparisons, with no feature imputation.
            usable = hdata.dropna(subset=FEATURES + [target_col, origin_col, rate_col]).copy()
            train = usable[usable["proposed_temporal_split"] == "TRAIN"]
            val = usable[usable["proposed_temporal_split"] == "VALIDATION"]
            test = usable[usable["proposed_temporal_split"] == "TEST"]
            if min(len(train), len(val), len(test)) < 10:
                warnings.warn(f"Insufficient rows for {horizon}m {target}; skipped.")
                continue

            X_train, y_train = train[FEATURES], train[target_col].astype(float)
            X_val, y_val = val[FEATURES], val[target_col].astype(float)
            X_test, y_test = test[FEATURES], test[target_col].astype(float)

            # Linear drift baseline: extrapolates from latest observed rate.
            for split_name, frame in [("VALIDATION", val), ("TEST", test)]:
                drift = (
                    frame[origin_col].astype(float).to_numpy()
                    + frame[rate_col].astype(float).to_numpy() * horizon
                )
                metrics.append(metric_row(
                    frame[target_col], drift, split_name, horizon, target, "LinearDrift"
                ))
                for idx, pred_value in zip(frame.index, drift):
                    predictions.append({
                        "row_index": int(idx), "split": split_name,
                        "horizon_minutes": horizon, "target": target,
                        "model": "LinearDrift", "actual": float(frame.loc[idx, target_col]),
                        "prediction": float(pred_value),
                    })

            # Ridge alpha selected only on validation MAE.
            best = None
            best_alpha = None
            for alpha in ALPHAS:
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ])
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, val_pred)
                if best is None or val_mae < best:
                    best, best_alpha = float(val_mae), float(alpha)

            # Refit using TRAIN only with validation-selected alpha.
            final_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=best_alpha)),
            ])
            final_model.fit(X_train, y_train)
            selections.append({
                "horizon_minutes": horizon, "target": target,
                "selected_alpha": best_alpha, "validation_MAE_for_selection": best,
                "train_samples": len(train), "validation_samples": len(val),
                "test_samples": len(test),
            })

            for split_name, frame, X in [
                ("VALIDATION", val, X_val), ("TEST", test, X_test)
            ]:
                ridge_pred = final_model.predict(X)
                metrics.append(metric_row(
                    frame[target_col], ridge_pred, split_name, horizon, target, "Ridge"
                ))
                for idx, pred_value in zip(frame.index, ridge_pred):
                    predictions.append({
                        "row_index": int(idx), "split": split_name,
                        "horizon_minutes": horizon, "target": target,
                        "model": "Ridge", "actual": float(frame.loc[idx, target_col]),
                        "prediction": float(pred_value),
                    })

            print(
                f"{horizon:>2} min | {target:<11} | "
                f"train={len(train):>5} val={len(val):>4} test={len(test):>4} | "
                f"Ridge alpha={best_alpha:g}"
            )

    new_metrics = pd.DataFrame(metrics)
    new_predictions = pd.DataFrame(predictions)
    selection_df = pd.DataFrame(selections)

    # Add Step 27 baseline metrics from its actual prediction records for direct comparison.
    legacy_metrics = []
    for (split, horizon, target), group in old_pred.groupby(["split", "requested_horizon_minutes", "target"]):
        for col, model_name in [
            ("persistence_prediction", "Persistence"),
            ("random_forest_prediction", "RandomForest"),
        ]:
            legacy_metrics.append(metric_row(
                group["actual_value"], group[col], split, horizon, target, model_name
            ))
    combined_metrics = pd.concat(
        [pd.DataFrame(legacy_metrics), new_metrics], ignore_index=True
    ).sort_values(["split", "horizon_minutes", "target", "model"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_metrics.to_csv(OUT_DIR / "01_model_comparison_metrics.csv", index=False)
    new_predictions.to_csv(OUT_DIR / "02_new_model_predictions.csv", index=False)
    selection_df.to_csv(OUT_DIR / "03_ridge_alpha_selection.csv", index=False)

    test = combined_metrics[combined_metrics["split"] == "TEST"]
    wins = test.pivot_table(
        index=["horizon_minutes", "target"], columns="model", values="MAE"
    )
    win_rows = []
    for (h, target), row in wins.iterrows():
        available = row.dropna()
        best_model = available.idxmin() if len(available) else "NA"
        win_rows.append({
            "horizon_minutes": h, "target": target,
            "best_test_MAE_model_descriptive_only": best_model,
            "best_test_MAE": float(available.min()) if len(available) else np.nan,
        })
    pd.DataFrame(win_rows).to_csv(OUT_DIR / "04_test_comparison_summary.csv", index=False)

    report = [
        "BOREMAN STEP 31 - IMPROVED FORECASTING MODEL COMPARISON",
        "=" * 68,
        f"Supervised rows read: {len(data):,}",
        f"New model prediction rows: {len(new_predictions):,}",
        f"Metrics rows including Step 27 benchmarks: {len(combined_metrics):,}",
        f"Forecast horizons: {HORIZONS}",
        f"Targets: {TARGETS}",
        "",
        "MODELS",
        "- Persistence and Random Forest are imported from Step 27 predictions.",
        "- LinearDrift = origin sensor value + latest rate per minute * requested horizon.",
        "- Ridge uses standardized engineered features and median imputation fitted on TRAIN only.",
        "- Ridge alpha is selected from [0.1, 1, 10, 100] using VALIDATION MAE.",
        "- Ridge is refitted on TRAIN only; TEST is not used for selection.",
        "- No source observations are deleted or modified.",
        "",
        "IMPORTANT INTERPRETATION",
        "- TEST metrics are final holdout comparisons; do not choose models by TEST results.",
        "- Compare models primarily using VALIDATION, then report TEST once.",
        "- Linear drift can extrapolate and may produce unrealistic values; inspect predictions.",
        "- This is forecasting sensor readings, not determining water safety.",
        "- The test period is one chronological holdout, not external validation.",
        "",
        "VALIDATION MAE LEADERS BY HORIZON AND TARGET",
    ]
    val = combined_metrics[combined_metrics["split"] == "VALIDATION"]
    for (h, target), grp in val.groupby(["horizon_minutes", "target"]):
        winner = grp.loc[grp["MAE"].idxmin()]
        report.append(
            f"{h} min | {target} | lowest validation MAE: "
            f"{winner['model']} ({winner['MAE']:.6f})"
        )
    report += [
        "",
        "TEST RESULTS ARE PROVIDED IN 01_MODEL_COMPARISON_METRICS.CSV.",
        "Do not select the final model based on test performance.",
    ]
    (OUT_DIR / "05_step31_model_comparison_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n" + "=" * 68)
    print("STEP 31 COMPLETED")
    print("=" * 68)
    print(f"Supervised rows read: {len(data):,}")
    print(f"New model prediction rows: {len(new_predictions):,}")
    print(f"Output folder: {OUT_DIR}")
    print("Step 25B and Step 27 inputs were not modified.")
    print("Review validation results before interpreting the final test results.")

if __name__ == "__main__":
    main()
