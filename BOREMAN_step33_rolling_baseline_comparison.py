"""
BOREMAN STEP 33 - ROLLING-MEAN BASELINE WALK-FORWARD COMPARISON

Compares:
- Persistence (current reading)
- Previous-five-observation rolling mean (existing causal feature)
- Ridge Regression (same feature set and inner chronological alpha selection)

Uses TRAIN-period supervised samples only and the same three expanding
chronological folds as Step 32. No Step 27 TEST data is read or evaluated.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
SUPERVISED_FILE = ROOT / "outputs" / "step27_time_series_forecasting" / "01_forecast_supervised_dataset.csv"
OUT_DIR = ROOT / "outputs" / "step33_rolling_baseline_comparison"

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
CURRENT = {
    "tds": "origin_tds",
    "humidity": "origin_humidity",
    "Temperature": "origin_Temperature",
}
ROLLING = {
    "tds": "tds_rolling_mean_previous_5",
    "humidity": "humidity_rolling_mean_previous_5",
    "Temperature": "Temperature_rolling_mean_previous_5",
}

def ridge_pipeline(alpha):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])

def metric(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "sample_count": len(actual),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "mean_signed_error": float(np.mean(predicted - actual)),
    }

def main():
    print("=" * 70)
    print("BOREMAN STEP 33 - ROLLING-MEAN BASELINE WALK-FORWARD COMPARISON")
    print("=" * 70)
    if not SUPERVISED_FILE.exists():
        raise FileNotFoundError(f"Missing Step 27 supervised file: {SUPERVISED_FILE}")

    df = pd.read_csv(SUPERVISED_FILE)
    required = FEATURES + [
        "origin_timestamp_namibia", "target_timestamp_namibia",
        "requested_horizon_minutes", "proposed_temporal_split",
    ] + [f"actual_future_{t}" for t in TARGETS] + list(CURRENT.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["proposed_temporal_split"].eq("TRAIN")].copy()
    df["origin_dt"] = pd.to_datetime(df["origin_timestamp_namibia"], utc=True, errors="coerce")
    df["target_dt"] = pd.to_datetime(df["target_timestamp_namibia"], utc=True, errors="coerce")
    df = df.dropna(subset=["origin_dt", "target_dt"])
    df["origin_date"] = df["origin_dt"].dt.date
    df["requested_horizon_minutes"] = pd.to_numeric(
        df["requested_horizon_minutes"], errors="coerce"
    )
    numeric_cols = FEATURES + list(CURRENT.values()) + [
        f"actual_future_{t}" for t in TARGETS
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dates = sorted(df["origin_date"].dropna().unique())
    if len(dates) < 11:
        raise ValueError(
            f"Expected at least 11 TRAIN dates for the configured folds; found {len(dates)}."
        )
    fold_specs = [(5, 7), (7, 9), (9, 11)]

    print(f"TRAIN supervised rows: {len(df):,}")
    print(f"TRAIN dates: {len(dates)}")
    print("Models: Persistence, RollingMean5, Ridge")
    print("TEST split is not read or evaluated.")

    metric_rows, alpha_rows = [], []
    for horizon in HORIZONS:
        hdf = df[df["requested_horizon_minutes"].eq(horizon)].copy()
        for target in TARGETS:
            ycol = f"actual_future_{target}"
            current_col = CURRENT[target]
            rolling_col = ROLLING[target]
            usable = hdf.dropna(
                subset=FEATURES + [ycol, current_col, rolling_col]
            ).copy()

            for fold_no, (train_end, val_end) in enumerate(fold_specs, 1):
                hist_dates = set(dates[:train_end])
                val_dates = set(dates[train_end:val_end])
                outer_cutoff = pd.Timestamp(dates[train_end], tz="UTC")
                fold_train = usable[
                    usable["origin_date"].isin(hist_dates)
                    & (usable["target_dt"] < outer_cutoff)
                ].copy()
                fold_val = usable[usable["origin_date"].isin(val_dates)].copy()

                inner_date = dates[train_end - 1]
                inner_cutoff = pd.Timestamp(inner_date, tz="UTC")
                inner_val = fold_train[fold_train["origin_date"].eq(inner_date)]
                inner_train = fold_train[
                    (fold_train["origin_date"] < inner_date)
                    & (fold_train["target_dt"] < inner_cutoff)
                ]

                if len(inner_train) < 50 or len(inner_val) < 20 or len(fold_val) < 20:
                    warnings.warn(f"Skipping fold {fold_no}, {horizon}m {target}: insufficient rows.")
                    continue

                best_alpha, best_mae = None, np.inf
                for alpha in ALPHAS:
                    candidate = ridge_pipeline(alpha)
                    candidate.fit(inner_train[FEATURES], inner_train[ycol])
                    val_prediction = candidate.predict(inner_val[FEATURES])
                    score = mean_absolute_error(inner_val[ycol], val_prediction)
                    if score < best_mae:
                        best_alpha, best_mae = float(alpha), float(score)

                ridge = ridge_pipeline(best_alpha)
                ridge.fit(fold_train[FEATURES], fold_train[ycol])
                ridge_pred = ridge.predict(fold_val[FEATURES])

                predictions = {
                    "Persistence": fold_val[current_col].to_numpy(dtype=float),
                    "RollingMean5": fold_val[rolling_col].to_numpy(dtype=float),
                    "Ridge": ridge_pred,
                }
                for model_name, pred in predictions.items():
                    metric_rows.append({
                        "fold": fold_no,
                        "horizon_minutes": horizon,
                        "target": target,
                        "model": model_name,
                        "train_end_date": str(dates[train_end - 1]),
                        "validation_start_date": str(dates[train_end]),
                        "validation_end_date": str(dates[val_end - 1]),
                        **metric(fold_val[ycol], pred),
                    })
                alpha_rows.append({
                    "fold": fold_no,
                    "horizon_minutes": horizon,
                    "target": target,
                    "selected_alpha": best_alpha,
                    "inner_validation_MAE": best_mae,
                    "inner_train_rows": len(inner_train),
                    "inner_validation_rows": len(inner_val),
                    "fold_train_rows": len(fold_train),
                    "fold_validation_rows": len(fold_val),
                })
            print(f"Completed {horizon} min | {target}")

    metrics = pd.DataFrame(metric_rows)
    alphas = pd.DataFrame(alpha_rows)
    if metrics.empty:
        raise RuntimeError("No Step 33 results were produced.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / "01_rolling_baseline_walk_forward_metrics.csv", index=False)
    alphas.to_csv(OUT_DIR / "02_ridge_alpha_selection.csv", index=False)

    aggregate = (
        metrics.groupby(["horizon_minutes", "target", "model"])
        .agg(
            folds_completed=("fold", "nunique"),
            total_validation_samples=("sample_count", "sum"),
            mean_fold_MAE=("MAE", "mean"),
            median_fold_MAE=("MAE", "median"),
            mean_fold_RMSE=("RMSE", "mean"),
            mean_fold_R2=("R2", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(OUT_DIR / "03_rolling_baseline_aggregate.csv", index=False)

    pivot = aggregate.pivot_table(
        index=["horizon_minutes", "target"],
        columns="model", values="mean_fold_MAE"
    ).reset_index()
    cols = [c for c in ["Persistence", "RollingMean5", "Ridge"] if c in pivot.columns]
    pivot["lowest_mean_fold_MAE"] = pivot[cols].idxmin(axis=1)
    pivot.to_csv(OUT_DIR / "04_rolling_baseline_model_leaders.csv", index=False)

    report = [
        "BOREMAN STEP 33 - ROLLING-MEAN BASELINE WALK-FORWARD COMPARISON",
        "=" * 70,
        f"TRAIN supervised rows: {len(df):,}",
        f"Metric rows: {len(metrics):,}",
        f"Alpha selection rows: {len(alphas):,}",
        "",
        "METHOD",
        "- TRAIN-period supervised data only.",
        "- Three expanding chronological folds: 5->2, 7->2, 9->2 dates.",
        "- Purges training samples whose target timestamp reaches the outer validation cutoff.",
        "- Ridge alpha selected on an inner chronological validation date.",
        "- RollingMean5 uses the previous-five-observation mean feature already engineered in Step 25B.",
        "- No Step 27 VALIDATION or TEST data is used.",
        "- No source observations are modified or imputed.",
        "",
        "INTERPRETATION",
        "- Compare fold consistency as well as mean fold MAE.",
        "- RollingMean5 is a simple causal baseline, not a trained model.",
        "- This forecasts sensor readings, not water safety.",
        "",
        "MEAN FOLD MAE LEADERS",
    ]
    for _, row in pivot.iterrows():
        report.append(
            f"{int(row['horizon_minutes'])} min | {row['target']} | "
            f"{row['lowest_mean_fold_MAE']}"
        )
    (OUT_DIR / "05_step33_rolling_baseline_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nSTEP 33 COMPLETED")
    print(f"Metric rows: {len(metrics):,}")
    print(f"Output folder: {OUT_DIR}")
    print("Only TRAIN-period data was used; final TEST remains untouched.")

if __name__ == "__main__":
    main()
