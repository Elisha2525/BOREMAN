"""
BOREMAN STEP 32 - EXPANDING-WINDOW WALK-FORWARD BACKTEST

Purpose:
- Test whether Ridge's Step 31 improvement is stable across multiple
  chronological validation windows inside the TRAIN period.
- Compare Ridge with persistence and linear drift.
- TEST and Step 27 TEST predictions are not read or used.
- Uses supervised rows whose proposed_temporal_split is TRAIN only.
- Purges training samples whose future target timestamp reaches the next
  validation window, preventing target leakage across each cutoff.

Ridge alpha is selected within each fold using an inner chronological
validation day from the fold's training history. The selected alpha is then
refit on the complete fold training history and evaluated on the next
two chronological dates.
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
OUT_DIR = ROOT / "outputs" / "step32_walk_forward_backtesting"

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
RATE = {
    "tds": "tds_rate_per_minute",
    "humidity": "humidity_rate_per_minute",
    "Temperature": "Temperature_rate_per_minute",
}

def make_ridge(alpha):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "sample_count": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan,
        "mean_signed_error": float(np.mean(predicted - actual)),
    }

def main():
    print("=" * 70)
    print("BOREMAN STEP 32 - EXPANDING-WINDOW WALK-FORWARD BACKTEST")
    print("=" * 70)

    if not SUPERVISED_FILE.exists():
        raise FileNotFoundError(
            f"Missing Step 27 supervised dataset: {SUPERVISED_FILE}"
        )

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
    for col in FEATURES + list(CURRENT.values()) + [f"actual_future_{t}" for t in TARGETS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dates = sorted(df["origin_date"].dropna().unique())
    if len(dates) < 9:
        raise ValueError(
            f"Only {len(dates)} TRAIN dates found; at least 9 are needed "
            "for three expanding-window folds."
        )

    # Three expanding folds: 5 dates -> next 2, 7 -> next 2, 9 -> next 2.
    fold_specs = []
    for train_end, val_end in [(5, 7), (7, 9), (9, 11)]:
        if val_end <= len(dates):
            fold_specs.append((train_end, val_end))
    if len(fold_specs) < 2:
        raise ValueError("Not enough chronological TRAIN dates for walk-forward folds.")

    print(f"TRAIN supervised rows: {len(df):,}")
    print(f"Distinct TRAIN origin dates: {len(dates)}")
    print(f"Walk-forward folds: {len(fold_specs)}")
    for i, (a, b) in enumerate(fold_specs, 1):
        print(f"Fold {i}: history through {dates[a-1]} | validate {dates[a]} to {dates[b-1]}")

    metrics_rows, alpha_rows = [], []
    for horizon in HORIZONS:
        horizon_df = df[df["requested_horizon_minutes"].eq(horizon)].copy()
        for target in TARGETS:
            target_col = f"actual_future_{target}"
            current_col = CURRENT[target]
            rate_col = RATE[target]
            usable = horizon_df.dropna(
                subset=FEATURES + [target_col, current_col, rate_col]
            ).copy()

            for fold_no, (train_end, val_end) in enumerate(fold_specs, 1):
                history_dates = set(dates[:train_end])
                validation_dates = set(dates[train_end:val_end])
                validation_start = pd.Timestamp(dates[train_end], tz="UTC")

                fold_train = usable[
                    usable["origin_date"].isin(history_dates)
                    & (usable["target_dt"] < validation_start)
                ].copy()
                fold_val = usable[usable["origin_date"].isin(validation_dates)].copy()

                # Inner chronological alpha selection: reserve the final
                # history date as inner validation, purge crossing targets.
                inner_date = dates[train_end - 1]
                inner_start = pd.Timestamp(inner_date, tz="UTC")
                inner_val = fold_train[fold_train["origin_date"].eq(inner_date)]
                inner_train = fold_train[
                    (fold_train["origin_date"] < inner_date)
                    & (fold_train["target_dt"] < inner_start)
                ]

                if len(inner_train) < 50 or len(inner_val) < 20 or len(fold_val) < 20:
                    warnings.warn(
                        f"Skipping fold {fold_no}, {horizon}m {target}: "
                        f"inner_train={len(inner_train)}, inner_val={len(inner_val)}, "
                        f"fold_val={len(fold_val)}"
                    )
                    continue

                best_alpha, best_mae = None, np.inf
                for alpha in ALPHAS:
                    candidate = make_ridge(alpha)
                    candidate.fit(inner_train[FEATURES], inner_train[target_col])
                    inner_pred = candidate.predict(inner_val[FEATURES])
                    score = mean_absolute_error(inner_val[target_col], inner_pred)
                    if score < best_mae:
                        best_alpha, best_mae = float(alpha), float(score)

                final_ridge = make_ridge(best_alpha)
                final_ridge.fit(fold_train[FEATURES], fold_train[target_col])
                ridge_pred = final_ridge.predict(fold_val[FEATURES])

                persistence_pred = fold_val[current_col].to_numpy(dtype=float)
                drift_pred = (
                    persistence_pred
                    + fold_val[rate_col].to_numpy(dtype=float) * horizon
                )

                for model_name, prediction in [
                    ("Persistence", persistence_pred),
                    ("LinearDrift", drift_pred),
                    ("Ridge", ridge_pred),
                ]:
                    row = {
                        "fold": fold_no,
                        "horizon_minutes": horizon,
                        "target": target,
                        "model": model_name,
                        "train_start_date": str(dates[0]),
                        "train_end_date": str(dates[train_end - 1]),
                        "validation_start_date": str(dates[train_end]),
                        "validation_end_date": str(dates[val_end - 1]),
                        **calculate_metrics(fold_val[target_col], prediction),
                    }
                    metrics_rows.append(row)

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

    metrics = pd.DataFrame(metrics_rows)
    alphas = pd.DataFrame(alpha_rows)
    if metrics.empty:
        raise RuntimeError("No walk-forward results were produced.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / "01_walk_forward_metrics.csv", index=False)
    alphas.to_csv(OUT_DIR / "02_walk_forward_alpha_selection.csv", index=False)

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
    aggregate.to_csv(OUT_DIR / "03_walk_forward_aggregate.csv", index=False)

    pivot = aggregate.pivot_table(
        index=["horizon_minutes", "target"],
        columns="model", values="mean_fold_MAE"
    ).reset_index()
    model_cols = [c for c in ["Persistence", "LinearDrift", "Ridge"] if c in pivot]
    pivot["lowest_mean_fold_MAE"] = pivot[model_cols].idxmin(axis=1)
    pivot.to_csv(OUT_DIR / "04_walk_forward_model_leaders.csv", index=False)

    report = [
        "BOREMAN STEP 32 - EXPANDING-WINDOW WALK-FORWARD BACKTEST",
        "=" * 70,
        f"TRAIN supervised rows read: {len(df):,}",
        f"Distinct TRAIN origin dates: {len(dates)}",
        f"Fold metric rows: {len(metrics)}",
        f"Alpha selection rows: {len(alphas)}",
        "",
        "METHOD",
        "- Uses TRAIN-period supervised samples only.",
        "- Expanding chronological history with the next two dates as validation.",
        "- Three folds are attempted: 5->2 dates, 7->2 dates, and 9->2 dates.",
        "- Training samples whose target timestamp reaches the validation cutoff are purged.",
        "- Ridge alpha is selected by an inner chronological validation date.",
        "- Ridge is refitted on the complete fold history after alpha selection.",
        "- Compares Ridge with persistence and linear drift.",
        "- Step 27 VALIDATION and TEST metrics are not used.",
        "- Step 27 TEST data is not read or evaluated.",
        "",
        "INTERPRETATION",
        "- Consistent fold performance is stronger evidence of stability than one validation window.",
        "- Small fold sample counts can make metrics unstable.",
        "- Linear drift may produce implausible values; inspect it as a diagnostic only.",
        "- This evaluates sensor forecasts, not water safety.",
        "",
        "MEAN FOLD MAE LEADERS",
    ]
    for _, row in pivot.iterrows():
        report.append(
            f"{int(row['horizon_minutes'])} min | {row['target']} | "
            f"{row['lowest_mean_fold_MAE']}"
        )
    (OUT_DIR / "05_step32_walk_forward_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nSTEP 32 COMPLETED")
    print(f"Metric rows: {len(metrics):,}")
    print(f"Alpha selection rows: {len(alphas):,}")
    print(f"Output folder: {OUT_DIR}")
    print("Only TRAIN-period data was used; Step 27 TEST was not accessed.")

if __name__ == "__main__":
    main()
