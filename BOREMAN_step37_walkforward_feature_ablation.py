from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
INPUT = ROOT / "outputs" / "step27_time_series_forecasting" / "01_forecast_supervised_dataset.csv"
OUT = ROOT / "outputs" / "step37_walkforward_feature_ablation"

TRAIN_DATES = [
    "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
    "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11",
    "2026-10-12", "2026-10-13", "2026-10-14"
]
FOLDS = [
    ("FOLD_1", TRAIN_DATES[:5], TRAIN_DATES[5:7]),
    ("FOLD_2", TRAIN_DATES[:7], TRAIN_DATES[7:9]),
    ("FOLD_3", TRAIN_DATES[:9], TRAIN_DATES[9:11]),
]
TARGETS = {
    "tds": ("origin_tds", "actual_future_tds"),
    "humidity": ("origin_humidity", "actual_future_humidity"),
    "Temperature": ("origin_Temperature", "actual_future_Temperature"),
}
SENSOR_PREFIX = {
    "tds": "tds_",
    "humidity": "humidity_",
    "Temperature": "Temperature_",
}
TIME_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "sampling_interval_minutes"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
HORIZONS = [5, 15, 30, 60]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def make_feature_sets(columns):
    available = set(columns)
    sensor_features = {}
    for sensor, prefix in SENSOR_PREFIX.items():
        sensor_features[sensor] = [
            c for c in columns
            if c.startswith(prefix)
            and c not in {"origin_tds", "origin_humidity", "origin_Temperature"}
            and not c.startswith("actual_future_")
        ]
    time_features = [c for c in TIME_FEATURES if c in available]

    configs = {
        "TARGET_HISTORY_ONLY": {},
        "TARGET_PLUS_OTHER_CURRENT": {},
        "FULL_SENSOR_HISTORY": {},
    }
    for target in TARGETS:
        own = sensor_features[target]
        others = [s for s in TARGETS if s != target]
        other_current = [
            c for s in others for c in sensor_features[s]
            if c.endswith("_current")
        ]
        all_sensor = [c for s in TARGETS for c in sensor_features[s]]
        configs["TARGET_HISTORY_ONLY"][target] = sorted(set(own + time_features))
        configs["TARGET_PLUS_OTHER_CURRENT"][target] = sorted(set(own + other_current + time_features))
        configs["FULL_SENSOR_HISTORY"][target] = sorted(set(all_sensor + time_features))
    return configs

def make_model(alpha):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

def evaluate_predictions(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse(y, pred),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 and np.var(y) > 0 else np.nan,
    }

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)
    required = [
        "origin_source_row_number", "origin_timestamp_namibia",
        "target_timestamp_namibia", "requested_horizon_minutes",
        "proposed_temporal_split", "feature_segment_id",
        "tds_current", "humidity_current", "Temperature_current",
        "origin_tds", "actual_future_tds", "origin_humidity",
        "actual_future_humidity", "origin_Temperature", "actual_future_Temperature"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    input_rows = len(df)
    split = df["proposed_temporal_split"].astype(str).str.strip().str.upper()
    d = df[split.eq("TRAIN")].copy()
    if d.empty:
        raise ValueError("No TRAIN supervised rows found.")

    d["_origin_ts"] = pd.to_datetime(d["origin_timestamp_namibia"], errors="coerce", utc=True)
    d["_target_ts"] = pd.to_datetime(d["target_timestamp_namibia"], errors="coerce", utc=True)
    d = d[d["_origin_ts"].notna() & d["_target_ts"].notna()].copy()
    d["_origin_ts"] = d["_origin_ts"].dt.tz_convert("Africa/Windhoek")
    d["_target_ts"] = d["_target_ts"].dt.tz_convert("Africa/Windhoek")
    d["_origin_date"] = d["_origin_ts"].dt.strftime("%Y-%m-%d")

    actual_dates = sorted(d["_origin_date"].dropna().unique().tolist())
    if actual_dates != TRAIN_DATES:
        raise ValueError(
            "TRAIN origin dates do not match the Step 32 11-date window. "
            f"Expected {TRAIN_DATES}; found {actual_dates}. Stopping."
        )

    d["requested_horizon_minutes"] = pd.to_numeric(d["requested_horizon_minutes"], errors="coerce")
    d = d[d["requested_horizon_minutes"].isin(HORIZONS)].copy()
    configs = make_feature_sets(d.columns.tolist())

    # Ensure feature lists contain numeric predictors only.
    for features_by_target in configs.values():
        for target, features in features_by_target.items():
            for c in features:
                d[c] = pd.to_numeric(d[c], errors="coerce")
    for _, (origin_col, future_col) in TARGETS.items():
        d[origin_col] = pd.to_numeric(d[origin_col], errors="coerce")
        d[future_col] = pd.to_numeric(d[future_col], errors="coerce")

    metric_rows = []
    prediction_frames = []
    alpha_rows = []

    for fold_name, train_dates, validation_dates in FOLDS:
        outer_start = pd.Timestamp(validation_dates[0], tz="Africa/Windhoek")
        outer_train = d[
            d["_origin_date"].isin(train_dates) &
            (d["_target_ts"] < outer_start)
        ].copy()
        outer_val = d[d["_origin_date"].isin(validation_dates)].copy()

        if outer_train.empty or outer_val.empty:
            raise ValueError(
                f"{fold_name} has empty train/validation data. "
                f"Train={len(outer_train)}, validation={len(outer_val)}"
            )

        inner_dates = train_dates[-2:]
        inner_start = pd.Timestamp(inner_dates[0], tz="Africa/Windhoek")
        inner_train = outer_train[
            (~outer_train["_origin_date"].isin(inner_dates)) &
            (outer_train["_target_ts"] < inner_start)
        ].copy()
        inner_val = outer_train[outer_train["_origin_date"].isin(inner_dates)].copy()
        if inner_train.empty or inner_val.empty:
            raise ValueError(f"{fold_name}: inner chronological alpha-selection split is empty.")

        for horizon in HORIZONS:
            tr_h = outer_train[outer_train["requested_horizon_minutes"].eq(horizon)].copy()
            va_h = outer_val[outer_val["requested_horizon_minutes"].eq(horizon)].copy()
            in_tr_h = inner_train[inner_train["requested_horizon_minutes"].eq(horizon)].copy()
            in_va_h = inner_val[inner_val["requested_horizon_minutes"].eq(horizon)].copy()
            if tr_h.empty or va_h.empty or in_tr_h.empty or in_va_h.empty:
                warnings.warn(f"Skipping {fold_name}, {horizon}m due to empty horizon split.")
                continue

            for target, (origin_col, future_col) in TARGETS.items():
                valid_val = va_h[future_col].notna() & va_h[origin_col].notna()
                va = va_h.loc[valid_val].copy()
                if va.empty:
                    continue

                # Persistence baseline: current origin reading is the forecast.
                persistence_pred = va[origin_col].to_numpy(dtype=float)
                persistence_metrics = evaluate_predictions(va[future_col].to_numpy(dtype=float), persistence_pred)
                metric_rows.append({
                    "fold": fold_name, "horizon_minutes": horizon, "target": target,
                    "model": "PERSISTENCE", "alpha": np.nan,
                    "train_samples": len(tr_h), "validation_samples": len(va),
                    **persistence_metrics
                })
                pframe = va[[
                    "origin_source_row_number", "origin_timestamp_namibia",
                    "target_timestamp_namibia", "requested_horizon_minutes"
                ]].copy()
                pframe["fold"] = fold_name
                pframe["target"] = target
                pframe["model"] = "PERSISTENCE"
                pframe["actual"] = va[future_col].to_numpy(dtype=float)
                pframe["prediction"] = persistence_pred
                prediction_frames.append(pframe)

                # Ridge ablation configurations.
                for config_name, features_by_target in configs.items():
                    features = features_by_target[target]
                    if not features:
                        continue
                    # Alpha selection is performed only on the inner chronological split.
                    inner_valid = in_va_h[future_col].notna()
                    inner_train_valid = in_tr_h[future_col].notna()
                    itr = in_tr_h.loc[inner_train_valid]
                    iva = in_va_h.loc[inner_valid]
                    if itr.empty or iva.empty:
                        continue

                    alpha_scores = []
                    for alpha in ALPHAS:
                        model = make_model(alpha)
                        model.fit(itr[features], itr[future_col])
                        pred_inner = model.predict(iva[features])
                        score = mean_absolute_error(iva[future_col], pred_inner)
                        alpha_scores.append((float(score), alpha))
                    best_score, best_alpha = min(alpha_scores, key=lambda x: (x[0], x[1]))
                    alpha_rows.append({
                        "fold": fold_name, "horizon_minutes": horizon, "target": target,
                        "configuration": config_name, "selected_alpha": best_alpha,
                        "inner_validation_mae": best_score,
                        "inner_train_samples": len(itr), "inner_validation_samples": len(iva)
                    })

                    train_valid = tr_h[future_col].notna()
                    tr = tr_h.loc[train_valid]
                    if tr.empty:
                        continue
                    final_model = make_model(best_alpha)
                    final_model.fit(tr[features], tr[future_col])
                    pred = final_model.predict(va[features])
                    metrics = evaluate_predictions(va[future_col].to_numpy(dtype=float), pred)
                    metric_rows.append({
                        "fold": fold_name, "horizon_minutes": horizon, "target": target,
                        "model": config_name, "alpha": best_alpha,
                        "train_samples": len(tr), "validation_samples": len(va),
                        **metrics
                    })
                    pframe = va[[
                        "origin_source_row_number", "origin_timestamp_namibia",
                        "target_timestamp_namibia", "requested_horizon_minutes"
                    ]].copy()
                    pframe["fold"] = fold_name
                    pframe["target"] = target
                    pframe["model"] = config_name
                    pframe["actual"] = va[future_col].to_numpy(dtype=float)
                    pframe["prediction"] = pred
                    prediction_frames.append(pframe)

    metrics_df = pd.DataFrame(metric_rows)
    alpha_df = pd.DataFrame(alpha_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    if metrics_df.empty:
        raise ValueError("No model metrics were produced. Check split coverage and target columns.")

    aggregate = (
        metrics_df.groupby(["horizon_minutes", "target", "model"])
        .agg(
            folds_evaluated=("fold", "nunique"),
            mean_fold_mae=("mae", "mean"),
            mean_fold_rmse=("rmse", "mean"),
            mean_fold_r2=("r2", "mean"),
            total_validation_samples=("validation_samples", "sum")
        ).reset_index()
    )
    leaders = (
        aggregate.sort_values(["horizon_minutes", "target", "mean_fold_mae"])
        .groupby(["horizon_minutes", "target"], as_index=False)
        .first()
    )

    feature_rows = []
    for config_name, features_by_target in configs.items():
        for target, features in features_by_target.items():
            feature_rows.append({
                "configuration": config_name, "target": target,
                "feature_count": len(features), "features": ";".join(features)
            })
    feature_df = pd.DataFrame(feature_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT / "01_walkforward_feature_ablation_metrics.csv", index=False)
    aggregate.to_csv(OUT / "02_aggregate_feature_ablation_metrics.csv", index=False)
    leaders.to_csv(OUT / "03_model_leaders_by_target_horizon.csv", index=False)
    alpha_df.to_csv(OUT / "04_ridge_alpha_selection.csv", index=False)
    feature_df.to_csv(OUT / "05_feature_configuration_dictionary.csv", index=False)
    predictions_df.to_csv(OUT / "06_walkforward_predictions.csv", index=False)

    report = [
        "BOREMAN STEP 37 - WALK-FORWARD FORECASTING FEATURE ABLATION",
        "=" * 82,
        f"Input supervised rows read: {input_rows}",
        f"TRAIN supervised rows used: {len(d)}",
        f"TRAIN origin dates verified: {', '.join(actual_dates)}",
        f"Metrics generated: {len(metrics_df)}",
        f"Prediction rows generated: {len(predictions_df)}",
        "",
        "FOLD DESIGN (Step 32 chronological windows)",
    ]
    for name, train_dates, validation_dates in FOLDS:
        report.append(f"- {name}: train origins {train_dates[0]} to {train_dates[-1]}; validation {validation_dates[0]} to {validation_dates[-1]}")
    report += [
        "",
        "CONFIGURATIONS",
        "- PERSISTENCE: origin sensor reading as future forecast.",
        "- TARGET_HISTORY_ONLY: target sensor's current/history features plus clock and sampling features.",
        "- TARGET_PLUS_OTHER_CURRENT: target history plus current readings from the other two sensors.",
        "- FULL_SENSOR_HISTORY: current, change, rate, lag and rolling-history features from all three sensors plus clock/sampling features.",
        "",
        "VALIDATION SAFEGUARDS",
        "- Only Step 27 supervised rows with proposed_temporal_split == TRAIN are used.",
        "- Step 32 three expanding-window folds are reproduced.",
        "- Training samples are purged when target timestamp reaches the validation period start.",
        "- Ridge alpha is selected using an inner chronological split within each outer training window.",
        "- Imputation and standardisation are fitted inside each training pipeline only.",
        "- The final Step 27 VALIDATION and TEST partitions are not used.",
        "- No model is selected for deployment from this analysis alone.",
        "",
        "METRIC INTERPRETATION",
        "- Lower MAE and RMSE are better; higher R-squared is better.",
        "- Compare feature configurations against persistence and across all three folds.",
        "- Small fold counts and non-stationary sensor behaviour limit generalisation claims.",
        "- Sensor correlations do not establish causation or measurement accuracy.",
        "",
        "OUTPUTS",
        "- 01_walkforward_feature_ablation_metrics.csv",
        "- 02_aggregate_feature_ablation_metrics.csv",
        "- 03_model_leaders_by_target_horizon.csv",
        "- 04_ridge_alpha_selection.csv",
        "- 05_feature_configuration_dictionary.csv",
        "- 06_walkforward_predictions.csv",
        "- 07_step37_feature_ablation_report.txt",
    ]
    (OUT / "07_step37_feature_ablation_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("BOREMAN STEP 37 - WALK-FORWARD FORECASTING FEATURE ABLATION")
    print("=" * 72)
    print(f"Input supervised rows: {input_rows:,}")
    print(f"TRAIN supervised rows: {len(d):,}")
    print(f"Metrics generated: {len(metrics_df):,}")
    print(f"Prediction rows generated: {len(predictions_df):,}")
    print(f"Output folder: {OUT.resolve()}")
    print("Only TRAIN folds used. Final VALIDATION and TEST partitions untouched.")

if __name__ == "__main__":
    main()
