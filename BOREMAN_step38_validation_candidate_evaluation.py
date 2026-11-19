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
OUT = ROOT / "outputs" / "step38_validation_candidate_evaluation"

TRAIN_DATES = [
    "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
    "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11",
    "2026-10-12", "2026-10-13", "2026-10-14"
]
VALIDATION_DATES = ["2026-10-15", "2026-10-16"]
TARGETS = {
    "tds": ("origin_tds", "actual_future_tds"),
    "humidity": ("origin_humidity", "actual_future_humidity"),
    "Temperature": ("origin_Temperature", "actual_future_Temperature"),
}
SENSOR_PREFIX = {"tds": "tds_", "humidity": "humidity_", "Temperature": "Temperature_"}
TIME_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "sampling_interval_minutes"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
HORIZONS = [5, 15, 30, 60]

# Candidate configurations carried forward from Step 37.
# All candidates are evaluated on VALIDATION; TEST remains untouched.
SELECTED_FROM_STEP37 = {
    ("tds", 5): "PERSISTENCE",
    ("tds", 15): "PERSISTENCE",
    ("tds", 30): "PERSISTENCE",
    ("tds", 60): "PERSISTENCE",
    ("humidity", 5): "PERSISTENCE",
    ("humidity", 15): "FULL_SENSOR_HISTORY",
    ("humidity", 30): "FULL_SENSOR_HISTORY",
    ("humidity", 60): "PERSISTENCE",
    ("Temperature", 5): "FULL_SENSOR_HISTORY",
    ("Temperature", 15): "TARGET_HISTORY_ONLY",
    ("Temperature", 30): "TARGET_HISTORY_ONLY",
    ("Temperature", 60): "TARGET_PLUS_OTHER_CURRENT",
}

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
    configs = {}
    for target in TARGETS:
        own = sensor_features[target]
        others = [s for s in TARGETS if s != target]
        other_current = [
            c for s in others for c in sensor_features[s]
            if c.endswith("_current")
        ]
        all_sensor = [c for s in TARGETS for c in sensor_features[s]]
        configs.setdefault("TARGET_HISTORY_ONLY", {})[target] = sorted(set(own + time_features))
        configs.setdefault("TARGET_PLUS_OTHER_CURRENT", {})[target] = sorted(set(own + other_current + time_features))
        configs.setdefault("FULL_SENSOR_HISTORY", {})[target] = sorted(set(all_sensor + time_features))
    return configs

def make_model(alpha):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

def metric_values(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
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
        "origin_tds", "actual_future_tds", "origin_humidity",
        "actual_future_humidity", "origin_Temperature", "actual_future_Temperature"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    input_rows = len(df)
    split = df["proposed_temporal_split"].astype(str).str.strip().str.upper()
    d = df[split.isin(["TRAIN", "VALIDATION"])].copy()
    d["_origin_ts"] = pd.to_datetime(d["origin_timestamp_namibia"], errors="coerce", utc=True)
    d["_target_ts"] = pd.to_datetime(d["target_timestamp_namibia"], errors="coerce", utc=True)
    d = d[d["_origin_ts"].notna() & d["_target_ts"].notna()].copy()
    d["_origin_ts"] = d["_origin_ts"].dt.tz_convert("Africa/Windhoek")
    d["_target_ts"] = d["_target_ts"].dt.tz_convert("Africa/Windhoek")
    d["_origin_date"] = d["_origin_ts"].dt.strftime("%Y-%m-%d")
    d["requested_horizon_minutes"] = pd.to_numeric(d["requested_horizon_minutes"], errors="coerce")
    d = d[d["requested_horizon_minutes"].isin(HORIZONS)].copy()

    train = d[split.loc[d.index].eq("TRAIN")].copy()
    validation = d[split.loc[d.index].eq("VALIDATION")].copy()
    if train.empty or validation.empty:
        raise ValueError(f"TRAIN or VALIDATION partition empty. TRAIN={len(train)}, VALIDATION={len(validation)}")

    train_dates = sorted(train["_origin_date"].dropna().unique().tolist())
    validation_dates = sorted(validation["_origin_date"].dropna().unique().tolist())
    if train_dates != TRAIN_DATES:
        raise ValueError(f"TRAIN date mismatch. Expected {TRAIN_DATES}; found {train_dates}")
    if validation_dates != VALIDATION_DATES:
        raise ValueError(f"VALIDATION date mismatch. Expected {VALIDATION_DATES}; found {validation_dates}")

    validation_start = validation["_origin_ts"].min()
    train_purged = train[train["_target_ts"] < validation_start].copy()
    if train_purged.empty:
        raise ValueError("No TRAIN rows remain after purging targets at/after validation start.")

    # Alpha selection: inner chronological split using Oct 13-14 as inner validation.
    inner_dates = ["2026-10-13", "2026-10-14"]
    inner_start = pd.Timestamp("2026-10-13", tz="Africa/Windhoek")
    inner_train = train_purged[
        (~train_purged["_origin_date"].isin(inner_dates)) &
        (train_purged["_target_ts"] < inner_start)
    ].copy()
    inner_validation = train_purged[train_purged["_origin_date"].isin(inner_dates)].copy()
    if inner_train.empty or inner_validation.empty:
        raise ValueError("Inner chronological alpha-selection split is empty.")

    configs = make_feature_sets(d.columns.tolist())
    all_feature_cols = sorted({c for conf in configs.values() for target in TARGETS for c in conf[target]})
    for c in all_feature_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for _, (origin_col, future_col) in TARGETS.items():
        d[origin_col] = pd.to_numeric(d[origin_col], errors="coerce")
        d[future_col] = pd.to_numeric(d[future_col], errors="coerce")

    metrics = []
    predictions = []
    alpha_rows = []
    coverage_rows = []

    for horizon in HORIZONS:
        tr_h = train_purged[train_purged["requested_horizon_minutes"].eq(horizon)].copy()
        in_tr_h = inner_train[inner_train["requested_horizon_minutes"].eq(horizon)].copy()
        in_va_h = inner_validation[inner_validation["requested_horizon_minutes"].eq(horizon)].copy()
        va_h = validation[validation["requested_horizon_minutes"].eq(horizon)].copy()
        coverage_rows.append({
            "horizon_minutes": horizon,
            "train_after_purge": len(tr_h),
            "inner_train": len(in_tr_h),
            "inner_validation": len(in_va_h),
            "outer_validation": len(va_h),
        })

        for target, (origin_col, future_col) in TARGETS.items():
            va = va_h[va_h[origin_col].notna() & va_h[future_col].notna()].copy()
            if va.empty:
                warnings.warn(f"No validation rows for {target}, {horizon}m")
                continue

            # Persistence candidate.
            persistence_pred = va[origin_col].to_numpy(dtype=float)
            metrics.append({
                "target": target, "horizon_minutes": horizon, "model": "PERSISTENCE",
                "selected_from_step37": SELECTED_FROM_STEP37[(target, horizon)] == "PERSISTENCE",
                "alpha": np.nan, "train_samples": len(tr_h), "validation_samples": len(va),
                **metric_values(va[future_col].to_numpy(dtype=float), persistence_pred)
            })
            pf = va[[
                "origin_source_row_number", "origin_timestamp_namibia",
                "target_timestamp_namibia", "requested_horizon_minutes"
            ]].copy()
            pf["target"] = target
            pf["model"] = "PERSISTENCE"
            pf["actual"] = va[future_col].to_numpy(dtype=float)
            pf["prediction"] = persistence_pred
            predictions.append(pf)

            # Evaluate all three Ridge feature configurations on validation.
            for config_name, feature_map in configs.items():
                features = feature_map[target]
                itr = in_tr_h[in_tr_h[future_col].notna()].copy()
                iva = in_va_h[in_va_h[future_col].notna()].copy()
                tr = tr_h[tr_h[future_col].notna()].copy()
                if itr.empty or iva.empty or tr.empty:
                    continue

                alpha_candidates = []
                for alpha in ALPHAS:
                    model = make_model(alpha)
                    model.fit(itr[features], itr[future_col])
                    inner_pred = model.predict(iva[features])
                    alpha_candidates.append((
                        float(mean_absolute_error(iva[future_col], inner_pred)),
                        alpha
                    ))
                inner_mae, best_alpha = min(alpha_candidates, key=lambda x: (x[0], x[1]))
                alpha_rows.append({
                    "target": target, "horizon_minutes": horizon,
                    "configuration": config_name, "selected_alpha": best_alpha,
                    "inner_validation_mae": inner_mae,
                    "inner_train_samples": len(itr), "inner_validation_samples": len(iva)
                })

                final_model = make_model(best_alpha)
                final_model.fit(tr[features], tr[future_col])
                pred = final_model.predict(va[features])
                metrics.append({
                    "target": target, "horizon_minutes": horizon, "model": config_name,
                    "selected_from_step37": SELECTED_FROM_STEP37[(target, horizon)] == config_name,
                    "alpha": best_alpha, "train_samples": len(tr), "validation_samples": len(va),
                    **metric_values(va[future_col].to_numpy(dtype=float), pred)
                })
                pf = va[[
                    "origin_source_row_number", "origin_timestamp_namibia",
                    "target_timestamp_namibia", "requested_horizon_minutes"
                ]].copy()
                pf["target"] = target
                pf["model"] = config_name
                pf["actual"] = va[future_col].to_numpy(dtype=float)
                pf["prediction"] = pred
                predictions.append(pf)

    metrics_df = pd.DataFrame(metrics)
    alpha_df = pd.DataFrame(alpha_rows)
    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    coverage_df = pd.DataFrame(coverage_rows)

    # Rank by validation MAE; this is candidate selection only, not final test evidence.
    rankings = (
        metrics_df.sort_values(["target", "horizon_minutes", "mae"])
        .groupby(["target", "horizon_minutes"], as_index=False)
        .first()
    )

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT / "01_validation_candidate_metrics.csv", index=False)
    rankings.to_csv(OUT / "02_validation_candidate_leaders.csv", index=False)
    alpha_df.to_csv(OUT / "03_validation_ridge_alpha_selection.csv", index=False)
    predictions_df.to_csv(OUT / "04_validation_candidate_predictions.csv", index=False)
    coverage_df.to_csv(OUT / "05_validation_sample_coverage.csv", index=False)

    report = [
        "BOREMAN STEP 38 - VALIDATION CANDIDATE EVALUATION",
        "=" * 78,
        f"Input supervised rows read: {input_rows}",
        f"TRAIN rows used for fitting: {len(train_purged)}",
        f"VALIDATION rows evaluated: {len(validation)}",
        f"TRAIN dates: {', '.join(train_dates)}",
        f"VALIDATION dates: {', '.join(validation_dates)}",
        "",
        "PROTOCOL",
        "- Candidate feature configurations are carried forward from Step 37.",
        "- All candidate configurations are evaluated on the untouched Step 27 VALIDATION partition.",
        "- Step 27 TEST rows are excluded from loading and evaluation.",
        "- TRAIN rows are purged when target timestamp reaches the first validation origin timestamp.",
        "- Ridge alpha is selected using an inner chronological split within TRAIN: Oct 13-14 inner validation.",
        "- Ridge pipelines fit median imputation and standardisation on training data only.",
        "- Validation MAE ranks candidates; this is model selection, not final independent performance estimation.",
        "",
        "STEP 37 PREFERRED CANDIDATES",
    ]
    for (target, horizon), model in SELECTED_FROM_STEP37.items():
        report.append(f"- {target}, {horizon} min: {model}")
    report += [
        "",
        "IMPORTANT",
        "- Do not inspect or use the final TEST partition until candidate configurations are frozen.",
        "- Validation results may be used to choose final candidate per target and horizon.",
        "- After selection, freeze features and alpha-selection procedure before final TEST evaluation.",
        "- Sensor readings do not establish water safety or laboratory accuracy.",
        "",
        "OUTPUTS",
        "- 01_validation_candidate_metrics.csv",
        "- 02_validation_candidate_leaders.csv",
        "- 03_validation_ridge_alpha_selection.csv",
        "- 04_validation_candidate_predictions.csv",
        "- 05_validation_sample_coverage.csv",
        "- 06_step38_validation_candidate_report.txt",
    ]
    (OUT / "06_step38_validation_candidate_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("BOREMAN STEP 38 - VALIDATION CANDIDATE EVALUATION")
    print("=" * 68)
    print(f"TRAIN rows after target-time purge: {len(train_purged):,}")
    print(f"VALIDATION supervised rows: {len(validation):,}")
    print(f"Metrics generated: {len(metrics_df):,}")
    print(f"Prediction rows generated: {len(predictions_df):,}")
    print(f"Output folder: {OUT.resolve()}")
    print("Final TEST partition was not loaded or evaluated.")

if __name__ == "__main__":
    main()
