from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
SUPERVISED_INPUT = ROOT / "outputs" / "step27_time_series_forecasting" / "01_forecast_supervised_dataset.csv"
ALPHA_INPUT = ROOT / "outputs" / "step38_validation_candidate_evaluation" / "03_validation_ridge_alpha_selection.csv"
OUT = ROOT / "outputs" / "step39_final_test_evaluation"

TRAIN_DATES = [
    "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
    "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11",
    "2026-10-12", "2026-10-13", "2026-10-14"
]
VALIDATION_DATES = ["2026-10-15", "2026-10-16"]
TEST_DATES = [
    "2026-10-17", "2026-10-18", "2026-10-19", "2026-10-20",
    "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24",
    "2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28",
    "2026-10-29"
]
HORIZONS = [5, 15, 30, 60]
TARGETS = {
    "tds": ("origin_tds", "actual_future_tds"),
    "humidity": ("origin_humidity", "actual_future_humidity"),
    "Temperature": ("origin_Temperature", "actual_future_Temperature"),
}
SENSOR_PREFIX = {"tds": "tds_", "humidity": "humidity_", "Temperature": "Temperature_"}
TIME_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "sampling_interval_minutes"]

# Frozen from Step 38 validation minimum-MAE results. Do not change after TEST evaluation.
FROZEN_CANDIDATES = {
    ("tds", 5): "FULL_SENSOR_HISTORY",
    ("tds", 15): "FULL_SENSOR_HISTORY",
    ("tds", 30): "FULL_SENSOR_HISTORY",
    ("tds", 60): "PERSISTENCE",
    ("humidity", 5): "FULL_SENSOR_HISTORY",
    ("humidity", 15): "FULL_SENSOR_HISTORY",
    ("humidity", 30): "TARGET_HISTORY_ONLY",
    ("humidity", 60): "PERSISTENCE",
    ("Temperature", 5): "FULL_SENSOR_HISTORY",
    ("Temperature", 15): "FULL_SENSOR_HISTORY",
    ("Temperature", 30): "TARGET_HISTORY_ONLY",
    ("Temperature", 60): "TARGET_HISTORY_ONLY",
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
        other_sensors = [s for s in TARGETS if s != target]
        other_current = [
            c for s in other_sensors for c in sensor_features[s]
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
        ("ridge", Ridge(alpha=float(alpha)))
    ])

def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else np.nan,
    }

def main():
    if not SUPERVISED_INPUT.exists():
        raise FileNotFoundError(f"Supervised input not found: {SUPERVISED_INPUT}")
    if not ALPHA_INPUT.exists():
        raise FileNotFoundError(
            f"Step 38 alpha-selection file not found: {ALPHA_INPUT}. "
            "Place 03_validation_ridge_alpha_selection.csv in the Step 38 output folder."
        )

    df = pd.read_csv(SUPERVISED_INPUT, low_memory=False)
    alpha_df = pd.read_csv(ALPHA_INPUT)

    required = [
        "origin_source_row_number", "origin_timestamp_namibia",
        "target_timestamp_namibia", "requested_horizon_minutes",
        "proposed_temporal_split", "feature_segment_id",
        "origin_tds", "actual_future_tds", "origin_humidity",
        "actual_future_humidity", "origin_Temperature", "actual_future_Temperature"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required supervised columns: {missing}")

    alpha_required = ["target", "horizon_minutes", "configuration", "selected_alpha"]
    missing_alpha = [c for c in alpha_required if c not in alpha_df.columns]
    if missing_alpha:
        raise ValueError(f"Missing required Step 38 alpha columns: {missing_alpha}")

    input_rows = len(df)
    split = df["proposed_temporal_split"].astype(str).str.strip().str.upper()
    d = df[split.isin(["TRAIN", "VALIDATION", "TEST"])].copy()

    d["_origin_ts"] = pd.to_datetime(d["origin_timestamp_namibia"], errors="coerce", utc=True)
    d["_target_ts"] = pd.to_datetime(d["target_timestamp_namibia"], errors="coerce", utc=True)
    d = d[d["_origin_ts"].notna() & d["_target_ts"].notna()].copy()
    d["_origin_ts"] = d["_origin_ts"].dt.tz_convert("Africa/Windhoek")
    d["_target_ts"] = d["_target_ts"].dt.tz_convert("Africa/Windhoek")
    d["_origin_date"] = d["_origin_ts"].dt.strftime("%Y-%m-%d")
    d["requested_horizon_minutes"] = pd.to_numeric(d["requested_horizon_minutes"], errors="coerce")
    d = d[d["requested_horizon_minutes"].isin(HORIZONS)].copy()

    split = d["proposed_temporal_split"].astype(str).str.strip().str.upper()
    train = d[split.eq("TRAIN")].copy()
    validation = d[split.eq("VALIDATION")].copy()
    test = d[split.eq("TEST")].copy()

    train_dates = sorted(train["_origin_date"].dropna().unique().tolist())
    validation_dates = sorted(validation["_origin_date"].dropna().unique().tolist())
    test_dates = sorted(test["_origin_date"].dropna().unique().tolist())
    if train_dates != TRAIN_DATES:
        raise ValueError(f"TRAIN dates mismatch. Expected {TRAIN_DATES}; found {train_dates}")
    if validation_dates != VALIDATION_DATES:
        raise ValueError(f"VALIDATION dates mismatch. Expected {VALIDATION_DATES}; found {validation_dates}")
    # Supervised origin dates are sparse when no future target pair can be
    # constructed on some calendar dates. Validate the approved date window,
    # not the presence of every calendar day.
    if not test_dates:
        raise ValueError("No TEST origin dates found in the supervised dataset.")
    unexpected_test_dates = sorted(set(test_dates) - set(TEST_DATES))
    if unexpected_test_dates:
        raise ValueError(
            "TEST origin dates fall outside the approved Step 23 TEST window. "
            f"Unexpected dates: {unexpected_test_dates}; observed: {test_dates}"
        )
    if min(test_dates) <= max(VALIDATION_DATES):
        raise ValueError(
            "TEST origin dates overlap or precede the VALIDATION period. "
            f"Observed TEST dates: {test_dates}"
        )

    test_start = test["_origin_ts"].min()
    pretest = d[split.isin(["TRAIN", "VALIDATION"])].copy()
    fit_data = pretest[pretest["_target_ts"] < test_start].copy()
    if fit_data.empty or test.empty:
        raise ValueError("No fitting data after target-time purge, or TEST data is empty.")

    configs = make_feature_sets(d.columns.tolist())
    needed_features = sorted({
        c for (target, horizon), config_name in FROZEN_CANDIDATES.items()
        if config_name != "PERSISTENCE"
        for c in configs[config_name][target]
    })
    for c in needed_features:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        fit_data[c] = pd.to_numeric(fit_data[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")
    for _, (origin_col, future_col) in TARGETS.items():
        for frame in (fit_data, test):
            frame[origin_col] = pd.to_numeric(frame[origin_col], errors="coerce")
            frame[future_col] = pd.to_numeric(frame[future_col], errors="coerce")

    metric_rows = []
    prediction_frames = []
    freeze_rows = []
    coverage_rows = []

    for horizon in HORIZONS:
        fit_h = fit_data[fit_data["requested_horizon_minutes"].eq(horizon)].copy()
        test_h = test[test["requested_horizon_minutes"].eq(horizon)].copy()
        coverage_rows.append({
            "horizon_minutes": horizon,
            "fit_samples_after_purge": len(fit_h),
            "test_samples": len(test_h),
            "test_origin_dates": test_h["_origin_date"].nunique(),
        })

        for target, (origin_col, future_col) in TARGETS.items():
            candidate = FROZEN_CANDIDATES[(target, horizon)]
            va = test_h[test_h[origin_col].notna() & test_h[future_col].notna()].copy()
            tr = fit_h[fit_h[future_col].notna()].copy()
            if va.empty or tr.empty:
                raise ValueError(f"Empty fit/test subset for {target}, {horizon}m.")

            # Record frozen selection before fitting/scoring.
            alpha = np.nan
            features = []
            if candidate != "PERSISTENCE":
                row = alpha_df[
                    alpha_df["target"].astype(str).eq(target) &
                    pd.to_numeric(alpha_df["horizon_minutes"], errors="coerce").eq(horizon) &
                    alpha_df["configuration"].astype(str).eq(candidate)
                ]
                if len(row) != 1:
                    raise ValueError(
                        f"Expected exactly one Step 38 alpha row for {target}/{horizon}/{candidate}; found {len(row)}"
                    )
                alpha = float(row.iloc[0]["selected_alpha"])
                features = configs[candidate][target]
                model = make_model(alpha)
                model.fit(tr[features], tr[future_col])
                prediction = model.predict(va[features])
            else:
                prediction = va[origin_col].to_numpy(dtype=float)

            actual = va[future_col].to_numpy(dtype=float)
            metrics = calculate_metrics(actual, prediction)
            metric_rows.append({
                "target": target, "horizon_minutes": horizon,
                "frozen_model": candidate, "alpha": alpha,
                "fit_samples": len(tr), "test_samples": len(va),
                **metrics
            })
            freeze_rows.append({
                "target": target, "horizon_minutes": horizon,
                "frozen_model": candidate, "alpha": alpha,
                "feature_count": len(features), "features": ";".join(features),
                "selection_basis": "Step 38 VALIDATION minimum MAE; frozen before TEST"
            })

            pf = va[[
                "origin_source_row_number", "origin_timestamp_namibia",
                "target_timestamp_namibia", "requested_horizon_minutes"
            ]].copy()
            pf["target"] = target
            pf["frozen_model"] = candidate
            pf["actual"] = actual
            pf["prediction"] = prediction
            pf["error"] = prediction - actual
            prediction_frames.append(pf)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    freeze_df = pd.DataFrame(freeze_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT / "01_final_test_metrics.csv", index=False)
    predictions_df.to_csv(OUT / "02_final_test_predictions.csv", index=False)
    freeze_df.to_csv(OUT / "03_frozen_model_configurations.csv", index=False)
    coverage_df.to_csv(OUT / "04_final_test_sample_coverage.csv", index=False)

    report = [
        "BOREMAN STEP 39 - FINAL UNTOUCHED TEST EVALUATION",
        "=" * 78,
        f"Supervised input rows read: {input_rows}",
        f"TRAIN supervised rows: {len(train)}",
        f"VALIDATION supervised rows: {len(validation)}",
        f"TEST supervised rows: {len(test)}",
        f"Training rows after target-time purge: {len(fit_data)}",
        f"TRAIN dates verified: {', '.join(train_dates)}",
        f"VALIDATION dates verified: {', '.join(validation_dates)}",
        f"Observed TEST origin dates (sparse supervised coverage): {', '.join(test_dates)}",
        "",
        "PROTOCOL",
        "- Candidate model per target/horizon was frozen using Step 38 VALIDATION minimum MAE.",
        "- Ridge alpha values are carried forward from Step 38 inner-TRAIN selection.",
        "- Final candidate Ridge models are refit on TRAIN + VALIDATION rows only.",
        "- Training rows with target timestamps at or after the first TEST origin are purged.",
        "- Imputation and standardisation are fitted on pre-TEST fitting data only.",
        "- TEST is evaluated once; no feature, alpha or model selection is performed using TEST results.",
        "- TEST supervised origin dates may be sparse because only valid future target pairs are represented.",
        "- Persistence is evaluated as the frozen baseline where selected.",
        "",
        "FROZEN CANDIDATES",
    ]
    for (target, horizon), model_name in FROZEN_CANDIDATES.items():
        report.append(f"- {target}, {horizon} min: {model_name}")
    report += [
        "",
        "INTERPRETATION",
        "- Lower MAE/RMSE is better; higher R-squared is better.",
        "- Test metrics estimate performance on this held-out time period, not universal future performance.",
        "- Do not tune or replace models based on these TEST results and then report the same TEST metrics as unbiased.",
        "- Sensor forecasts are not water-safety classifications or laboratory validation.",
        "",
        "OUTPUTS",
        "- 01_final_test_metrics.csv",
        "- 02_final_test_predictions.csv",
        "- 03_frozen_model_configurations.csv",
        "- 04_final_test_sample_coverage.csv",
        "- 05_step39_final_test_report.txt",
    ]
    (OUT / "05_step39_final_test_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("BOREMAN STEP 39 - FINAL UNTOUCHED TEST EVALUATION")
    print("=" * 66)
    print(f"TRAIN rows: {len(train):,}")
    print(f"VALIDATION rows: {len(validation):,}")
    print(f"TEST rows: {len(test):,}")
    print(f"Pre-TEST fitting rows after purge: {len(fit_data):,}")
    print(f"Metrics generated: {len(metrics_df):,}")
    print(f"Prediction rows generated: {len(predictions_df):,}")
    print(f"Output folder: {OUT.resolve()}")
    print("Candidate configurations frozen before TEST evaluation.")

if __name__ == "__main__":
    main()
