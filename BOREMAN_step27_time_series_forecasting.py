"""
BOREMAN STEP 27 - TIME-SERIES FORECASTING

Targets: TDS, humidity, Temperature
Horizons: 5, 15, 30 and 60 minutes
Models: Persistence baseline and multi-output Random Forest.

Only PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS records are used.
Random Forest fitting uses TRAIN samples only.
VALIDATION and TEST samples are evaluated separately.

Future target matching:
- Same feature_segment_id.
- Same chronological split.
- Strictly later than the forecast origin.
- Nearest observation to the requested horizon, within +/- 90 seconds.

No missing values are imputed. The original input file is never modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path.cwd()

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "step25B_ai_feature_engineering"
    / "01_all_records_with_engineered_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "step27_time_series_forecasting"
)

PRIMARY_GROUP = "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS"
HORIZONS_MINUTES = [5, 15, 30, 60]
TARGETS = ["tds", "humidity", "Temperature"]
TARGET_TOLERANCE_SECONDS = 90

FEATURES = [
    "tds_current",
    "humidity_current",
    "Temperature_current",
    "tds_change_from_previous",
    "humidity_change_from_previous",
    "Temperature_change_from_previous",
    "tds_rate_per_minute",
    "humidity_rate_per_minute",
    "Temperature_rate_per_minute",
    "tds_lag_1",
    "tds_lag_3",
    "tds_lag_5",
    "humidity_lag_1",
    "humidity_lag_3",
    "humidity_lag_5",
    "Temperature_lag_1",
    "Temperature_lag_3",
    "Temperature_lag_5",
    "tds_rolling_mean_previous_5",
    "humidity_rolling_mean_previous_5",
    "Temperature_rolling_mean_previous_5",
    "tds_rolling_std_previous_5",
    "humidity_rolling_std_previous_5",
    "Temperature_rolling_std_previous_5",
    "tds_deviation_from_previous_5_mean",
    "humidity_deviation_from_previous_5_mean",
    "Temperature_deviation_from_previous_5_mean",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "sampling_interval_minutes",
]

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 0.8
N_JOBS = -1


# ============================================================
# 2. HELPERS
# ============================================================

def require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        return {
            "sample_count": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }

    mse = mean_squared_error(actual, predicted)

    return {
        "sample_count": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mse)),
        "r2": (
            float(r2_score(actual, predicted))
            if len(actual) >= 2
            else np.nan
        ),
    }


# ============================================================
# 3. CREATE SUPERVISED FUTURE TARGETS
# ============================================================

def build_supervised_dataset(primary):
    """
    Vectorised matching of origin observations to future targets.
    A target cannot cross a feature segment or split boundary.
    """

    parts = []
    grouped = primary.groupby(
        ["proposed_temporal_split", "feature_segment_id"],
        dropna=False,
        sort=False,
    )

    tolerance_ns = TARGET_TOLERANCE_SECONDS * 1_000_000_000

    for (split_name, segment_id), group in grouped:

        group = group.sort_values(
            ["timestamp_namibia_dt", "source_row_number"]
        ).reset_index(drop=True)

        n = len(group)
        if n < 2:
            continue

        # pandas may store parsed datetimes at microsecond resolution
        # (rather than nanoseconds). Convert explicitly to ns before
        # comparing with horizon offsets expressed in nanoseconds.
        timestamps_ns = (
            group["timestamp_namibia_dt"]
            .dt.as_unit("ns")
            .astype("int64")
            .to_numpy()
        )
        origin_positions = np.arange(n, dtype=np.int64)

        for horizon in HORIZONS_MINUTES:

            desired_ns = (
                timestamps_ns
                + horizon * 60 * 1_000_000_000
            )

            right_pos = np.searchsorted(
                timestamps_ns,
                desired_ns,
                side="left",
            )
            left_pos = right_pos - 1

            right_valid = (
                (right_pos < n)
                & (right_pos > origin_positions)
            )
            left_valid = (
                (left_pos > origin_positions)
                & (left_pos < n)
            )

            right_dist = np.full(
                n, np.iinfo(np.int64).max, dtype=np.int64
            )
            left_dist = np.full(
                n, np.iinfo(np.int64).max, dtype=np.int64
            )

            right_dist[right_valid] = np.abs(
                timestamps_ns[right_pos[right_valid]]
                - desired_ns[right_valid]
            )
            left_dist[left_valid] = np.abs(
                timestamps_ns[left_pos[left_valid]]
                - desired_ns[left_valid]
            )

            choose_left = (
                left_valid
                & (
                    ~right_valid
                    | (left_dist <= right_dist)
                )
            )

            target_pos = np.where(
                choose_left,
                left_pos,
                right_pos,
            )
            target_dist = np.where(
                choose_left,
                left_dist,
                right_dist,
            )

            valid = (
                (target_pos > origin_positions)
                & (target_pos < n)
                & (target_dist <= tolerance_ns)
            )

            if not valid.any():
                continue

            origin = group.loc[valid].reset_index(drop=True)
            target = group.iloc[target_pos[valid]].reset_index(drop=True)

            actual_horizon = (
                timestamps_ns[target_pos[valid]]
                - timestamps_ns[valid]
            ) / 60_000_000_000

            time_error = (
                timestamps_ns[target_pos[valid]]
                - desired_ns[valid]
            ) / 1_000_000_000

            part = pd.DataFrame({
                "origin_source_row_number": (
                    origin["source_row_number"].astype("int64")
                ),
                "target_source_row_number": (
                    target["source_row_number"].astype("int64")
                ),
                "origin_timestamp_namibia": (
                    origin["timestamp_namibia"].to_numpy()
                ),
                "target_timestamp_namibia": (
                    target["timestamp_namibia"].to_numpy()
                ),
                "actual_horizon_minutes": actual_horizon,
                "requested_horizon_minutes": horizon,
                "target_time_error_seconds": time_error,
                "proposed_temporal_split": split_name,
                "feature_segment_id": segment_id,
            })

            for feature in FEATURES:
                part[feature] = origin[feature].to_numpy()

            for target_name in TARGETS:
                part[f"origin_{target_name}"] = (
                    origin[target_name].to_numpy()
                )
                part[f"actual_future_{target_name}"] = (
                    target[target_name].to_numpy()
                )

            parts.append(part)

    if not parts:
        raise ValueError(
            "No future target pairs were created. Check the timestamps, "
            "segments and target tolerance."
        )

    supervised = pd.concat(parts, ignore_index=True)

    # Do not impute. Exclude only incomplete supervised pairs from modelling.
    complete_x = supervised[FEATURES].notna().all(axis=1)
    target_columns = [
        f"actual_future_{name}" for name in TARGETS
    ]
    complete_y = supervised[target_columns].notna().all(axis=1)

    return supervised.loc[complete_x & complete_y].copy()


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 68)
    print("BOREMAN STEP 27 - TIME-SERIES FORECASTING")
    print("=" * 68)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
            "Run Step 25B first and execute from the project root."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nReading Step 25B engineered dataset...")
    df = pd.read_csv(INPUT_FILE)
    original_count = len(df)

    required = [
        "source_row_number",
        "timestamp_namibia",
        "modelling_readiness_group",
        "proposed_temporal_split",
        "feature_segment_id",
        *TARGETS,
        *FEATURES,
    ]
    require_columns(df, required)

    if df["source_row_number"].isna().any():
        raise AssertionError("source_row_number contains missing values.")
    if df["source_row_number"].duplicated().any():
        raise AssertionError("source_row_number is not unique.")

    # Step 25B timestamp strings may have mixed ISO formats.
    timestamp_utc = pd.to_datetime(
        df["timestamp_namibia"],
        errors="coerce",
        utc=True,
        format="mixed",
    )
    df["timestamp_namibia_dt"] = timestamp_utc.dt.tz_convert(
        "Africa/Windhoek"
    )

    for column in FEATURES + TARGETS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    primary = df.loc[
        df["modelling_readiness_group"].eq(PRIMARY_GROUP)
    ].copy()

    if primary.empty:
        raise ValueError("No primary-candidate records found.")

    if primary["timestamp_namibia_dt"].isna().any():
        bad = int(primary["timestamp_namibia_dt"].isna().sum())
        raise ValueError(
            f"{bad} primary-candidate records have invalid Namibia timestamps."
        )

    print(f"All source records: {original_count:,}")
    print(f"Primary-candidate records: {len(primary):,}")
    print("Primary split counts:")
    print(primary["proposed_temporal_split"].value_counts().to_string())

    # --------------------------------------------------------
    # 4.1 BUILD FUTURE TARGET DATASET
    # --------------------------------------------------------

    print("\nCreating future targets for 5, 15, 30 and 60 minutes...")
    supervised = build_supervised_dataset(primary)

    if supervised.empty:
        raise ValueError("No complete forecasting samples were created.")

    if (supervised["actual_horizon_minutes"] <= 0).any():
        raise AssertionError("A forecast target is not after its origin.")

    # --------------------------------------------------------
    # 4.2 COVERAGE SUMMARY
    # --------------------------------------------------------

    coverage = (
        supervised.groupby(
            ["requested_horizon_minutes", "proposed_temporal_split"],
            dropna=False,
        )
        .agg(
            sample_count=("origin_source_row_number", "size"),
            unique_origin_records=("origin_source_row_number", "nunique"),
            mean_actual_horizon_minutes=("actual_horizon_minutes", "mean"),
            max_abs_target_time_error_seconds=(
                "target_time_error_seconds",
                lambda s: float(s.abs().max()),
            ),
        )
        .reset_index()
    )

    coverage_rows = []
    for horizon in HORIZONS_MINUTES:
        for split in ["TRAIN", "VALIDATION", "TEST"]:
            found = coverage.loc[
                coverage["requested_horizon_minutes"].eq(horizon)
                & coverage["proposed_temporal_split"].eq(split)
            ]
            if found.empty:
                coverage_rows.append({
                    "requested_horizon_minutes": horizon,
                    "proposed_temporal_split": split,
                    "sample_count": 0,
                    "unique_origin_records": 0,
                    "mean_actual_horizon_minutes": np.nan,
                    "max_abs_target_time_error_seconds": np.nan,
                })
            else:
                coverage_rows.append(found.iloc[0].to_dict())

    coverage = pd.DataFrame(coverage_rows)

    print("\nForecast sample coverage:")
    print(coverage.to_string(index=False))

    # --------------------------------------------------------
    # 4.3 MODEL TRAINING AND EVALUATION
    # --------------------------------------------------------

    predictions_parts = []
    metrics_rows = []
    importance_rows = []

    for horizon in HORIZONS_MINUTES:

        horizon_data = supervised.loc[
            supervised["requested_horizon_minutes"].eq(horizon)
        ]

        train = horizon_data.loc[
            horizon_data["proposed_temporal_split"].eq("TRAIN")
        ]
        validation = horizon_data.loc[
            horizon_data["proposed_temporal_split"].eq("VALIDATION")
        ]
        test = horizon_data.loc[
            horizon_data["proposed_temporal_split"].eq("TEST")
        ]

        if len(train) < 100 or validation.empty or test.empty:
            print(
                f"\nSkipping {horizon}-minute horizon: "
                "insufficient train/validation/test samples."
            )
            continue

        print(
            f"\nTraining {horizon}-minute multi-output Random Forest "
            f"({len(train):,} training samples)..."
        )

        X_train = train[FEATURES]
        y_train = train[
            [f"actual_future_{target}" for target in TARGETS]
        ]

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
        model.fit(X_train, y_train)

        for split_name, evaluation in [
            ("VALIDATION", validation),
            ("TEST", test),
        ]:

            rf_matrix = model.predict(evaluation[FEATURES])

            # Produce one output block per sensor target.
            for target_index, target_name in enumerate(TARGETS):

                actual = evaluation[
                    f"actual_future_{target_name}"
                ].to_numpy(dtype=float)

                rf_prediction = rf_matrix[:, target_index]

                # Baseline: hold the current reading constant.
                persistence_prediction = evaluation[
                    f"origin_{target_name}"
                ].to_numpy(dtype=float)

                rf_metrics = calculate_metrics(actual, rf_prediction)
                baseline_metrics = calculate_metrics(
                    actual,
                    persistence_prediction,
                )

                metrics_rows.append({
                    "horizon_minutes": horizon,
                    "target": target_name,
                    "split": split_name,
                    "model": "RandomForest",
                    **rf_metrics,
                })
                metrics_rows.append({
                    "horizon_minutes": horizon,
                    "target": target_name,
                    "split": split_name,
                    "model": "Persistence",
                    **baseline_metrics,
                })

                block = pd.DataFrame({
                    "origin_source_row_number": (
                        evaluation["origin_source_row_number"].to_numpy()
                    ),
                    "target_source_row_number": (
                        evaluation["target_source_row_number"].to_numpy()
                    ),
                    "origin_timestamp_namibia": (
                        evaluation["origin_timestamp_namibia"].to_numpy()
                    ),
                    "target_timestamp_namibia": (
                        evaluation["target_timestamp_namibia"].to_numpy()
                    ),
                    "requested_horizon_minutes": horizon,
                    "actual_horizon_minutes": (
                        evaluation["actual_horizon_minutes"].to_numpy()
                    ),
                    "split": split_name,
                    "target": target_name,
                    "actual_value": actual,
                    "persistence_prediction": persistence_prediction,
                    "random_forest_prediction": rf_prediction,
                    "persistence_absolute_error": np.abs(
                        actual - persistence_prediction
                    ),
                    "random_forest_absolute_error": np.abs(
                        actual - rf_prediction
                    ),
                })
                predictions_parts.append(block)

        for feature, importance in zip(
            FEATURES,
            model.feature_importances_,
        ):
            importance_rows.append({
                "horizon_minutes": horizon,
                "feature": feature,
                "importance": float(importance),
            })

    if not metrics_rows:
        raise ValueError(
            "No forecasting models were evaluated. "
            "Review sample coverage before proceeding."
        )

    predictions = pd.concat(predictions_parts, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    feature_importance = pd.DataFrame(importance_rows)

    # --------------------------------------------------------
    # 4.4 SAVE OUTPUTS
    # --------------------------------------------------------

    print("\nSaving Step 27 outputs...")

    write_csv(
        supervised,
        OUTPUT_DIR / "01_forecast_supervised_dataset.csv",
    )
    write_csv(
        predictions,
        OUTPUT_DIR / "02_forecast_predictions.csv",
    )
    write_csv(
        metrics,
        OUTPUT_DIR / "03_forecast_metrics.csv",
    )
    write_csv(
        coverage,
        OUTPUT_DIR / "04_forecast_sample_coverage.csv",
    )
    write_csv(
        feature_importance,
        OUTPUT_DIR / "05_feature_importance.csv",
    )

    configuration = pd.DataFrame([
        ["algorithm", "RandomForestRegressor"],
        ["model_strategy", "One multi-output model per forecast horizon"],
        ["baseline", "Persistence: future value equals current value"],
        ["training_split", "TRAIN only"],
        ["validation_split", "VALIDATION; evaluation only"],
        ["test_split", "TEST; held-out evaluation"],
        ["horizons_minutes", ", ".join(map(str, HORIZONS_MINUTES))],
        ["targets", ", ".join(TARGETS)],
        ["target_matching_tolerance_seconds", TARGET_TOLERANCE_SECONDS],
        ["n_estimators", N_ESTIMATORS],
        ["max_depth", MAX_DEPTH],
        ["min_samples_leaf", MIN_SAMPLES_LEAF],
        ["max_features", MAX_FEATURES],
        ["random_state", RANDOM_STATE],
        ["missing_value_policy", "No imputation; incomplete samples excluded"],
        ["segment_policy", "Future target must remain in same feature segment and split"],
    ], columns=["setting", "value"])

    write_csv(
        configuration,
        OUTPUT_DIR / "06_model_configuration.csv",
    )

    # --------------------------------------------------------
    # 4.5 REPORT
    # --------------------------------------------------------

    lines = [
        "BOREMAN STEP 27 - TIME-SERIES FORECASTING REPORT",
        "=" * 68,
        f"Source records read: {original_count:,}",
        f"Primary-candidate records: {len(primary):,}",
        f"Supervised forecast samples: {len(supervised):,}",
        f"Prediction rows: {len(predictions):,}",
        f"Forecast horizons: {HORIZONS_MINUTES}",
        f"Forecast targets: {TARGETS}",
        f"Target matching tolerance: +/- {TARGET_TOLERANCE_SECONDS} seconds",
        f"Number of input features: {len(FEATURES)}",
        "",
        "MODELS",
        "- Persistence baseline: future value equals current sensor reading.",
        "- Random Forest: one multi-output model per horizon.",
        "- Each Random Forest forecasts TDS, humidity and Temperature together.",
        "- Models are fitted using TRAIN samples only.",
        "- VALIDATION and TEST are evaluated separately.",
        "",
        "TEMPORAL SAFEGUARDS",
        "- Future targets remain within the same feature segment.",
        "- Future targets remain within the same chronological split.",
        "- Target timestamps are strictly later than origin timestamps.",
        "- Future targets must be within the configured matching tolerance.",
        "- No missing values are imputed.",
        "",
        "SCIENTIFIC LIMITATIONS",
        "- Forecasting sensor readings does not establish water safety.",
        "- Sampling is not perfectly uniform.",
        "- Target horizons use the nearest eligible future reading within tolerance.",
        "- Test performance is an initial holdout evaluation, not external validation.",
        "- Compare Random Forest results against the persistence baseline.",
        "- Negative R2 values are possible and indicate performance worse than the test-set mean baseline.",
        "",
        "METRIC DEFINITIONS",
        "- MAE: mean absolute error, in the target sensor's units.",
        "- RMSE: root mean squared error, in the target sensor's units.",
        "- R2: coefficient of determination.",
        "",
        "RESULTS",
    ]

    for _, row in metrics.iterrows():
        lines.append(
            f"{int(row['horizon_minutes'])} min | {row['target']} | "
            f"{row['split']} | {row['model']} | "
            f"n={int(row['sample_count'])} | "
            f"MAE={row['mae']:.6f} | "
            f"RMSE={row['rmse']:.6f} | "
            f"R2={row['r2']:.6f}"
        )

    lines.extend([
        "",
        "SOURCE PRESERVATION",
        f"Source records read: {original_count:,}",
        "Original Step 25B input file was not modified.",
        "Only eligible future pairs are included in the supervised modelling dataset.",
        "No source observations were deleted from the original dataset.",
    ])

    (OUTPUT_DIR / "07_step27_forecasting_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("STEP 27 COMPLETED")
    print("=" * 68)
    print(f"Primary source records: {len(primary):,}")
    print(f"Supervised forecast samples: {len(supervised):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Metric rows: {len(metrics):,}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Original Step 25B dataset was not modified.")
    print("Random Forest models were trained using TRAIN samples only.")
    print("Review generated outputs before proceeding.")


if __name__ == "__main__":
    main()
