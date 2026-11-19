
"""
BOREMAN STEP 26 - ISOLATION FOREST ANOMALY DETECTION

Purpose:
Train an unsupervised Isolation Forest using ONLY primary-candidate
TRAIN records. Score validation, test and review records separately.

Input:
outputs/step25B_ai_feature_engineering/
01_all_records_with_engineered_features.csv

Outputs:
outputs/step26_anomaly_detection/
    01_all_records_with_anomaly_scores.csv
    02_primary_candidate_anomaly_results.csv
    03_anomaly_summary_by_split.csv
    04_anomaly_summary_by_readiness_group.csv
    05_top_100_anomaly_review_records.csv
    06_model_configuration.csv
    07_step26_anomaly_detection_report.txt

Important:
- No source records are deleted.
- No missing features are imputed.
- Model fitting uses only primary TRAIN records.
- Validation and test records are scored, not used for fitting.
- Secondary and plateau records are scored for exploratory review.
- An anomaly alert is not proof of a sensor fault or unsafe water.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


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
    / "step26_anomaly_detection"
)

PRIMARY_GROUP = "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS"
TRAIN_SPLIT = "TRAIN"

FEATURES = [
    "tds_current",
    "humidity_current",
    "Temperature_current",
    "tds_deviation_from_previous_5_mean",
    "humidity_deviation_from_previous_5_mean",
    "Temperature_deviation_from_previous_5_mean",
]

RANDOM_STATE = 42
N_ESTIMATORS = 300

# Initial modelling threshold, not measured real-world prevalence.
CONTAMINATION = 0.01


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def require_columns(df, columns):

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def safe_write_csv(df, path):

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 3. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("BOREMAN STEP 26 - ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 65)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
            "Run Step 25B first and execute this script "
            "from the project root folder."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nReading Step 25B dataset...")

    df = pd.read_csv(INPUT_FILE)

    original_count = len(df)

    # --------------------------------------------------------
    # 3.1 REQUIRED COLUMN CHECK
    # --------------------------------------------------------

    required_columns = [
        "source_row_number",
        "timestamp_namibia",
        "modelling_readiness_group",
        "proposed_temporal_split",
        "tds",
        "humidity",
        "Temperature",
        *FEATURES,
    ]

    require_columns(
        df,
        required_columns
    )

    if df["source_row_number"].isna().any():

        raise AssertionError(
            "source_row_number contains missing values."
        )

    if df["source_row_number"].duplicated().any():

        raise AssertionError(
            "source_row_number is not unique."
        )

    # --------------------------------------------------------
    # 3.2 NUMERIC FEATURE VALIDATION
    # --------------------------------------------------------

    for column in FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    complete_features = (
        df[FEATURES]
        .notna()
        .all(axis=1)
    )

    # --------------------------------------------------------
    # 3.3 SELECT TRAINING DATA
    # --------------------------------------------------------

    train_mask = (
        df["modelling_readiness_group"].eq(
            PRIMARY_GROUP
        )
        &
        df["proposed_temporal_split"].eq(
            TRAIN_SPLIT
        )
        &
        complete_features
    )

    train_count = int(
        train_mask.sum()
    )

    if train_count < 100:

        raise ValueError(
            f"Only {train_count} complete primary TRAIN "
            "records are available. At least 100 are required."
        )

    # Explicit safeguards against accidental data leakage.

    if not df.loc[
        train_mask,
        "proposed_temporal_split"
    ].eq("TRAIN").all():

        raise AssertionError(
            "Training mask includes non-TRAIN records."
        )

    if not df.loc[
        train_mask,
        "modelling_readiness_group"
    ].eq(PRIMARY_GROUP).all():

        raise AssertionError(
            "Training mask includes non-primary records."
        )

    print(f"\nTotal input records: {original_count:,}")

    print(
        "Complete primary TRAIN records: "
        f"{train_count:,}"
    )

    print(
        f"Number of model features: {len(FEATURES)}"
    )

    print(
        f"Initial contamination setting: "
        f"{CONTAMINATION:.1%}"
    )

    # --------------------------------------------------------
    # 3.4 INITIALISE ISOLATION FOREST
    # --------------------------------------------------------

    model = IsolationForest(

        n_estimators=N_ESTIMATORS,

        contamination=CONTAMINATION,

        max_samples="auto",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    print("\nTraining Isolation Forest...")

    model.fit(
        df.loc[
            train_mask,
            FEATURES
        ]
    )

    print("Model training completed.")

    # --------------------------------------------------------
    # 3.5 SCORE ALL RECORDS WITH COMPLETE FEATURES
    # --------------------------------------------------------

    result = df.copy()

    result["anomaly_model_status"] = (
        "NOT_SCORED_INSUFFICIENT_FEATURES"
    )

    result["anomaly_score"] = np.nan

    result["anomaly_decision_score"] = np.nan

    result["anomaly_prediction"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["anomaly_alert"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean"
    )

    result["anomaly_threshold_score"] = (
        -float(model.offset_)
    )

    score_mask = complete_features

    score_values = result.loc[
        score_mask,
        FEATURES
    ]

    print("\nScoring complete-feature records...")

    score_samples = model.score_samples(
        score_values
    )

    decision_scores = model.decision_function(
        score_values
    )

    predictions = model.predict(
        score_values
    )

    # Higher anomaly_score means more unusual.
    result.loc[
        score_mask,
        "anomaly_score"
    ] = -score_samples

    result.loc[
        score_mask,
        "anomaly_decision_score"
    ] = decision_scores

    result.loc[
        score_mask,
        "anomaly_prediction"
    ] = predictions.astype(int)

    result.loc[
        score_mask,
        "anomaly_alert"
    ] = predictions == -1

    result.loc[
        score_mask,
        "anomaly_model_status"
    ] = "SCORED"

    # --------------------------------------------------------
    # 3.6 IDENTIFY SCORING ROLE
    # --------------------------------------------------------

    result["anomaly_scoring_role"] = (
        "EXPLORATORY_REVIEW_GROUP"
    )

    primary_mask = result[
        "modelling_readiness_group"
    ].eq(PRIMARY_GROUP)

    result.loc[
        primary_mask
        &
        result["proposed_temporal_split"].eq("TRAIN"),
        "anomaly_scoring_role"
    ] = "TRAINING_BASELINE"

    result.loc[
        primary_mask
        &
        result["proposed_temporal_split"].eq("VALIDATION"),
        "anomaly_scoring_role"
    ] = "VALIDATION"

    result.loc[
        primary_mask
        &
        result["proposed_temporal_split"].eq("TEST"),
        "anomaly_scoring_role"
    ] = "TEST"

    result.loc[
        ~primary_mask,
        "anomaly_scoring_role"
    ] = "EXPLORATORY_REVIEW_GROUP"

    # --------------------------------------------------------
    # 3.7 DATA INTEGRITY CHECKS
    # --------------------------------------------------------

    if len(result) != original_count:

        raise AssertionError(
            "Output row count changed."
        )

    if result[
        "source_row_number"
    ].nunique() != original_count:

        raise AssertionError(
            "Source row identity check failed."
        )

    if set(
        result["source_row_number"]
    ) != set(
        df["source_row_number"]
    ):

        raise AssertionError(
            "Source row set changed."
        )

    if result.loc[
        ~score_mask,
        "anomaly_prediction"
    ].notna().any():

        raise AssertionError(
            "Rows without complete features received predictions."
        )

    print("Data preservation checks passed.")

    # --------------------------------------------------------
    # 3.8 PRIMARY CANDIDATE RESULTS
    # --------------------------------------------------------

    primary_results = result.loc[
        result["modelling_readiness_group"].eq(
            PRIMARY_GROUP
        )
    ].copy()

    # --------------------------------------------------------
    # 3.9 SUMMARY BY TEMPORAL SPLIT
    # --------------------------------------------------------

    split_summary = (
        primary_results
        .groupby(
            "proposed_temporal_split",
            dropna=False
        )
        .agg(
            record_count=(
                "source_row_number",
                "size"
            ),

            scored_count=(
                "anomaly_model_status",
                lambda s: int(
                    (s == "SCORED").sum()
                )
            ),

            alert_count=(
                "anomaly_alert",
                lambda s: int(
                    (s == True).sum()
                )
            ),

            mean_anomaly_score=(
                "anomaly_score",
                "mean"
            ),

            median_anomaly_score=(
                "anomaly_score",
                "median"
            )
        )
        .reset_index()
    )

    split_summary[
        "alert_rate_percent_of_scored"
    ] = np.where(

        split_summary["scored_count"] > 0,

        (
            split_summary["alert_count"]
            /
            split_summary["scored_count"]
        ) * 100,

        np.nan
    )

    # --------------------------------------------------------
    # 3.10 SUMMARY BY READINESS GROUP
    # --------------------------------------------------------

    group_summary = (
        result
        .groupby(
            "modelling_readiness_group",
            dropna=False
        )
        .agg(
            record_count=(
                "source_row_number",
                "size"
            ),

            scored_count=(
                "anomaly_model_status",
                lambda s: int(
                    (s == "SCORED").sum()
                )
            ),

            alert_count=(
                "anomaly_alert",
                lambda s: int(
                    (s == True).sum()
                )
            ),

            mean_anomaly_score=(
                "anomaly_score",
                "mean"
            ),

            median_anomaly_score=(
                "anomaly_score",
                "median"
            )
        )
        .reset_index()
    )

    group_summary[
        "alert_rate_percent_of_scored"
    ] = np.where(

        group_summary["scored_count"] > 0,

        (
            group_summary["alert_count"]
            /
            group_summary["scored_count"]
        ) * 100,

        np.nan
    )

    # --------------------------------------------------------
    # 3.11 TOP 100 ANOMALY REVIEW RECORDS
    # --------------------------------------------------------

    top_review = (
        result.loc[
            result["anomaly_model_status"].eq("SCORED")
        ]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(100)
        .copy()
    )

    # --------------------------------------------------------
    # 3.12 MODEL CONFIGURATION
    # --------------------------------------------------------

    configuration = pd.DataFrame(

        [
            ["algorithm", "IsolationForest"],

            ["training_group", PRIMARY_GROUP],

            ["training_split", TRAIN_SPLIT],

            ["training_rows_complete", train_count],

            ["n_estimators", N_ESTIMATORS],

            ["contamination", CONTAMINATION],

            ["random_state", RANDOM_STATE],

            ["max_samples", "auto"],

            ["features", ", ".join(FEATURES)],

            [
                "anomaly_score_definition",
                "Negative of sklearn score_samples; "
                "higher means more unusual"
            ],

            [
                "alert_rule",
                "Isolation Forest prediction == -1"
            ],

            [
                "threshold_score",
                -float(model.offset_)
            ],

            [
                "ground_truth_available",
                "No verified fault labels"
            ],
        ],

        columns=[
            "setting",
            "value"
        ]
    )

    # --------------------------------------------------------
    # 3.13 SAVE OUTPUT FILES
    # --------------------------------------------------------

    print("\nSaving Step 26 outputs...")

    safe_write_csv(
        result,
        OUTPUT_DIR
        / "01_all_records_with_anomaly_scores.csv"
    )

    safe_write_csv(
        primary_results,
        OUTPUT_DIR
        / "02_primary_candidate_anomaly_results.csv"
    )

    safe_write_csv(
        split_summary,
        OUTPUT_DIR
        / "03_anomaly_summary_by_split.csv"
    )

    safe_write_csv(
        group_summary,
        OUTPUT_DIR
        / "04_anomaly_summary_by_readiness_group.csv"
    )

    safe_write_csv(
        top_review,
        OUTPUT_DIR
        / "05_top_100_anomaly_review_records.csv"
    )

    safe_write_csv(
        configuration,
        OUTPUT_DIR
        / "06_model_configuration.csv"
    )

    # --------------------------------------------------------
    # 3.14 GENERATE REPORT
    # --------------------------------------------------------

    scored_count = int(
        score_mask.sum()
    )

    unscored_count = (
        original_count - scored_count
    )

    primary_alerts = int(
        primary_results[
            "anomaly_alert"
        ].fillna(False).sum()
    )

    total_alerts = int(
        result[
            "anomaly_alert"
        ].fillna(False).sum()
    )

    report_lines = [

        "BOREMAN STEP 26 - ISOLATION FOREST "
        "ANOMALY DETECTION REPORT",

        "=" * 68,

        f"Input records: {original_count:,}",

        f"Output records: {len(result):,}",

        (
            "Unique source row numbers: "
            f"{result['source_row_number'].nunique():,}"
        ),

        f"Complete-feature records scored: {scored_count:,}",

        (
            "Records not scored due to missing model features: "
            f"{unscored_count:,}"
        ),

        "",

        "MODEL FITTING",

        "Algorithm: IsolationForest",

        f"Training group: {PRIMARY_GROUP}",

        f"Training split: {TRAIN_SPLIT}",

        f"Complete training records: {train_count:,}",

        f"Features: {', '.join(FEATURES)}",

        f"Estimators: {N_ESTIMATORS}",

        f"Contamination: {CONTAMINATION:.4f}",

        f"Random state: {RANDOM_STATE}",

        (
            "Anomaly score threshold: "
            f"{-float(model.offset_):.10f}"
        ),

        "",

        "SCORING SUMMARY",

        f"Alerts across all scored groups: {total_alerts:,}",

        f"Alerts within primary candidates: {primary_alerts:,}",

        "",

        "INTERPRETATION AND LIMITATIONS",

        (
            "- The model was fitted only on complete "
            "primary-candidate TRAIN records."
        ),

        (
            "- Primary VALIDATION and TEST records were "
            "scored, not used for fitting."
        ),

        (
            "- Secondary and plateau groups were scored "
            "for exploratory review only."
        ),

        (
            "- Primary candidates are not verified "
            "normal ground truth."
        ),

        (
            "- Anomaly alerts indicate statistical "
            "unusualness, not confirmed sensor faults."
        ),

        (
            "- No verified fault/event labels are available; "
            "precision, recall, F1 and accuracy cannot be claimed."
        ),

        (
            "- The 1% contamination setting is an initial "
            "modelling threshold, not measured anomaly prevalence."
        ),

        (
            "- No source records were deleted and no missing "
            "features were imputed."
        ),

        (
            "- All original sensor values and quality flags "
            "are retained."
        ),

        "",

        "INTEGRITY CHECKS",

        "Row count preserved: PASSED",

        "Source row identity preserved: PASSED",

        (
            "Missing-feature rows retained without predictions: "
            "PASSED"
        ),

        "Training restricted to primary TRAIN split: PASSED",
    ]

    report_path = (
        OUTPUT_DIR
        / "07_step26_anomaly_detection_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 3.15 FINAL TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STEP 26 COMPLETED")
    print("=" * 65)

    print(
        f"Rows preserved: {len(result):,}"
    )

    print(
        f"Rows scored: {scored_count:,}"
    )

    print(
        f"Rows not scored: {unscored_count:,}"
    )

    print(
        f"Alerts across scored groups: {total_alerts:,}"
    )

    print(
        f"Primary-candidate alerts: {primary_alerts:,}"
    )

    print(
        f"Output folder: {OUTPUT_DIR}"
    )

    print("\nNo records deleted or imputed.")

    print(
        "Model trained only on primary TRAIN records."
    )

    print(
        "Review generated outputs before proceeding."
    )


# ============================================================
# 4. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
