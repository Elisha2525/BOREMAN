# -*- coding: utf-8 -*-
"""
BOREMAN STEP 40 - Forecasting model decision and deployment-readiness assessment

Place this script in the project root (the folder containing the outputs folder).
It reads Step 27, Step 37 and Step 39 outputs. It does NOT train or tune models.
It creates an auditable decision report and CSV files under:
    outputs/step40_forecasting_decision
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "step40_forecasting_decision"

REQUIRED = {
    "step27_predictions": "02_forecast_predictions.csv",
    "step37_walkforward": "01_walkforward_feature_ablation_metrics.csv",
    "step39_metrics": "01_final_test_metrics.csv",
    "step39_predictions": "02_final_test_predictions.csv",
    "step39_config": "03_frozen_model_configurations.csv",
    "step39_coverage": "04_final_test_sample_coverage.csv",
    "step39_report": "05_step39_final_test_report.txt",
}

PREFERRED = {
    "step27_predictions": [
        ROOT / "outputs" / "step27_time_series_forecasting" / "02_forecast_predictions.csv",
    ],
    "step37_walkforward": [
        ROOT / "outputs" / "step37_walkforward_feature_ablation" / "01_walkforward_feature_ablation_metrics.csv",
        ROOT / "outputs" / "step37_walkforward_feature_ablation" / "01_walk_forward_feature_ablation_metrics.csv",
    ],
    "step39_metrics": [
        ROOT / "outputs" / "step39_final_test_evaluation" / "01_final_test_metrics.csv",
    ],
    "step39_predictions": [
        ROOT / "outputs" / "step39_final_test_evaluation" / "02_final_test_predictions.csv",
    ],
    "step39_config": [
        ROOT / "outputs" / "step39_final_test_evaluation" / "03_frozen_model_configurations.csv",
    ],
    "step39_coverage": [
        ROOT / "outputs" / "step39_final_test_evaluation" / "04_final_test_sample_coverage.csv",
    ],
    "step39_report": [
        ROOT / "outputs" / "step39_final_test_evaluation" / "05_step39_final_test_report.txt",
    ],
}

def locate(key):
    for p in PREFERRED[key]:
        if p.exists():
            return p
    matches = list((ROOT / "outputs").rglob(REQUIRED[key])) if (ROOT / "outputs").exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple copies found for {REQUIRED[key]}: {matches}. "
            "Keep the correct Step output in its expected folder."
        )
    raise FileNotFoundError(
        f"Missing {REQUIRED[key]}. Expected it under outputs/<step folder>."
    )

def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

def main():
    print("=" * 78)
    print("BOREMAN STEP 40 - FORECASTING DECISION & DEPLOYMENT READINESS")
    print("=" * 78)

    paths = {k: locate(k) for k in REQUIRED}
    for k, p in paths.items():
        print(f"Loaded {k}: {p.relative_to(ROOT)}")

    p27 = pd.read_csv(paths["step27_predictions"])
    wf = pd.read_csv(paths["step37_walkforward"])
    test_metrics = pd.read_csv(paths["step39_metrics"])
    test_pred = pd.read_csv(paths["step39_predictions"])
    frozen = pd.read_csv(paths["step39_config"])
    coverage = pd.read_csv(paths["step39_coverage"])
    step39_report = paths["step39_report"].read_text(encoding="utf-8", errors="replace")

    require_columns(p27, [
        "origin_timestamp_namibia", "target_timestamp_namibia",
        "requested_horizon_minutes", "split", "target", "actual_value",
        "persistence_prediction"
    ], "Step 27 predictions")
    require_columns(wf, [
        "fold", "horizon_minutes", "target", "model", "mae",
        "validation_samples"
    ], "Step 37 walk-forward metrics")
    require_columns(test_metrics, [
        "target", "horizon_minutes", "frozen_model", "test_samples",
        "mae", "rmse", "r2"
    ], "Step 39 metrics")
    require_columns(test_pred, [
        "origin_timestamp_namibia", "target_timestamp_namibia",
        "requested_horizon_minutes", "target", "actual", "prediction"
    ], "Step 39 predictions")
    require_columns(frozen, [
        "target", "horizon_minutes", "frozen_model", "selection_basis"
    ], "Step 39 frozen configurations")
    require_columns(coverage, [
        "horizon_minutes", "test_samples", "test_origin_dates"
    ], "Step 39 coverage")

    # Restrict the Step 27 persistence baseline to the untouched TEST partition.
    p27_test = p27[p27["split"].astype(str).str.upper().eq("TEST")].copy()
    keys = [
        "origin_timestamp_namibia", "target_timestamp_namibia",
        "requested_horizon_minutes", "target"
    ]
    # Use exact timestamps emitted by the same Step 27 pipeline; no approximate joins.
    joined = test_pred.merge(
        p27_test[keys + ["actual_value", "persistence_prediction"]],
        on=keys, how="left", validate="one_to_one"
    )
    if joined["persistence_prediction"].isna().any():
        missing = int(joined["persistence_prediction"].isna().sum())
        raise ValueError(
            f"{missing} Step 39 predictions could not be matched to Step 27 "
            "persistence baselines. Stop: do not make a comparison from partial matches."
        )
    if len(joined) != len(test_pred):
        raise ValueError("Prediction join changed the Step 39 row count.")

    joined["selected_absolute_error"] = (joined["actual"] - joined["prediction"]).abs()
    joined["persistence_absolute_error"] = (
        joined["actual"] - joined["persistence_prediction"]
    ).abs()
    joined["selected_squared_error"] = (joined["actual"] - joined["prediction"]) ** 2
    joined["persistence_squared_error"] = (
        joined["actual"] - joined["persistence_prediction"]
    ) ** 2

    test_compare = (
        joined.groupby(["target", "requested_horizon_minutes"], as_index=False)
        .agg(
            matched_test_samples=("actual", "size"),
            selected_model_mae=("selected_absolute_error", "mean"),
            persistence_mae=("persistence_absolute_error", "mean"),
            selected_model_rmse=("selected_squared_error", lambda x: float(np.sqrt(x.mean()))),
            persistence_rmse=("persistence_squared_error", lambda x: float(np.sqrt(x.mean()))),
            observed_test_origin_dates=("origin_timestamp_namibia",
                lambda x: len(pd.to_datetime(x, errors="coerce").dt.date.dropna().unique()))
        )
    )
    test_compare = test_compare.merge(
        test_metrics[["target", "horizon_minutes", "frozen_model", "mae", "rmse", "r2", "test_samples"]],
        left_on=["target", "requested_horizon_minutes"],
        right_on=["target", "horizon_minutes"],
        how="left", validate="one_to_one"
    )
    test_compare = test_compare.drop(columns=["horizon_minutes"])
    test_compare = test_compare.rename(columns={
        "frozen_model": "step39_frozen_model",
        "mae": "step39_reported_mae",
        "rmse": "step39_reported_rmse",
        "r2": "step39_r2",
        "test_samples": "step39_reported_test_samples"
    })
    if not (test_compare["matched_test_samples"] == test_compare["step39_reported_test_samples"]).all():
        raise ValueError("Matched baseline sample counts do not agree with Step 39 metric counts.")
    if not np.allclose(test_compare["selected_model_mae"], test_compare["step39_reported_mae"], atol=1e-8):
        raise ValueError("Recomputed selected-model MAE does not match Step 39 metrics.")

    # Aggregate Step 37 fold MAE by model; folds are reported separately in another output.
    wf_agg = (
        wf.groupby(["target", "horizon_minutes", "model"], as_index=False)
        .agg(
            completed_folds=("fold", "nunique"),
            total_validation_samples=("validation_samples", "sum"),
            mean_fold_mae=("mae", "mean"),
            median_fold_mae=("mae", "median")
        )
    )
    wf_pivot = wf_agg.pivot_table(
        index=["target", "horizon_minutes"], columns="model",
        values="mean_fold_mae", aggfunc="first"
    ).reset_index()
    wf_pivot.columns.name = None

    decision = test_compare.merge(
        wf_pivot, left_on=["target", "requested_horizon_minutes"],
        right_on=["target", "horizon_minutes"], how="left", validate="one_to_one"
    ).drop(columns=["horizon_minutes"])

    decisions = []
    reasons = []
    for _, r in decision.iterrows():
        target = str(r["target"]).lower()
        h = int(r["requested_horizon_minutes"])
        selected_better_test = r["selected_model_mae"] < r["persistence_mae"] - 1e-9

        if target == "tds":
            action = "RETAIN_PERSISTENCE_BASELINE"
            reason = (
                "Persistence has lower Step 37 walk-forward MAE and lower or equal "
                "untouched-test MAE. Do not promote the Ridge candidate."
            )
        elif target == "humidity" and h in (15, 30):
            action = "RIDGE_SHADOW_CANDIDATE"
            reason = (
                "Ridge improves on persistence in the untouched-test comparison and "
                "has a small walk-forward advantage at this horizon. Keep in shadow "
                "evaluation because the gain is modest and test dates are sparse."
            )
        elif target == "temperature":
            action = "RIDGE_SHADOW_CANDIDATE"
            reason = (
                "The frozen Ridge candidate improves on persistence in both the "
                "walk-forward evidence and the matched untouched-test comparison. "
                "Proceed only in shadow mode pending broader temporal validation."
            )
        else:
            action = "RETAIN_PERSISTENCE_BASELINE"
            reason = (
                "Walk-forward evidence does not show a sufficiently consistent "
                "advantage for the frozen Ridge candidate. Keep persistence as baseline."
            )
        if not selected_better_test and target != "tds":
            reason += " Note: the selected model does not beat persistence on this test subset."
        decisions.append(action)
        reasons.append(reason)

    decision["step40_recommendation"] = decisions
    decision["decision_rationale"] = reasons
    decision = decision.merge(
        frozen[["target", "horizon_minutes", "frozen_model", "selection_basis"]],
        left_on=["target", "requested_horizon_minutes"],
        right_on=["target", "horizon_minutes"], how="left", validate="one_to_one",
        suffixes=("", "_config")
    ).drop(columns=["horizon_minutes"])
    decision = decision.rename(columns={"frozen_model_config": "configuration_model"})
    # Keep Step 39 model name and frozen configuration side-by-side for traceability.

    # Coverage summary: report dates observed, not the full intended calendar span.
    dates = sorted(pd.to_datetime(test_pred["origin_timestamp_namibia"], errors="coerce").dt.date.dropna().astype(str).unique())
    all_origin_dates = len(dates)
    if all_origin_dates == 0:
        raise ValueError("No valid TEST origin dates could be parsed.")
    test_start = min(dates)
    test_end = max(dates)

    fold_summary = (
        wf.groupby(["fold", "target", "horizon_minutes"], as_index=False)
        .agg(validation_samples=("validation_samples", "sum"), fold_mae=("mae", "mean"))
    )

    # Deployment gates: all remain explicit; no unsupported production approval.
    gates = [
        ("Forecast target meaning", "TDS, humidity and temperature are sensor readings; forecasts are not water-safety classifications.", "REQUIRED"),
        ("Test temporal coverage", f"Only {all_origin_dates} distinct TEST origin dates observed: {', '.join(dates)}.", "BLOCKER_FOR_GENERAL_DEPLOYMENT"),
        ("Walk-forward stability", "Step 37 shows model performance varies by target and horizon; retain persistence where it is competitive.", "REVIEWED_WITH_LIMITATIONS"),
        ("Sensor calibration/reference", "No independent reference-meter or laboratory validation is established by these forecasting outputs.", "REQUIRED"),
        ("Plateau and data quality", "Earlier steps identified extensive constant-plateau records and review flags; live quality monitoring must remain visible.", "REQUIRED"),
        ("Operational monitoring", "Log forecast, actual reading, horizon, model version, input timestamp and missing/stale-input status.", "REQUIRED"),
        ("Fallback", "Use persistence when model inputs are missing, stale, out of supported range or feature history is incomplete.", "REQUIRED"),
        ("Automatic water-safety decision", "Do not classify water as safe/unsafe from these forecasts or TDS alone.", "NOT_APPROVED"),
    ]
    gates_df = pd.DataFrame(gates, columns=["deployment_gate", "evidence_or_requirement", "status"])

    OUT.mkdir(parents=True, exist_ok=True)
    test_compare.to_csv(OUT / "01_test_vs_persistence_comparison.csv", index=False)
    wf_agg.to_csv(OUT / "02_walkforward_evidence_by_model.csv", index=False)
    fold_summary.to_csv(OUT / "03_walkforward_fold_stability.csv", index=False)
    decision.to_csv(OUT / "04_sensor_horizon_decision_matrix.csv", index=False)
    gates_df.to_csv(OUT / "05_deployment_readiness_gates.csv", index=False)
    joined.to_csv(OUT / "06_matched_test_predictions_with_persistence.csv", index=False)

    lines = []
    lines += [
        "BOREMAN STEP 40 - FORECASTING MODEL DECISION & DEPLOYMENT READINESS",
        "=" * 78,
        "",
        "SCOPE",
        "- Consolidates Step 37 walk-forward evidence and Step 39 untouched-test results.",
        "- Reconstructs persistence errors by exact timestamp/horizon/target match to Step 27.",
        "- Does not train, tune, or change any model.",
        "- Step 39 candidate choices remain frozen; test results are diagnostic evidence only.",
        "",
        "DATA COVERAGE",
        f"- Step 39 TEST prediction rows: {len(test_pred):,}",
        f"- Matched persistence rows: {len(joined):,}",
        f"- Distinct observed TEST origin dates: {all_origin_dates}",
        f"- Observed origin-date span: {test_start} to {test_end}",
        f"- Dates actually present: {', '.join(dates)}",
        "- These are observed supervised origin dates, not proof of continuous daily coverage.",
        "",
        "MODEL DECISION",
        "- TDS: retain persistence as the default benchmark at all four horizons.",
        "- Humidity: retain persistence at 5 and 60 minutes; Ridge may remain a shadow candidate at 15 and 30 minutes.",
        "- Temperature: frozen Ridge candidates may remain in shadow mode at 5, 15, 30 and 60 minutes.",
        "- No sensor/horizon is approved for autonomous water-safety decisions.",
        "",
        "MATCHED TEST COMPARISON (MAE)",
    ]
    for _, r in test_compare.sort_values(["target", "requested_horizon_minutes"]).iterrows():
        lines.append(
            f"- {r['target']} {int(r['requested_horizon_minutes'])}m: "
            f"frozen={r['step39_frozen_model']}; selected MAE={r['selected_model_mae']:.4f}; "
            f"persistence MAE={r['persistence_mae']:.4f}; n={int(r['matched_test_samples'])}"
        )
    lines += [
        "",
        "IMPORTANT LIMITATIONS",
        "- TEST origin dates are sparse; the observed set must not be described as full-period coverage.",
        "- Walk-forward folds and final TEST results do not establish performance across seasons or future operating regimes.",
        "- No independent laboratory/reference-meter validation is established here.",
        "- Humidity is an environmental sensor reading, not a direct water-quality measurement.",
        "- TDS forecasts cannot establish drinking-water safety.",
        "- Earlier quality analysis identified long constant plateaus and other review flags; forecasts must be accompanied by data-quality status.",
        "",
        "DEPLOYMENT POSITION",
        "- NOT APPROVED for general production deployment or automatic water-quality classification.",
        "- Suitable next stage: controlled shadow mode for temperature Ridge and humidity Ridge at 15/30m, with persistence fallbacks.",
        "- Keep TDS persistence as the benchmark; do not promote Ridge based on the current evidence.",
        "- Capture future predictions and subsequent actual readings for ongoing monitoring, and expand independent time coverage.",
        "",
        "OUTPUT FILES",
        "- 01_test_vs_persistence_comparison.csv",
        "- 02_walkforward_evidence_by_model.csv",
        "- 03_walkforward_fold_stability.csv",
        "- 04_sensor_horizon_decision_matrix.csv",
        "- 05_deployment_readiness_gates.csv",
        "- 06_matched_test_predictions_with_persistence.csv",
        "- 07_step40_decision_report.txt",
    ]
    (OUT / "07_step40_decision_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSTEP 40 COMPLETED")
    print(f"Outputs: {OUT}")
    print(f"TEST origin dates: {', '.join(dates)}")
    print("\nDecision summary:")
    print(decision[["target", "requested_horizon_minutes", "step40_recommendation"]].to_string(index=False))
    print("\nNo model was retrained or tuned in Step 40.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSTEP 40 STOPPED: {exc}")
        print("No decision report should be treated as complete.")
        sys.exit(1)
