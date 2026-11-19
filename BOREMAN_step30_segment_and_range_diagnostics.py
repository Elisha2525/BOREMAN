"""BOREMAN STEP 30 - SEGMENT, SAMPLING-INTERVAL AND FORECAST-RANGE DIAGNOSTICS

Read-only diagnostic. Does not retrain models or modify any previous output.
Run from the BOREMAN project root:
    python .\\BOREMAN_step30_segment_and_range_diagnostics.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_CANDIDATES = [
    ROOT / "outputs" / "step25B_ai_feature_engineering" / "01_all_records_with_engineered_features.csv",
    ROOT / "outputs" / "step25B_feature_engineering" / "01_all_records_with_engineered_features.csv",
    ROOT / "01_all_records_with_engineered_features.csv",
]
STEP27_DIR_CANDIDATES = [
    ROOT / "outputs" / "step27_time_series_forecasting",
    ROOT,
]
OUT_DIR = ROOT / "outputs" / "step30_segment_and_range_diagnostics"
SENSORS = ["tds", "humidity", "Temperature"]
HORIZONS = [5, 15, 30, 60]


def locate(candidates, label):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {label}. Checked:\n" + "\n".join(str(p) for p in candidates)
    )


def safe_read(path):
    return pd.read_csv(path, low_memory=False)


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def error_metrics(g):
    actual = numeric(g["actual_value"])
    rf = numeric(g["random_forest_prediction"])
    persistence = numeric(g["persistence_prediction"])
    rf_err = rf - actual
    p_err = persistence - actual
    return {
        "sample_count": int(actual.notna().sum()),
        "rf_mae": float(rf_err.abs().mean()) if rf_err.notna().any() else np.nan,
        "persistence_mae": float(p_err.abs().mean()) if p_err.notna().any() else np.nan,
        "rf_mean_signed_error": float(rf_err.mean()) if rf_err.notna().any() else np.nan,
        "rf_overprediction_pct": float((rf_err > 0).mean() * 100) if rf_err.notna().any() else np.nan,
        "rf_underprediction_pct": float((rf_err < 0).mean() * 100) if rf_err.notna().any() else np.nan,
    }


def main():
    print("=" * 70)
    print("BOREMAN STEP 30 - SEGMENT AND RANGE DIAGNOSTICS")
    print("=" * 70)

    base_path = locate(BASE_CANDIDATES, "Step 25B engineered dataset")
    step27_dir = next(
        (p for p in STEP27_DIR_CANDIDATES if
         (p / "01_forecast_supervised_dataset.csv").exists() and
         (p / "02_forecast_predictions.csv").exists()),
        None
    )
    if step27_dir is None:
        raise FileNotFoundError(
            "Could not find Step 27 files 01_forecast_supervised_dataset.csv and "
            "02_forecast_predictions.csv in outputs\\step27_time_series_forecasting."
        )

    base = safe_read(base_path)
    supervised = safe_read(step27_dir / "01_forecast_supervised_dataset.csv")
    pred = safe_read(step27_dir / "02_forecast_predictions.csv")

    required_base = {
        "source_row_number", "timestamp_namibia", "modelling_readiness_group",
        "proposed_temporal_split", "feature_segment_id", "sampling_interval_minutes",
        *SENSORS
    }
    required_sup = {
        "proposed_temporal_split", "requested_horizon_minutes",
        "actual_future_tds", "actual_future_humidity", "actual_future_Temperature"
    }
    required_pred = {
        "origin_source_row_number", "split", "requested_horizon_minutes", "target",
        "actual_value", "persistence_prediction", "random_forest_prediction"
    }
    for label, frame, required in [
        ("Step 25B dataset", base, required_base),
        ("Step 27 supervised dataset", supervised, required_sup),
        ("Step 27 predictions", pred, required_pred),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use only the same primary cohort and chronological splits used in Step 27.
    primary = base[
        base["modelling_readiness_group"].astype(str).str.upper().str.contains("PRIMARY", na=False)
    ].copy()
    primary["split"] = primary["proposed_temporal_split"].astype(str).str.upper()
    primary = primary[primary["split"].isin(["TRAIN", "VALIDATION", "TEST"])].copy()
    primary["timestamp_dt"] = pd.to_datetime(primary["timestamp_namibia"], errors="coerce", utc=True)
    primary["sampling_interval_minutes"] = numeric(primary["sampling_interval_minutes"])

    # A) Summarize feature segments within split. Segment ID is a history/reset
    # control, not a predictive variable.
    seg_rows = []
    for (split, seg_id), g in primary.groupby(["split", "feature_segment_id"], dropna=False):
        g = g.sort_values("timestamp_dt")
        times = g["timestamp_dt"].dropna()
        intervals = g["sampling_interval_minutes"].dropna()
        seg_rows.append({
            "split": split,
            "feature_segment_id": seg_id,
            "record_count": len(g),
            "start_timestamp_namibia": times.min().isoformat() if len(times) else "",
            "end_timestamp_namibia": times.max().isoformat() if len(times) else "",
            "duration_minutes": (times.max() - times.min()).total_seconds() / 60 if len(times) > 1 else 0.0,
            "median_sampling_interval_minutes": intervals.median() if len(intervals) else np.nan,
            "mean_sampling_interval_minutes": intervals.mean() if len(intervals) else np.nan,
            "max_sampling_interval_minutes": intervals.max() if len(intervals) else np.nan,
            "records_with_interval_over_5min": int((intervals > 5).sum()),
            "records_with_interval_over_60min": int((intervals > 60).sum()),
        })
    segments = pd.DataFrame(seg_rows)
    segments.to_csv(OUT_DIR / "01_feature_segment_summary.csv", index=False)

    # B) Sampling interval distribution, with explicit tail counts.
    interval_rows = []
    for split, g in primary.groupby("split"):
        x = g["sampling_interval_minutes"].dropna()
        interval_rows.append({
            "split": split, "record_count": len(g), "valid_interval_count": len(x),
            "missing_interval_count": int(g["sampling_interval_minutes"].isna().sum()),
            "mean_minutes": x.mean(), "median_minutes": x.median(),
            "p90_minutes": x.quantile(.90), "p95_minutes": x.quantile(.95),
            "p99_minutes": x.quantile(.99), "maximum_minutes": x.max(),
            "count_over_2min": int((x > 2).sum()),
            "count_over_5min": int((x > 5).sum()),
            "count_over_15min": int((x > 15).sum()),
            "count_over_60min": int((x > 60).sum()),
            "pct_over_5min": float((x > 5).mean() * 100) if len(x) else np.nan,
        })
    pd.DataFrame(interval_rows).to_csv(OUT_DIR / "02_sampling_interval_diagnostics.csv", index=False)

    # Attach origin-side interval and elapsed time since feature-segment start to predictions.
    origin = primary[[
        "source_row_number", "sampling_interval_minutes", "feature_segment_id",
        "timestamp_dt", "split"
    ]].copy()
    origin = origin.rename(columns={
        "source_row_number": "origin_source_row_number",
        "sampling_interval_minutes": "origin_sampling_interval_minutes",
        "timestamp_dt": "origin_timestamp_dt",
        "split": "origin_split",
    })
    origin["feature_segment_id"] = origin["feature_segment_id"].astype(str)
    origin["origin_source_row_number"] = numeric(origin["origin_source_row_number"])
    origin = origin.sort_values("origin_timestamp_dt")
    origin["segment_elapsed_minutes"] = np.nan
    for (_, _), idx in origin.groupby(["origin_split", "feature_segment_id"], dropna=False).groups.items():
        group_times = origin.loc[idx, "origin_timestamp_dt"]
        start = group_times.min()
        origin.loc[idx, "segment_elapsed_minutes"] = (group_times - start).dt.total_seconds() / 60

    pred["origin_source_row_number"] = numeric(pred["origin_source_row_number"])
    pred = pred.merge(origin, on="origin_source_row_number", how="left", validate="many_to_one")
    pred["split"] = pred["split"].astype(str).str.upper()
    pred["rf_signed_error"] = numeric(pred["random_forest_prediction"]) - numeric(pred["actual_value"])
    pred["persistence_signed_error"] = numeric(pred["persistence_prediction"]) - numeric(pred["actual_value"])

    # C) Forecast error by origin-side sampling interval bands.
    edges = [-np.inf, 1.5, 2, 5, 15, 60, np.inf]
    labels = ["<=1.5 min", "1.5-2 min", "2-5 min", "5-15 min", "15-60 min", ">60 min"]
    pred["sampling_interval_band"] = pd.cut(
        pred["origin_sampling_interval_minutes"], bins=edges, labels=labels, include_lowest=True
    )
    rows = []
    for keys, g in pred.groupby(["split", "requested_horizon_minutes", "target", "sampling_interval_band"], observed=False, dropna=False):
        split, horizon, target, band = keys
        m = error_metrics(g)
        rows.append({"split": split, "horizon_minutes": horizon, "target": target,
                     "origin_sampling_interval_band": str(band), **m})
    pd.DataFrame(rows).to_csv(OUT_DIR / "03_forecast_error_by_origin_interval.csv", index=False)

    # D) Forecast error by time elapsed since feature-segment start.
    age_edges = [-np.inf, 5, 15, 30, 60, 180, np.inf]
    age_labels = ["<=5 min", "5-15 min", "15-30 min", "30-60 min", "1-3 hr", ">3 hr"]
    pred["segment_age_band"] = pd.cut(
        pred["segment_elapsed_minutes"], bins=age_edges, labels=age_labels, include_lowest=True
    )
    rows = []
    for keys, g in pred.groupby(["split", "requested_horizon_minutes", "target", "segment_age_band"], observed=False, dropna=False):
        split, horizon, target, band = keys
        m = error_metrics(g)
        rows.append({"split": split, "horizon_minutes": horizon, "target": target,
                     "segment_age_band": str(band), **m})
    pd.DataFrame(rows).to_csv(OUT_DIR / "04_forecast_error_by_segment_age.csv", index=False)

    # E) Corrected range audit: compare model predictions and actual future targets
    # with TRAIN future-target ranges, separately for each sensor and horizon.
    train = supervised[supervised["proposed_temporal_split"].astype(str).str.upper() == "TRAIN"].copy()
    range_rows = []
    for horizon in HORIZONS:
        train_h = train[numeric(train["requested_horizon_minutes"]) == horizon]
        for target in SENSORS:
            target_col = f"actual_future_{target}"
            if target_col not in train_h.columns:
                continue
            train_values = numeric(train_h[target_col]).dropna()
            if train_values.empty:
                continue
            train_min, train_max = train_values.min(), train_values.max()
            for split in ["VALIDATION", "TEST"]:
                g = pred[
                    (pred["split"] == split) &
                    (numeric(pred["requested_horizon_minutes"]) == horizon) &
                    (pred["target"].astype(str) == target)
                ].copy()
                if g.empty:
                    continue
                rf = numeric(g["random_forest_prediction"])
                actual = numeric(g["actual_value"])
                range_rows.append({
                    "split": split, "horizon_minutes": horizon, "target": target,
                    "training_target_sample_count": int(train_values.size),
                    "training_target_min": train_min, "training_target_max": train_max,
                    "evaluation_sample_count": int(actual.notna().sum()),
                    "rf_prediction_below_train_min_count": int((rf < train_min).sum()),
                    "rf_prediction_above_train_max_count": int((rf > train_max).sum()),
                    "rf_prediction_outside_train_range_pct": float(((rf < train_min) | (rf > train_max)).mean() * 100),
                    "actual_target_below_train_min_count": int((actual < train_min).sum()),
                    "actual_target_above_train_max_count": int((actual > train_max).sum()),
                    "actual_target_outside_train_range_pct": float(((actual < train_min) | (actual > train_max)).mean() * 100),
                    "rf_prediction_min": rf.min(), "rf_prediction_max": rf.max(),
                    "actual_target_min": actual.min(), "actual_target_max": actual.max(),
                })
    range_audit = pd.DataFrame(range_rows)
    range_audit.to_csv(OUT_DIR / "05_prediction_training_range_audit.csv", index=False)

    # F) Text report.
    lines = [
        "BOREMAN STEP 30 - SEGMENT, SAMPLING-INTERVAL AND RANGE DIAGNOSTICS",
        "=" * 76,
        f"Step 25B source rows read: {len(base):,}",
        f"Primary chronological-split records: {len(primary):,}",
        f"Step 27 supervised rows: {len(supervised):,}",
        f"Step 27 prediction rows: {len(pred):,}",
        "",
        "SEGMENT SUMMARY",
        f"Feature-segment groups within split: {len(segments):,}",
        f"Segments with duration <= 5 minutes: {int((segments['duration_minutes'] <= 5).sum()) if not segments.empty else 0:,}",
        "",
        "SAMPLING INTERVAL",
    ]
    for _, r in pd.DataFrame(interval_rows).iterrows():
        lines.append(
            f"{r['split']} | n={int(r['valid_interval_count'])} | median={r['median_minutes']:.4f} min "
            f"| mean={r['mean_minutes']:.4f} min | p99={r['p99_minutes']:.4f} min "
            f"| max={r['maximum_minutes']:.4f} min | >5min={int(r['count_over_5min'])} "
            f"| >60min={int(r['count_over_60min'])}"
        )
    lines += ["", "TEST FORECAST ERROR BY ORIGIN SAMPLING INTERVAL"]
    test_interval = pd.DataFrame(rows) if False else pd.DataFrame()
    interval_error = pd.read_csv(OUT_DIR / "03_forecast_error_by_origin_interval.csv")
    if not interval_error.empty:
        for (h, target), g in interval_error[interval_error["split"] == "TEST"].groupby(["horizon_minutes", "target"]):
            valid = g[g["sample_count"] > 0]
            if not valid.empty:
                best = valid.sort_values("rf_mae").iloc[0]
                lines.append(f"{h} min | {target} | lowest RF MAE band={best['origin_sampling_interval_band']} "
                             f"| RF MAE={best['rf_mae']:.5f} | persistence MAE={best['persistence_mae']:.5f}")
    lines += ["", "TRAINING-RANGE AUDIT"]
    if not range_audit.empty:
        for _, r in range_audit[range_audit["split"] == "TEST"].iterrows():
            lines.append(
                f"{int(r['horizon_minutes'])} min | {r['target']} | "
                f"TRAIN range={r['training_target_min']:.5f} to {r['training_target_max']:.5f} | "
                f"RF predictions outside={int(r['rf_prediction_below_train_min_count'] + r['rf_prediction_above_train_max_count'])}/"
                f"{int(r['evaluation_sample_count'])} ({r['rf_prediction_outside_train_range_pct']:.2f}%) | "
                f"actual targets outside={int(r['actual_target_below_train_min_count'] + r['actual_target_above_train_max_count'])}/"
                f"{int(r['evaluation_sample_count'])} ({r['actual_target_outside_train_range_pct']:.2f}%)"
            )
    lines += [
        "",
        "INTERPRETATION LIMITATIONS",
        "- A range exceedance is a statistical comparison, not proof of physical invalidity.",
        "- Feature-segment IDs are reset/control identifiers, not predictive features.",
        "- Segment age is measured from the earliest timestamp in each split/segment group.",
        "- Interval bands are descriptive; small groups may have unstable error estimates.",
        "- The test set is one chronological holdout, not external validation.",
        "- Forecasting sensor readings does not establish water safety.",
        "",
        "SOURCE PRESERVATION",
        "- Step 25B and Step 27 inputs were read only.",
        "- No model retraining, imputation, source-row deletion or prior-output modification occurred.",
    ]
    report_path = OUT_DIR / "06_step30_diagnostic_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nStep 30 completed.")
    print(f"Primary records: {len(primary):,}")
    print(f"Feature segment groups: {len(segments):,}")
    print(f"Prediction rows analysed: {len(pred):,}")
    print(f"Range-audit rows: {len(range_audit):,}")
    print(f"Output folder: {OUT_DIR}")
    print("Previous datasets and model outputs were not modified.")


if __name__ == "__main__":
    main()
