from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import joblib
import pandas as pd

from BOREMAN_step39_final_test_evaluation import (
    ROOT, SUPERVISED_INPUT, ALPHA_INPUT,
    TRAIN_DATES, VALIDATION_DATES, HORIZONS,
    TARGETS, FROZEN_CANDIDATES, make_feature_sets, make_model
)

OUT = ROOT / "boreman-ai-api" / "models"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SUPERVISED_INPUT, low_memory=False)
alpha_df = pd.read_csv(ALPHA_INPUT)

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
train = d[split.eq("TRAIN")]
validation = d[split.eq("VALIDATION")]
test = d[split.eq("TEST")]

if sorted(train["_origin_date"].dropna().unique()) != TRAIN_DATES:
    raise ValueError("TRAIN date verification failed.")
if sorted(validation["_origin_date"].dropna().unique()) != VALIDATION_DATES:
    raise ValueError("VALIDATION date verification failed.")
if test.empty:
    raise ValueError("TEST data is empty.")

test_start = test["_origin_ts"].min()
fit_data = d[split.isin(["TRAIN", "VALIDATION"])].copy()
fit_data = fit_data[fit_data["_target_ts"] < test_start].copy()
if fit_data.empty:
    raise ValueError("No fitting data remains after target-time purge.")

configs = make_feature_sets(d.columns.tolist())
deployment = {
    ("Temperature", 5), ("Temperature", 15),
    ("Temperature", 30), ("Temperature", 60),
    ("humidity", 5), ("humidity", 15), ("humidity", 30)
}

manifest = {
    "model_version": "BOREMAN_STEP40_SHADOW_V1",
    "timezone": "Africa/Windhoek",
    "training_protocol": "Step39 TRAIN+VALIDATION with target-time purge",
    "test_origin_dates": sorted(test["_origin_date"].dropna().unique().tolist()),
    "deployment_policy": {
        "Temperature": {"5": "RIDGE_SHADOW", "15": "RIDGE_SHADOW", "30": "RIDGE_SHADOW", "60": "RIDGE_SHADOW"},
        "humidity": {"5": "RIDGE_SHADOW", "15": "RIDGE_SHADOW", "30": "RIDGE_SHADOW", "60": "PERSISTENCE"},
        "tds": {"5": "PERSISTENCE", "15": "PERSISTENCE", "30": "PERSISTENCE", "60": "PERSISTENCE"}
    },
    "models": {}
}

for target, horizon in sorted(deployment):
    config_name = FROZEN_CANDIDATES[(target, horizon)]
    row = alpha_df[
        alpha_df["target"].astype(str).eq(target)
        & pd.to_numeric(alpha_df["horizon_minutes"], errors="coerce").eq(horizon)
        & alpha_df["configuration"].astype(str).eq(config_name)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one alpha row for {target}/{horizon}; found {len(row)}")

    alpha = float(row.iloc[0]["selected_alpha"])
    features = configs[config_name][target]
    origin_col, future_col = TARGETS[target]

    fit_h = fit_data[fit_data["requested_horizon_minutes"].eq(horizon)].copy()
    fit_h[future_col] = pd.to_numeric(fit_h[future_col], errors="coerce")
    fit_h = fit_h[fit_h[future_col].notna()].copy()
    for feature in features:
        fit_h[feature] = pd.to_numeric(fit_h[feature], errors="coerce")

    if fit_h.empty:
        raise ValueError(f"No training rows for {target}/{horizon}")

    model = make_model(alpha)
    model.fit(fit_h[features], fit_h[future_col])

    filename = f"{target.lower()}_{horizon}m_ridge.joblib"
    joblib.dump(model, OUT / filename)

    manifest["models"][f"{target}_{horizon}m"] = {
        "file": filename,
        "target": target,
        "horizon_minutes": horizon,
        "configuration": config_name,
        "alpha": alpha,
        "features": features,
        "training_samples": len(fit_h)
    }
    print(f"EXPORTED {target} {horizon}m | alpha={alpha} | rows={len(fit_h)}")

(OUT / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"\nExport complete: {OUT.resolve()}")
print(f"Ridge models exported: {len(manifest['models'])}")
print("TDS remains persistence-only. No test data was used for fitting.")

