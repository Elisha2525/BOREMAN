from pathlib import Path
import pandas as pd

# Run this script from the BOREMAN project root.
ROOT = Path(__file__).resolve().parents[1]
CLEAN_GW = ROOT / "outputs" / "cleaned" / "BOREMAN_OKAHANDJA_QUALITY_FLAGGED.csv"
SOIL_PROFILE = ROOT / "outputs" / "engineered_soil" / "02_BOREMAN_ENGINEERED_PROFILE_FEATURES.csv"
SPATIAL_LINK = ROOT / "outputs" / "engineered_soil" / "04_BOREMAN_GROUNDWATER_SOIL_FEATURE_LINK.csv"
OUT = ROOT / "outputs" / "modelling_ready"
OUT.mkdir(parents=True, exist_ok=True)

for path in [CLEAN_GW, SOIL_PROFILE, SPATIAL_LINK]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

gw = pd.read_csv(CLEAN_GW, low_memory=False)
soil = pd.read_csv(SOIL_PROFILE, low_memory=False)
link = pd.read_csv(SPATIAL_LINK, low_memory=False)

required_link = ["GW_BOREMAN_RECORD_ID", "SOIL_PRID", "DISTANCE_KM", "DISTANCE_BAND"]
required_soil = ["PRID"]
for col in required_link:
    if col not in link.columns:
        raise ValueError(f"Missing required column in spatial link: {col}")
for col in required_soil:
    if col not in soil.columns:
        raise ValueError(f"Missing required column in soil profile file: {col}")

if link["GW_BOREMAN_RECORD_ID"].duplicated().any():
    raise ValueError("Duplicate groundwater record IDs found in spatial link.")
if soil["PRID"].duplicated().any():
    raise ValueError("Duplicate PRID values found in engineered soil profiles.")
if link["SOIL_PRID"].isna().any():
    raise ValueError("Some groundwater records have no linked soil profile.")
if not link["SOIL_PRID"].isin(soil["PRID"]).all():
    raise ValueError("Some linked SOIL_PRID values are absent from soil profile data.")

# Verify overall profile means match the engineered profile table.
mean_cols = [
    c for c in soil.columns
    if c.endswith("_mean")
    and not c.endswith("_topsoil_mean")
    and not c.endswith("_subsoil_mean")
    and c in link.columns
]
soil_idx = soil.set_index("PRID")
mismatches = []
for col in mean_cols:
    expected = link["SOIL_PRID"].map(soil_idx[col])
    actual = pd.to_numeric(link[col], errors="coerce")
    expected = pd.to_numeric(expected, errors="coerce")
    bad = ~(actual.eq(expected) | (actual.isna() & expected.isna()))
    if bad.any():
        mismatches.append(f"{col}: {int(bad.sum())} mismatches")
if mismatches:
    raise ValueError("Soil feature mismatch between spatial link and profile table: " + "; ".join(mismatches))

gw_cols = [
    "GW_BOREMAN_RECORD_ID", "GW_Source", "GW_ID",
    "GW_LONGITUDE", "GW_LATITUDE", "GW_ELEVATION",
    "GW_SAMPLE_NAM", "GW_SAMPLING_D",
    "GW_TOTAL_DISS", "GW_TOTAL_DISS_NUMERIC", "GW_TOTAL_DISS_QUALITY_FLAG",
    "GW_SULPHATE_A", "GW_SULPHATE_A_NUMERIC", "GW_SULPHATE_A_QUALITY_FLAG",
    "GW_FLUORIDE_A", "GW_FLUORIDE_A_NUMERIC", "GW_FLUORIDE_A_QUALITY_FLAG",
    "GW_NITRATE_AS", "GW_NITRATE_AS_NUMERIC", "GW_NITRATE_AS_QUALITY_FLAG",
    "GW_DATE_QUALITY_FLAG", "GW_SAMPLING_DATE_CLEAN",
    "GW_IDENTIFIER_QUALITY_FLAG", "GW_OVERALL_QUALITY_FLAG",
    "GW_distance_km_from_okahandja"
]
gw_cols = [c for c in gw_cols if c in link.columns]

soil_meta_cols = [
    "SOIL_PRID", "SOIL_Dataset", "SOIL_CRS_METADATA",
    "SOIL_COORDINATE_TRANSFORM_STATUS", "SOIL_LATITUDE_WGS84",
    "SOIL_LONGITUDE_WGS84", "SOIL_DISTANCE_FROM_OKAHANDJA_KM",
    "SOIL_RANK", "DISTANCE_KM", "DISTANCE_BAND",
    "parent_material", "lithology", "profile_drainage",
    "soil_depth_class", "slope", "horizon_count",
    "depth_resolved_horizon_count"
]
soil_meta_cols = [c for c in soil_meta_cols if c in link.columns]
count_cols = [
    c for c in soil.columns
    if c.endswith("_measurement_count") and c in link.columns
]
selected = list(dict.fromkeys(gw_cols + soil_meta_cols + mean_cols + sorted(count_cols)))
dataset = link[selected].copy()

# Do not include classifications derived from the groundwater chemistry as predictors.
for forbidden in ["GW_WQ", "GW_T", "GW_S", "GW_F", "GW_NO3"]:
    if forbidden in dataset.columns:
        raise ValueError(f"Derived classification field unexpectedly included: {forbidden}")

dataset_path = OUT / "BOREMAN_OKAHANDJA_MODELLING_READY.csv"
dataset.to_csv(dataset_path, index=False)

targets = [
    c for c in [
        "GW_TOTAL_DISS_NUMERIC", "GW_SULPHATE_A_NUMERIC",
        "GW_FLUORIDE_A_NUMERIC", "GW_NITRATE_AS_NUMERIC"
    ] if c in dataset.columns
]
target_coverage = pd.DataFrame([{
    "target": c,
    "records": len(dataset),
    "available_numeric_values": int(dataset[c].notna().sum()),
    "missing_numeric_values": int(dataset[c].isna().sum()),
    "coverage_percent": round(dataset[c].notna().mean() * 100, 2)
} for c in targets])
target_coverage.to_csv(OUT / "TARGET_COVERAGE.csv", index=False)

feature_coverage = pd.DataFrame([{
    "feature": c,
    "available_values": int(dataset[c].notna().sum()),
    "missing_values": int(dataset[c].isna().sum()),
    "coverage_percent": round(dataset[c].notna().mean() * 100, 2),
    "unique_soil_profiles_with_value": int(dataset.loc[dataset[c].notna(), "SOIL_PRID"].nunique())
} for c in mean_cols])
feature_coverage.to_csv(OUT / "SOIL_FEATURE_COVERAGE.csv", index=False)

distance = dataset["DISTANCE_BAND"].value_counts(dropna=False).rename_axis(
    "distance_band"
).reset_index(name="groundwater_records")
distance.to_csv(OUT / "SPATIAL_DISTANCE_DISTRIBUTION.csv", index=False)

report = f"""BOREMAN – STEP 14 MODELLING-READY DATASET REPORT

Input groundwater rows: {len(gw):,}
Engineered soil profiles: {len(soil):,}
Spatial link rows: {len(link):,}
Output rows: {len(dataset):,}
Output columns: {len(dataset.columns):,}
Unique soil profiles used: {link['SOIL_PRID'].nunique():,}
Overall profile mean columns verified: {len(mean_cols)}
Feature mismatches: 0

IMPORTANT
- Missing or invalid groundwater targets are not imputed.
- Soil data are nearby profile-level proxies, not direct borehole measurements.
- Multiple boreholes reuse some soil profiles; grouped/spatial validation is required.
- Do not use GW_WQ, GW_T, GW_S, GW_F or GW_NO3 as predictors of chemistry.
- Soil pH and EC are not groundwater pH and EC.
- Sand definition remains subject to source documentation.
- This is a prepared dataset, not a trained model.
"""
(OUT / "STEP14_MODELLING_DATASET_REPORT.txt").write_text(report, encoding="utf-8")

print("STEP 14 COMPLETED")
print(f"Rows: {len(dataset):,}")
print(f"Columns: {len(dataset.columns):,}")
print(f"Unique soil profiles: {link['SOIL_PRID'].nunique():,}")
print(f"Output folder: {OUT}")
