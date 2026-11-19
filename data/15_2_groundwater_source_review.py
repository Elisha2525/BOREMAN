from pathlib import Path
import pandas as pd
import numpy as np

# BOREMAN Step 15.2 – Groundwater outlier and source verification
# Run from the project root:
# .\.venv\Scripts\python.exe data\15_2_groundwater_source_review.py

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "BOREMAN_OKAHANDJA_MODELLING_READY.csv"
OUT = ROOT / "outputs" / "groundwater_source_review"
OUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT}\n"
        "Place BOREMAN_OKAHANDJA_MODELLING_READY.csv in the project's data folder."
    )

df = pd.read_csv(INPUT, low_memory=False)

TARGETS = {
    "GW_TOTAL_DISS_NUMERIC": "TDS",
    "GW_SULPHATE_A_NUMERIC": "Sulphate",
    "GW_FLUORIDE_A_NUMERIC": "Fluoride",
    "GW_NITRATE_AS_NUMERIC": "Nitrate",
}
ID_COLS = [
    "GW_BOREMAN_RECORD_ID", "GW_Source", "GW_ID",
    "GW_SAMPLE_NAM", "GW_SAMPLING_D",
    "GW_LATITUDE", "GW_LONGITUDE", "GW_ELEVATION",
    "GW_DATE_QUALITY_FLAG", "GW_OVERALL_QUALITY_FLAG",
]

for col in TARGETS:
    if col not in df.columns:
        raise ValueError(f"Required target column is missing: {col}")

# Convert targets to numeric for analysis only; original columns remain untouched.
for col in TARGETS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 1. Source-level summary: count, coverage, distribution and zero count.
source_rows = []
if "GW_Source" in df.columns:
    for source, group in df.groupby("GW_Source", dropna=False):
        for col, label in TARGETS.items():
            values = group[col].dropna()
            source_rows.append({
                "source": source,
                "parameter": label,
                "records_from_source": len(group),
                "valid_numeric": len(values),
                "missing_or_unusable": len(group) - len(values),
                "coverage_percent": round(len(values) / len(group) * 100, 2) if len(group) else np.nan,
                "zero_count": int((values == 0).sum()),
                "minimum": values.min() if len(values) else np.nan,
                "median": values.median() if len(values) else np.nan,
                "maximum": values.max() if len(values) else np.nan,
            })
source_summary = pd.DataFrame(source_rows)
source_summary.to_csv(OUT / "01_SOURCE_PARAMETER_SUMMARY.csv", index=False)

# 2. Overall zero-value review. Zero is flagged, never assumed invalid.
zero_frames = []
for col, label in TARGETS.items():
    mask = df[col].eq(0)
    if mask.any():
        keep = [c for c in ID_COLS if c in df.columns] + [col]
        part = df.loc[mask, keep].copy()
        part["parameter"] = label
        part["review_reason"] = "ZERO_VALUE_REQUIRES_SOURCE_AND_METHOD_CHECK"
        zero_frames.append(part)
zero_review = pd.concat(zero_frames, ignore_index=True) if zero_frames else pd.DataFrame()
zero_review.to_csv(OUT / "02_ZERO_VALUE_REVIEW.csv", index=False)

# 3. Extreme values: retain top and bottom 20 for each parameter.
extreme_frames = []
for col, label in TARGETS.items():
    valid = df.loc[df[col].notna()].copy()
    if valid.empty:
        continue
    keep = [c for c in ID_COLS if c in valid.columns] + [col]
    low = valid.nsmallest(min(20, len(valid)), col)[keep].copy()
    low["parameter"] = label
    low["extreme_group"] = "LOWEST_20"
    high = valid.nlargest(min(20, len(valid)), col)[keep].copy()
    high["parameter"] = label
    high["extreme_group"] = "HIGHEST_20"
    extreme_frames.extend([low, high])
extremes = pd.concat(extreme_frames, ignore_index=True) if extreme_frames else pd.DataFrame()
extremes.to_csv(OUT / "03_LOWEST_HIGHEST_VALUES.csv", index=False)

# 4. IQR outlier records, with source, coordinates and quality flags.
outlier_frames = []
outlier_summary = []
for col, label in TARGETS.items():
    values = df[col]
    valid = values.dropna()
    if valid.empty:
        continue
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = values.notna() & ((values < lower) | (values > upper))
    keep = [c for c in ID_COLS if c in df.columns] + [col]
    part = df.loc[mask, keep].copy()
    part["parameter"] = label
    part["iqr_lower_fence"] = lower
    part["iqr_upper_fence"] = upper
    part["review_reason"] = "STATISTICAL_IQR_FLAG_NOT_CONFIRMED_ERROR"
    outlier_frames.append(part)
    outlier_summary.append({
        "parameter": label, "valid_records": len(valid),
        "q1": q1, "median": valid.median(), "q3": q3,
        "lower_fence": lower, "upper_fence": upper,
        "flagged_records": int(mask.sum()),
        "flagged_percent": round(mask.sum() / len(valid) * 100, 2),
    })
all_outliers = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame()
all_outliers.to_csv(OUT / "04_IQR_OUTLIER_RECORDS.csv", index=False)
pd.DataFrame(outlier_summary).to_csv(OUT / "05_IQR_OUTLIER_SUMMARY.csv", index=False)

# 5. Geographic grid summary for IQR flags.
# A 0.1-degree grid is a descriptive grouping only, not a statistical cluster test.
if {"GW_LATITUDE", "GW_LONGITUDE"}.issubset(df.columns):
    geo = df.copy()
    geo["latitude"] = pd.to_numeric(geo["GW_LATITUDE"], errors="coerce")
    geo["longitude"] = pd.to_numeric(geo["GW_LONGITUDE"], errors="coerce")
    geo = geo.dropna(subset=["latitude", "longitude"])
    geo["grid_lat_0_1deg"] = (np.floor(geo["latitude"] * 10) / 10).round(1)
    geo["grid_lon_0_1deg"] = (np.floor(geo["longitude"] * 10) / 10).round(1)
    geo_rows = []
    for col, label in TARGETS.items():
        valid = geo[col].dropna()
        if valid.empty:
            continue
        q1, q3 = valid.quantile(.25), valid.quantile(.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        geo["is_iqr_flag"] = geo[col].notna() & ((geo[col] < lower) | (geo[col] > upper))
        summary = geo.groupby(["grid_lat_0_1deg", "grid_lon_0_1deg"], dropna=False).agg(
            groundwater_records=(col, "size"),
            valid_parameter_records=(col, "count"),
            median_value=(col, "median"),
            maximum_value=(col, "max"),
            iqr_flagged_records=("is_iqr_flag", "sum"),
        ).reset_index()
        summary["parameter"] = label
        geo_rows.append(summary)
    geo_summary = pd.concat(geo_rows, ignore_index=True) if geo_rows else pd.DataFrame()
    geo_summary.to_csv(OUT / "06_GEOGRAPHIC_GRID_SUMMARY.csv", index=False)

# 6. Overall source counts and quality-flag counts.
if "GW_Source" in df.columns:
    df["GW_Source"].value_counts(dropna=False).rename_axis("source").reset_index(
        name="records"
    ).to_csv(OUT / "07_SOURCE_RECORD_COUNTS.csv", index=False)

for col in ["GW_DATE_QUALITY_FLAG", "GW_OVERALL_QUALITY_FLAG"]:
    if col in df.columns:
        df[col].value_counts(dropna=False).rename_axis("quality_flag").reset_index(
            name="records"
        ).to_csv(OUT / f"08_{col}_COUNTS.csv", index=False)

# 7. Report
lines = [
    "BOREMAN – STEP 15.2 GROUNDWATER OUTLIER AND SOURCE REVIEW",
    "",
    f"Input records: {len(df):,}",
    f"Unique groundwater record IDs: {df['GW_BOREMAN_RECORD_ID'].nunique() if 'GW_BOREMAN_RECORD_ID' in df else 'Not available'}",
    "",
    "SOURCE COUNTS",
    df["GW_Source"].value_counts(dropna=False).to_string() if "GW_Source" in df.columns else "Source field unavailable.",
    "",
    "PARAMETER REVIEW",
    pd.DataFrame(outlier_summary).to_string(index=False),
    "",
    "ZERO VALUES",
    zero_review.groupby("parameter").size().to_string() if not zero_review.empty else "No zero values found.",
    "",
    "IMPORTANT INTERPRETATION RULES",
    "- IQR flags are statistical review indicators, not confirmed measurement errors.",
    "- Zero values are flagged for investigation, not automatically changed or deleted.",
    "- The script preserves source rows and does not modify the input dataset.",
    "- Geographic grid summaries are descriptive bins, not proof of spatial clustering.",
    "- Source comparisons must account for unequal source sample sizes and possible method differences.",
    "- Confirm parameter units, analytical methods, detection limits and source definitions from Atlas documentation.",
    "- Do not interpret drinking-water compliance from these values until units and methods are verified.",
]
(OUT / "STEP15_2_GROUNDWATER_SOURCE_REVIEW_REPORT.txt").write_text(
    "\n".join(lines), encoding="utf-8"
)

print("STEP 15.2 COMPLETED")
print(f"Records reviewed: {len(df):,}")
print(f"Sources: {df['GW_Source'].nunique(dropna=False) if 'GW_Source' in df.columns else 'Not available'}")
print(f"Zero-value rows flagged: {len(zero_review):,}")
print(f"IQR outlier rows across parameters: {len(all_outliers):,}")
print(f"Output folder: {OUT}")
