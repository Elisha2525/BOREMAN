from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "BOREMAN_OKAHANDJA_MODELLING_READY.csv"
OUT = ROOT / "outputs" / "eda"
CHARTS = OUT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(f"Modelling-ready CSV not found: {INPUT}")

df = pd.read_csv(INPUT, low_memory=False)
targets = {
    "GW_TOTAL_DISS_NUMERIC": "Total dissolved solids",
    "GW_SULPHATE_A_NUMERIC": "Sulphate",
    "GW_FLUORIDE_A_NUMERIC": "Fluoride",
    "GW_NITRATE_AS_NUMERIC": "Nitrate"
}
target_stats, outlier_rows = [], []
for col, label in targets.items():
    if col not in df.columns:
        continue
    values = pd.to_numeric(df[col], errors="coerce")
    s = values.dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3-q1
    low, high = q1-1.5*iqr, q3+1.5*iqr
    mask = values.notna() & ((values < low) | (values > high))
    target_stats.append({
        "target": col, "parameter": label, "total_records": len(df),
        "valid_numeric": len(s), "missing_or_unusable": len(df)-len(s),
        "coverage_percent": round(len(s)/len(df)*100, 2),
        "mean": s.mean(), "median": s.median(), "std_dev": s.std(),
        "minimum": s.min(), "q1": q1, "q3": q3, "maximum": s.max(),
        "skewness": s.skew()
    })
    outlier_rows.append({
        "target": col, "parameter": label, "lower_iqr_fence": low,
        "upper_iqr_fence": high, "flagged_records": int(mask.sum()),
        "flagged_percent_of_valid": round(mask.sum()/len(s)*100, 2) if len(s) else 0
    })
    flagged = df.loc[mask, ["GW_BOREMAN_RECORD_ID","GW_Source","GW_ID",
                            "GW_LATITUDE","GW_LONGITUDE",col]].copy()
    flagged["parameter"] = label
    flagged.to_csv(OUT / f"OUTLIER_REVIEW_{col}.csv", index=False)
    if not s.empty:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(s, bins=40)
        ax.set(title=f"{label}: distribution", xlabel=f"{label} (source units)",
               ylabel="Number of records")
        fig.tight_layout()
        fig.savefig(CHARTS / f"{col}_histogram.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8,3.8))
        ax.boxplot(s, vert=False)
        ax.set(title=f"{label}: boxplot (IQR screening only)",
               xlabel=f"{label} (source units)")
        fig.tight_layout()
        fig.savefig(CHARTS / f"{col}_boxplot.png", dpi=160)
        plt.close(fig)

pd.DataFrame(target_stats).to_csv(OUT / "01_GROUNDWATER_TARGET_STATISTICS.csv", index=False)
pd.DataFrame(outlier_rows).to_csv(OUT / "02_IQR_OUTLIER_SCREENING.csv", index=False)

target_cols = [c for c in targets if c in df.columns]
numeric_targets = df[target_cols].apply(pd.to_numeric, errors="coerce")
numeric_targets.corr(method="pearson", min_periods=30).to_csv(
    OUT / "03_GROUNDWATER_TARGET_PEARSON_CORRELATION.csv")
numeric_targets.corr(method="spearman", min_periods=30).to_csv(
    OUT / "04_GROUNDWATER_TARGET_SPEARMAN_CORRELATION.csv")

soil_features = [c for c in df.columns if c.endswith("_mean")]
coverage = []
for col in soil_features:
    vals = pd.to_numeric(df[col], errors="coerce")
    count_col = col.replace("_mean", "_measurement_count")
    counts = pd.to_numeric(df[count_col], errors="coerce") if count_col in df else pd.Series(dtype=float)
    coverage.append({
        "soil_feature": col, "available_groundwater_rows": int(vals.notna().sum()),
        "missing_groundwater_rows": int(vals.isna().sum()),
        "coverage_percent": round(vals.notna().mean()*100, 2),
        "unique_soil_profiles_with_measurement": int(df.loc[vals.notna(),"SOIL_PRID"].nunique()),
        "mean_source_measurements_per_profile": counts.mean() if len(counts) else np.nan
    })
coverage_df = pd.DataFrame(coverage).sort_values("coverage_percent", ascending=False)
coverage_df.to_csv(OUT / "05_SOIL_FEATURE_COVERAGE_EDA.csv", index=False)
if not coverage_df.empty:
    top = coverage_df.head(18).sort_values("coverage_percent")
    fig, ax = plt.subplots(figsize=(9,7))
    ax.barh(top["soil_feature"], top["coverage_percent"])
    ax.set(xlabel="Coverage across groundwater records (%)",
           title="Top soil features by data coverage")
    fig.tight_layout()
    fig.savefig(CHARTS / "soil_feature_coverage.png", dpi=160)
    plt.close(fig)

for col, filename, label in [
    ("DISTANCE_BAND","06_SPATIAL_DISTANCE_COUNTS.csv","distance_band"),
    ("GW_Source","07_GROUNDWATER_SOURCE_COUNTS.csv","groundwater_source"),
    ("SOIL_Dataset","08_NEAREST_SOIL_DATASET_COUNTS.csv","soil_dataset")
]:
    if col in df:
        df[col].value_counts(dropna=False).rename_axis(label).reset_index(
            name="records").to_csv(OUT / filename, index=False)

report = [
    "BOREMAN – STEP 15 EXPLORATORY DATA ANALYSIS (EDA)",
    f"Dataset rows: {len(df):,}",
    f"Dataset columns: {len(df.columns):,}",
    f"Unique nearest soil profiles: {df['SOIL_PRID'].nunique():,}",
    "",
    "TARGET STATISTICS", pd.DataFrame(target_stats).to_string(index=False),
    "", "IQR OUTLIER SCREENING", pd.DataFrame(outlier_rows).to_string(index=False),
    "", "SOIL FEATURE COVERAGE", coverage_df.head(18).to_string(index=False),
    "", "LIMITATIONS",
    "- IQR flags are review indicators, not proof of erroneous measurements; no values were deleted.",
    "- Correlation is descriptive, not causal.",
    "- Soil features are nearest-profile proxies, not direct borehole measurements.",
    "- Reused soil profiles require grouped/spatial validation to prevent leakage.",
    "- Confirm groundwater measurement units from Atlas metadata before interpreting thresholds.",
    "- Missing values were not imputed.",
    "- Soil pH and EC are not groundwater pH and EC.",
    "- EDA does not establish model suitability or predictive performance."
]
(OUT / "STEP15_EDA_REPORT.txt").write_text("\n".join(report), encoding="utf-8")
print("STEP 15 EDA COMPLETED")
print(f"Rows analysed: {len(df):,}")
print(f"Outputs: {OUT}")
