
import os
import glob
import numpy as np
import pandas as pd

# --------------------------------------------------
# BOREMAN GROUNDWATER QUALITY CHECK
# --------------------------------------------------

DATA_DIR = r"C:\Users\mayan\dataset"
OUTPUT_DIR = os.path.join(DATA_DIR, "06_BOREMAN_QUALITY_CHECK")

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(
    DATA_DIR,
    "05_OKAHANDJA_FILTERED",
    "OKAHANDJA_GROUNDWATER_CHEMISTRY.csv"
)

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_FILE}")

print("=" * 65)
print("BOREMAN GROUNDWATER QUALITY CHECK")
print("=" * 65)

# Load data
df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"\nInput file: {INPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# Preserve original data
df_original = df.copy()

# --------------------------------------------------
# 1. MISSING VALUES
# --------------------------------------------------

missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": [df[c].isna().sum() for c in df.columns],
    "missing_percentage": [
        round(df[c].isna().mean() * 100, 2) for c in df.columns
    ]
})

missing.to_csv(
    os.path.join(OUTPUT_DIR, "BOREMAN_MISSING_VALUES.csv"),
    index=False
)

# --------------------------------------------------
# 2. DUPLICATE CHECK
# --------------------------------------------------

duplicate_count = int(df.duplicated().sum())

duplicate_report = pd.DataFrame([{
    "total_rows": len(df),
    "exact_duplicate_rows": duplicate_count,
    "unique_rows": len(df) - duplicate_count
}])

duplicate_report.to_csv(
    os.path.join(OUTPUT_DIR, "BOREMAN_DUPLICATE_CHECK.csv"),
    index=False
)

# --------------------------------------------------
# 3. NUMERIC PARAMETER SUMMARY
# --------------------------------------------------

numeric = df.select_dtypes(include=[np.number])

if not numeric.empty:
    summary = numeric.describe().T.reset_index()
    summary = summary.rename(columns={"index": "parameter"})
else:
    summary = pd.DataFrame(
        columns=["parameter", "count", "mean", "std", "min", "max"]
    )

summary.to_csv(
    os.path.join(OUTPUT_DIR, "BOREMAN_PARAMETER_SUMMARY.csv"),
    index=False
)

# --------------------------------------------------
# 4. CLEANING AND WARNING FLAGS
# --------------------------------------------------

clean = df.copy()

# Flag negative numeric measurements.
# Do not silently delete them: -1 may be a source-specific
# missing-value code and needs official documentation.
negative_summary = []

for col in numeric.columns:
    values = pd.to_numeric(clean[col], errors="coerce")
    negative_flag = values.lt(0).fillna(False)

    clean[col + "_negative_flag"] = negative_flag

    negative_summary.append({
        "parameter": col,
        "negative_value_count": int(negative_flag.sum())
    })

pd.DataFrame(negative_summary).to_csv(
    os.path.join(OUTPUT_DIR, "BOREMAN_NEGATIVE_VALUE_FLAGS.csv"),
    index=False
)

# Preserve original measurements. No automatic imputation,
# deletion, or replacement of censored values.

clean.to_csv(
    os.path.join(OUTPUT_DIR, "BOREMAN_CLEANED_DATA.csv"),
    index=False
)

# --------------------------------------------------
# 5. TEXT QUALITY REPORT
# --------------------------------------------------

report_path = os.path.join(
    OUTPUT_DIR,
    "BOREMAN_DATA_QUALITY_REPORT.txt"
)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("BOREMAN GROUNDWATER DATA QUALITY REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Total rows: {len(df):,}\n")
    f.write(f"Total columns: {len(df.columns)}\n")
    f.write(f"Exact duplicate rows: {duplicate_count:,}\n\n")

    f.write("COLUMN NAMES\n")
    f.write("-" * 60 + "\n")
    for col in df.columns:
        f.write(f"{col}\n")

    f.write("\nMISSING VALUES\n")
    f.write("-" * 60 + "\n")
    f.write(missing.to_string(index=False))

    f.write("\n\nNUMERIC SUMMARY\n")
    f.write("-" * 60 + "\n")
    f.write(summary.to_string(index=False))

    f.write("\n\nIMPORTANT DATA LIMITATIONS\n")
    f.write("-" * 60 + "\n")
    f.write(
        "Negative values are flagged but retained.\n"
        "Censored measurements such as <0.5 are not converted "
        "to exact numeric values.\n"
        "No missing-value imputation has been performed.\n"
        "No records have been deleted.\n"
        "Units, dates, detection limits and sentinel values "
        "require source documentation verification.\n"
    )

print("\nQUALITY CHECK COMPLETED")
print("Output folder:", OUTPUT_DIR)
print("Files generated:")
for name in os.listdir(OUTPUT_DIR):
    print(" -", name)

print("=" * 65)
