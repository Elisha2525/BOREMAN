
import re
from pathlib import Path
import pandas as pd

# ==========================================
# BOREMAN DATA QUALITY ASSESSMENT
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
QC_DIR = OUTPUT_DIR / "quality_check"

INPUT_FILE = (
    OUTPUT_DIR /
    "BOREMAN_OKAHANDJA_100KM_GROUNDWATER.csv"
)

QC_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

print("=" * 65)
print("BOREMAN GROUNDWATER DATA QUALITY ASSESSMENT")
print("=" * 65)

# Read as text first to preserve original values,
# including censored measurements such as <0.5.
df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

# ==========================================
# 1. MISSING VALUE ANALYSIS
# ==========================================

missing_rows = []

for col in df.columns:
    values = df[col].astype(str).str.strip()
    missing_count = int(values.isin(["", "NA", "N/A", "NULL"]).sum())

    missing_rows.append({
        "column": col,
        "missing_count": missing_count,
        "missing_percentage": round(
            missing_count / len(df) * 100, 2
        ) if len(df) else 0
    })

missing_df = pd.DataFrame(missing_rows)

missing_df.to_csv(
    QC_DIR / "01_missing_values.csv",
    index=False
)

# ==========================================
# 2. EXACT DUPLICATE ANALYSIS
# ==========================================

exact_duplicates = int(df.duplicated().sum())

duplicate_rows = df[df.duplicated(keep=False)].copy()

duplicate_rows.to_csv(
    QC_DIR / "02_exact_duplicate_records.csv",
    index=False
)

# ==========================================
# 3. REPEATED SOURCE AND ID ANALYSIS
# ==========================================

if {"Source", "ID"}.issubset(df.columns):

    identifiers = df.copy()

    identifiers["Source"] = (
        identifiers["Source"].str.strip()
    )
    identifiers["ID"] = identifiers["ID"].str.strip()

    valid_ids = identifiers[
        ~identifiers["ID"].isin(["", "?", "NA", "N/A"])
    ]

    repeated_ids = (
        valid_ids.groupby(["Source", "ID"])
        .size()
        .reset_index(name="record_count")
    )

    repeated_ids = repeated_ids[
        repeated_ids["record_count"] > 1
    ].sort_values("record_count", ascending=False)

else:
    repeated_ids = pd.DataFrame(
        columns=["Source", "ID", "record_count"]
    )

repeated_ids.to_csv(
    QC_DIR / "03_repeated_borehole_ids.csv",
    index=False
)

# ==========================================
# 4. CHEMICAL PARAMETER ANALYSIS
# ==========================================

parameters = [
    "TOTAL_DISS",
    "SULPHATE_A",
    "FLUORIDE_A",
    "NITRATE_AS"
]

parameter_summary = []
clean = df.copy()

for col in parameters:

    if col not in df.columns:
        continue

    original = df[col].astype(str).str.strip()

    # Preserve original field.
    # Create separate numeric analysis column.
    numeric = pd.to_numeric(
        original.replace("", pd.NA),
        errors="coerce"
    )

    censored = original.str.match(
        r"^\s*<\s*\d", na=False
    )

    missing = original.isin(
        ["", "NA", "N/A", "NULL"]
    )

    invalid = (
        ~missing
        & ~censored
        & numeric.isna()
    )

    negative = numeric.lt(0).fillna(False)

    parameter_summary.append({
        "parameter": col,
        "total_records": len(df),
        "numeric_values": int(numeric.notna().sum()),
        "missing_values": int(missing.sum()),
        "censored_values": int(censored.sum()),
        "unparsed_values": int(invalid.sum()),
        "negative_values": int(negative.sum()),
        "minimum": numeric.min(),
        "maximum": numeric.max(),
        "mean": numeric.mean(),
        "median": numeric.median()
    })

    clean[col + "_numeric"] = numeric
    clean[col + "_censored_flag"] = censored
    clean[col + "_negative_flag"] = negative
    clean[col + "_unparsed_flag"] = invalid

summary_df = pd.DataFrame(parameter_summary)

summary_df.to_csv(
    QC_DIR / "04_chemical_parameter_summary.csv",
    index=False
)

# ==========================================
# 5. SAMPLING DATE ANALYSIS
# ==========================================

if "SAMPLING_D" in df.columns:

    date_values = df["SAMPLING_D"].astype(str).str.strip()

    parsed_dates = pd.to_datetime(
        date_values.replace("", pd.NA),
        format="mixed",
        errors="coerce"
    )

    date_report = pd.DataFrame([{
        "total_records": len(df),
        "valid_dates": int(parsed_dates.notna().sum()),
        "unparsed_or_missing_dates": int(
            parsed_dates.isna().sum()
        ),
        "earliest_date": (
            parsed_dates.min()
            if parsed_dates.notna().any()
            else None
        ),
        "latest_date": (
            parsed_dates.max()
            if parsed_dates.notna().any()
            else None
        )
    }])

else:
    date_report = pd.DataFrame()

date_report.to_csv(
    QC_DIR / "05_sampling_date_summary.csv",
    index=False
)

# ==========================================
# 6. WATER QUALITY CLASS DISTRIBUTION
# ==========================================

classification_fields = [
    "T", "S", "F", "NO3", "WQ"
]

class_reports = []

for col in classification_fields:

    if col not in df.columns:
        continue

    counts = df[col].value_counts(dropna=False)

    for category, count in counts.items():

        class_reports.append({
            "parameter": col,
            "classification": category,
            "count": int(count),
            "percentage": round(
                count / len(df) * 100, 2
            ) if len(df) else 0
        })

class_df = pd.DataFrame(class_reports)

class_df.to_csv(
    QC_DIR / "06_classification_distribution.csv",
    index=False
)

# ==========================================
# 7. PRESERVE ENRICHED ORIGINAL DATA
# ==========================================

clean.to_csv(
    QC_DIR / "07_audited_dataset_with_flags.csv",
    index=False
)

# ==========================================
# 8. GENERATE TEXT REPORT
# ==========================================

report_file = QC_DIR / "BOREMAN_DATA_QUALITY_REPORT.txt"

with open(report_file, "w", encoding="utf-8") as report:

    report.write("BOREMAN GROUNDWATER DATA QUALITY REPORT\n")
    report.write("=" * 65 + "\n\n")

    report.write(f"Input: {INPUT_FILE}\n")
    report.write(f"Total records: {len(df)}\n")
    report.write(f"Total columns: {len(df.columns)}\n")

    report.write(
        f"Exact duplicate rows: {exact_duplicates}\n"
    )

    report.write(
        f"Repeated source/ID combinations: {len(repeated_ids)}\n"
    )

    report.write("\nMISSING VALUES\n")
    report.write(missing_df.to_string(index=False))

    report.write("\n\nCHEMICAL PARAMETER SUMMARY\n")
    report.write(summary_df.to_string(index=False))

    report.write("\n\nSAMPLING DATE SUMMARY\n")
    report.write(date_report.to_string(index=False))

    report.write("\n\nCLASSIFICATION DISTRIBUTION\n")
    report.write(class_df.to_string(index=False))

    report.write("\n\nIMPORTANT NOTES\n")
    report.write(
        "\nOriginal chemical measurements are preserved."
    )
    report.write(
        "\nCensored measurements are flagged, not replaced."
    )
    report.write(
        "\nNegative values are flagged, not deleted."
    )
    report.write(
        "\nRepeated IDs are not automatically duplicates."
    )
    report.write(
        "\nUnits and classification thresholds need verification."
    )

# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\nQUALITY ASSESSMENT COMPLETED")
print("=" * 65)

print("Total records:", len(df))
print("Exact duplicate records:", exact_duplicates)
print("Repeated source/ID combinations:", len(repeated_ids))

print("\nCHEMICAL PARAMETERS:")
print(summary_df.to_string(index=False))

print("\nOUTPUT DIRECTORY:")
print(QC_DIR)

print("\nGenerated files:")

for file in sorted(QC_DIR.iterdir()):
    if file.is_file():
        print("-", file.name)

print("\nNext stage: environmental data integration.")
