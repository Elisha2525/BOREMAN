
from pathlib import Path
from datetime import datetime
import pandas as pd

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "BOREMAN_OKAHANDJA_100KM_GROUNDWATER.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "BOREMAN_OKAHANDJA_QUALITY_FLAGGED.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "BOREMAN_CLEANING_REPORT.txt"
)

# ==================================================
# 2. LOAD ORIGINAL DATA
# ==================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    keep_default_na=False
)

original_count = len(df)

if original_count == 0:
    raise ValueError("Input dataset contains no records.")

required_columns = [
    "Source",
    "ID",
    "SAMPLING_D",
    "TOTAL_DISS",
    "SULPHATE_A",
    "FLUORIDE_A",
    "NITRATE_AS"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Create a unique reference for each row.
df.insert(
    0,
    "BOREMAN_RECORD_ID",
    [
        f"BOR-{i:05d}"
        for i in range(1, len(df) + 1)
    ]
)

print("=" * 60)
print("BOREMAN DATA CLEANING")
print("=" * 60)

print("Original records:", original_count)

# ==================================================
# 3. DATE QUALITY FLAGS
# ==================================================

raw_dates = df["SAMPLING_D"].astype(str).str.strip()

date_status = []
clean_dates = []

DATE_FORMATS = [
    "%Y-%m-%d %I:%M:%S %p",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y"
]

for value in raw_dates:

    if value == "":
        date_status.append("MISSING_DATE")
        clean_dates.append("")

    elif value.startswith("1980-01-01"):
        date_status.append(
            "YEAR_ONLY_OR_DEFAULT_UNVERIFIED"
        )
        clean_dates.append("")

    elif value.startswith("1900-01-01"):
        date_status.append(
            "LIKELY_PLACEHOLDER_UNVERIFIED"
        )
        clean_dates.append("")

    else:

        parsed = None

        for date_format in DATE_FORMATS:

            try:
                parsed = datetime.strptime(
                    value,
                    date_format
                )
                break

            except ValueError:
                continue

        if parsed is None:
            date_status.append("UNPARSED_DATE")
            clean_dates.append("")

        else:
            date_status.append(
                "RECORDED_DATE_PARSED"
            )
            clean_dates.append(
                parsed.strftime("%Y-%m-%d")
            )

# Verify date flags before assigning them.
assert len(date_status) == len(df), (
    f"Date flags: {len(date_status)}, "
    f"records: {len(df)}"
)

assert len(clean_dates) == len(df)

df["DATE_QUALITY_FLAG"] = date_status
df["SAMPLING_DATE_CLEAN"] = clean_dates

# ==================================================
# 4. IDENTIFIER QUALITY FLAGS
# ==================================================

def identifier_flag(row):

    source = str(row["Source"]).strip()
    identifier = str(row["ID"]).strip()

    if identifier == "":
        return "MISSING_ID"

    if identifier.upper() in [
        "?",
        "NA",
        "N/A",
        "NULL"
    ]:
        return "UNKNOWN_ID"

    if source == "":
        return "MISSING_SOURCE"

    return "USABLE_ID"


df["IDENTIFIER_QUALITY_FLAG"] = df.apply(
    identifier_flag,
    axis=1
)

# ==================================================
# 5. CHEMICAL DATA QUALITY FLAGS
# ==================================================

parameters = [
    "TOTAL_DISS",
    "SULPHATE_A",
    "FLUORIDE_A",
    "NITRATE_AS"
]

for col in parameters:

    raw_values = (
        df[col]
        .astype(str)
        .str.strip()
    )

    numeric_values = pd.to_numeric(
        raw_values,
        errors="coerce"
    )

    flags = []
    model_values = []

    for raw, numeric in zip(
        raw_values,
        numeric_values
    ):

        if raw == "":
            flags.append("MISSING_VALUE")
            model_values.append(float("nan"))

        elif raw.startswith("<"):
            flags.append("CENSORED_VALUE")
            model_values.append(float("nan"))

        elif pd.isna(numeric):
            flags.append("UNPARSED_VALUE")
            model_values.append(float("nan"))

        elif numeric < 0:
            flags.append(
                "NEGATIVE_VALUE_UNVERIFIED_CODE"
            )
            model_values.append(float("nan"))

        else:
            flags.append(
                "VALID_NONNEGATIVE_NUMERIC"
            )
            model_values.append(float(numeric))

    assert len(flags) == len(df)
    assert len(model_values) == len(df)

    # Keep original measurements unchanged.
    # These additional columns are for modelling.
    df[col + "_NUMERIC"] = model_values

    df[col + "_QUALITY_FLAG"] = flags

# ==================================================
# 6. OVERALL RECORD QUALITY
# ==================================================

def overall_flag(row):

    problems = []

    if (
        row["DATE_QUALITY_FLAG"]
        != "RECORDED_DATE_PARSED"
    ):
        problems.append("DATE")

    if (
        row["IDENTIFIER_QUALITY_FLAG"]
        != "USABLE_ID"
    ):
        problems.append("IDENTIFIER")

    for col in parameters:

        if (
            row[col + "_QUALITY_FLAG"]
            != "VALID_NONNEGATIVE_NUMERIC"
        ):
            problems.append(col)

    if problems:
        return (
            "REVIEW_REQUIRED: "
            + "; ".join(problems)
        )

    return "NO_FLAGGED_ISSUES_IN_SELECTED_FIELDS"


df["OVERALL_QUALITY_FLAG"] = df.apply(
    overall_flag,
    axis=1
)

# ==================================================
# 7. FINAL VALIDATION
# ==================================================

if len(df) != original_count:
    raise ValueError(
        "Record count changed during cleaning."
    )

quality_columns = [
    "DATE_QUALITY_FLAG",
    "IDENTIFIER_QUALITY_FLAG",
    "OVERALL_QUALITY_FLAG"
]

for col in quality_columns:

    if df[col].isna().any():
        raise ValueError(
            f"Missing quality flags detected in {col}"
        )

# ==================================================
# 8. SAVE CLEANED DATASET
# ==================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# 9. GENERATE CLEANING REPORT
# ==================================================

report = []

report.append(
    "BOREMAN CLEANING AND QUALITY FLAG REPORT"
)

report.append("=" * 60)
report.append("")

report.append(
    f"Original records: {original_count}"
)

report.append(
    f"Output records: {len(df)}"
)

report.append(
    f"Records removed: {original_count - len(df)}"
)

report.append("")

# Date report
report.append("DATE QUALITY")
report.append("-" * 40)

report.append(
    df["DATE_QUALITY_FLAG"]
    .value_counts(dropna=False)
    .to_string()
)

report.append("")

# Identifier report
report.append("IDENTIFIER QUALITY")
report.append("-" * 40)

report.append(
    df["IDENTIFIER_QUALITY_FLAG"]
    .value_counts(dropna=False)
    .to_string()
)

report.append("")

# Chemical report
report.append("CHEMICAL QUALITY")
report.append("-" * 40)

for col in parameters:

    report.append("")
    report.append(col)

    report.append(
        df[col + "_QUALITY_FLAG"]
        .value_counts(dropna=False)
        .to_string()
    )

report.append("")

# Overall quality
report.append("OVERALL RECORD QUALITY")
report.append("-" * 40)

report.append(
    df["OVERALL_QUALITY_FLAG"]
    .value_counts(dropna=False)
    .to_string()
)

report.append("")

# Cleaning decisions
report.append("CLEANING DECISIONS")
report.append("-" * 40)

decisions = [
    "All original records were retained.",
    "Original chemical columns were not overwritten.",
    "Negative chemical values were excluded from numeric modelling columns.",
    "Censored values were preserved in original columns.",
    "Censored values were not converted to zero.",
    "Negative values were not converted to zero.",
    "1900 and 1980 dates were flagged and excluded from clean date fields.",
    "Blank identifiers were retained and flagged.",
    "No original source records were deleted.",
    "Units and negative-value codes still require source verification."
]

for decision in decisions:
    report.append("- " + decision)

report.append("")

report.append(
    "The cleaned dataset is a quality-flagged research dataset, "
    "not yet a fully validated modelling dataset."
)

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 60)
print("BOREMAN CLEANING COMPLETED")
print("=" * 60)

print("Original records:", original_count)
print("Output records:", len(df))
print("Records removed:", original_count - len(df))

print("\nDATE QUALITY")
print(
    df["DATE_QUALITY_FLAG"].value_counts().to_string()
)

print("\nIDENTIFIER QUALITY")
print(
    df["IDENTIFIER_QUALITY_FLAG"].value_counts().to_string()
)

print("\nCHEMICAL QUALITY")

for col in parameters:

    print(f"\n{col}")

    print(
        df[col + "_QUALITY_FLAG"]
        .value_counts()
        .to_string()
    )

print("\nOUTPUT FILES")
print("Cleaned dataset:", OUTPUT_FILE)
print("Cleaning report:", REPORT_FILE)

print("\nAll validation checks passed.")
print("Completed successfully.")
