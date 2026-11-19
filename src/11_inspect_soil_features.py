
from pathlib import Path
import pandas as pd
import re

# ---------------------------------------------------------
# BOREMAN - STEP 11
# Soil feature inspection
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "outputs" / "soil"
OUTPUT_DIR = BASE_DIR / "outputs" / "soil_feature_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "profiles": INPUT_DIR / "BOREMAN_OKAHANDJA_SOIL_PROFILES.csv",
    "lab_horizons": INPUT_DIR / "BOREMAN_OKAHANDJA_SOIL_HORIZONS.csv",
    "field_horizons": INPUT_DIR / "BOREMAN_OKAHANDJA_SOIL_FIELD_HORIZONS.csv",
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def read_csv_safely(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                low_memory=False
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read file with supported encodings: {path}")


def normalize_header(value):
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def numeric_parse(value):
    """
    Inspect whether a value can be interpreted as numeric.
    Original source values are never changed.
    """
    value = clean_value(value)

    if not value:
        return None

    value = value.replace(",", "")

    # Do not interpret censored or inequality-based values as numbers.
    if any(symbol in value for symbol in ["<", ">", "≤", "≥"]):
        return None

    try:
        return float(value)
    except ValueError:
        return None


def identify_key_column(columns, key):
    for column in columns:
        if normalize_header(column) == key.lower():
            return column
    return None


def find_relevant_columns(columns):
    keywords = [
        "depth", "thickness", "sand", "silt", "clay",
        "ph", "ec", "electrical conductivity",
        "organic carbon", "carbon", "nitrogen",
        "calcium", "magnesium", "potassium", "sodium",
        "sulphate", "sulfate", "chloride", "cec",
        "bulk density", "texture", "carbonate",
        "drainage", "lithology", "parent material",
        "geology", "porosity", "roots", "structure",
        "elevation", "slope", "latitude", "longitude"
    ]

    return [
        col for col in columns
        if any(word in normalize_header(col) for word in keywords)
    ]


# ---------------------------------------------------------
# Inspect each file
# ---------------------------------------------------------

all_inventory = []
all_coverage = []
all_samples = []
report_lines = []

report_lines.append("BOREMAN - SOIL FEATURE INSPECTION")
report_lines.append("=" * 65)
report_lines.append("This report inspects source data without modifying it.")
report_lines.append("")

datasets = {}

for dataset_name, file_path in FILES.items():

    print(f"\nInspecting: {dataset_name}")

    df = read_csv_safely(file_path)
    datasets[dataset_name] = df

    row_count = len(df)
    column_count = len(df.columns)

    report_lines.append(f"\nDATASET: {dataset_name}")
    report_lines.append("-" * 65)
    report_lines.append(f"File: {file_path.name}")
    report_lines.append(f"Rows: {row_count:,}")
    report_lines.append(f"Columns: {column_count:,}")

    print(f"Rows: {row_count:,}")
    print(f"Columns: {column_count:,}")

    # Key fields
    prid_column = identify_key_column(df.columns, "prid")
    honu_column = identify_key_column(df.columns, "honu")
    dataset_column = identify_key_column(df.columns, "dataset")

    report_lines.append(f"PRID column: {prid_column}")
    report_lines.append(f"HONU column: {honu_column}")
    report_lines.append(f"Dataset column: {dataset_column}")

    # Dataset distribution
    if dataset_column:
        counts = df[dataset_column].replace("", "(blank)").value_counts()

        report_lines.append("\nDataset distribution:")
        for name, count in counts.items():
            report_lines.append(f"  {name}: {count:,}")

    # Column inventory and coverage
    for position, column in enumerate(df.columns):

        series = df[column].map(clean_value)
        nonblank = series.ne("")
        nonblank_count = int(nonblank.sum())
        missing_count = row_count - nonblank_count

        numeric_values = series[nonblank].map(numeric_parse)
        numeric_count = int(numeric_values.notna().sum())

        unique_count = int(series[nonblank].nunique())

        inventory_row = {
            "file_type": dataset_name,
            "column_position": position,
            "column_name": column,
            "normalized_name": normalize_header(column),
        }
        all_inventory.append(inventory_row)

        coverage_row = {
            "file_type": dataset_name,
            "column_name": column,
            "total_rows": row_count,
            "nonblank_count": nonblank_count,
            "missing_or_blank_count": missing_count,
            "nonblank_percentage": round(
                nonblank_count / row_count * 100, 2
            ) if row_count else 0,
            "numeric_parse_count": numeric_count,
            "numeric_parse_percentage_of_all_rows": round(
                numeric_count / row_count * 100, 2
            ) if row_count else 0,
            "unique_nonblank_values": unique_count,
        }
        all_coverage.append(coverage_row)

        # Capture up to five distinct examples.
        examples = series[nonblank].drop_duplicates().head(5).tolist()

        all_samples.append({
            "file_type": dataset_name,
            "column_name": column,
            "example_values": " | ".join(examples),
        })

    # Key and depth checks
    for key_name, key_column in [
        ("PRID", prid_column),
        ("HONU", honu_column)
    ]:
        if key_column:
            usable = df[key_column].map(clean_value)
            usable = usable[usable.ne("")]

            report_lines.append(
                f"{key_name}: {len(usable):,} populated; "
                f"{usable.nunique():,} unique"
            )

    relevant_columns = find_relevant_columns(df.columns)

    report_lines.append("\nPotentially relevant soil columns:")
    for column in relevant_columns:
        report_lines.append(f"  - {column}")

    print(f"Inspection completed: {dataset_name}")


# ---------------------------------------------------------
# Save inspection outputs
# ---------------------------------------------------------

inventory_df = pd.DataFrame(all_inventory)
coverage_df = pd.DataFrame(all_coverage)
samples_df = pd.DataFrame(all_samples)

inventory_df.to_csv(
    OUTPUT_DIR / "01_SOIL_COLUMN_INVENTORY.csv",
    index=False,
    encoding="utf-8-sig"
)

coverage_df.to_csv(
    OUTPUT_DIR / "02_SOIL_COLUMN_COVERAGE.csv",
    index=False,
    encoding="utf-8-sig"
)

samples_df.to_csv(
    OUTPUT_DIR / "03_SOIL_COLUMN_SAMPLE_VALUES.csv",
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------------------------
# Specific feature and depth inventory
# ---------------------------------------------------------

feature_keywords = [
    "depth", "thickness", "sand", "silt", "clay",
    "ph", "ec", "organic carbon", "carbon", "nitrogen",
    "calcium", "magnesium", "potassium", "sodium",
    "sulphate", "sulfate", "chloride", "cec",
    "bulk density", "texture", "carbonate",
    "drainage", "lithology", "parent material",
    "geology", "porosity", "roots", "structure"
]

feature_rows = []

for row in all_coverage:
    column = row["column_name"]
    normalized = normalize_header(column)

    matched_keywords = [
        keyword for keyword in feature_keywords
        if keyword in normalized
    ]

    if matched_keywords:
        feature_rows.append({
            **row,
            "matched_feature_keywords": ", ".join(matched_keywords)
        })

pd.DataFrame(feature_rows).to_csv(
    OUTPUT_DIR / "04_POTENTIAL_SOIL_FEATURES.csv",
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

report_lines.append("\n\nOUTPUT FILES")
report_lines.append("=" * 65)
report_lines.append("01_SOIL_COLUMN_INVENTORY.csv")
report_lines.append("02_SOIL_COLUMN_COVERAGE.csv")
report_lines.append("03_SOIL_COLUMN_SAMPLE_VALUES.csv")
report_lines.append("04_POTENTIAL_SOIL_FEATURES.csv")
report_lines.append("")
report_lines.append(
    "IMPORTANT: Numeric parsing is an inspection only. "
    "It does not confirm measurement units, scientific validity, "
    "or suitability for groundwater prediction."
)

report_path = OUTPUT_DIR / "BOREMAN_SOIL_FEATURE_INSPECTION_REPORT.txt"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("BOREMAN SOIL FEATURE INSPECTION COMPLETED")
print("=" * 65)
print(f"Output folder: {OUTPUT_DIR}")
print(f"Report: {report_path}")
print("\nPlease review the generated CSV files before feature engineering.")
