from pathlib import Path
import pandas as pd
import numpy as np
import re
from datetime import datetime

# ============================================================
# BOREMAN - SOIL DATASET VALIDATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOIL_DIR = PROJECT_DIR / "outputs" / "soil"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "soil_validation"

WORKBOOK = Path(
    r"C:\Users\mayan\dataset\02_NAMIBIA_SOIL"
    r"\Namibia_Soil_Profile_2025.xlsx"
)

PROFILE_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_PROFILES.csv"
LAB_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_HORIZONS.csv"
FIELD_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_FIELD_HORIZONS.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(value):
    """Normalize column names and identifiers for comparison."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def find_column(df, candidates):
    """Find a column using normalized candidate names."""
    lookup = {normalize(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize(candidate)
        if key in lookup:
            return lookup[key]

    return None


def read_csv_checked(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    df = pd.read_csv(path, dtype=str, low_memory=False)

    print(f"{description}: {len(df):,} rows loaded")
    return df


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def write_csv(df, filename):
    df.to_csv(OUTPUT_DIR / filename, index=False)


# ============================================================
# 1. LOAD EXTRACTED DATA
# ============================================================

print("\n" + "=" * 65)
print("BOREMAN SOIL DATA VALIDATION")
print("=" * 65)

profiles = read_csv_checked(
    PROFILE_FILE, "Soil profiles"
)

lab = read_csv_checked(
    LAB_FILE, "Laboratory horizons"
)

field = read_csv_checked(
    FIELD_FILE, "Field horizons"
)

# Identify important columns
profile_id_col = find_column(profiles, ["PRID"])
profile_dataset_col = find_column(profiles, ["Dataset"])
profile_lat_col = find_column(profiles, ["Latitude"])
profile_lon_col = find_column(profiles, ["Longitude"])

lab_id_col = find_column(lab, ["PRID"])
field_id_col = find_column(field, ["PRID"])

if not profile_id_col:
    raise ValueError("PRID column not found in soil profiles.")

if not lab_id_col:
    raise ValueError("PRID column not found in laboratory horizons.")

if not field_id_col:
    raise ValueError("PRID column not found in field horizons.")

if not profile_lat_col or not profile_lon_col:
    raise ValueError(
        "Latitude or Longitude column not found in profiles."
    )


# ============================================================
# 2. PROFILE IDENTIFIER VALIDATION
# ============================================================

print("\nChecking profile identifiers...")

profiles["_PRID"] = profiles[profile_id_col].fillna("").str.strip()
lab["_PRID"] = lab[lab_id_col].fillna("").str.strip()
field["_PRID"] = field[field_id_col].fillna("").str.strip()

valid_profiles = profiles[profiles["_PRID"] != ""]

profile_ids = set(valid_profiles["_PRID"])

duplicate_profile_rows = valid_profiles[
    valid_profiles["_PRID"].duplicated(keep=False)
].copy()

lab_orphans = lab[
    (lab["_PRID"] != "") &
    (~lab["_PRID"].isin(profile_ids))
].copy()

field_orphans = field[
    (field["_PRID"] != "") &
    (~field["_PRID"].isin(profile_ids))
].copy()

lab_missing_id = lab[lab["_PRID"] == ""].copy()
field_missing_id = field[field["_PRID"] == ""].copy()

write_csv(
    duplicate_profile_rows,
    "01_duplicate_profile_ids.csv"
)

write_csv(
    lab_orphans,
    "02_lab_horizons_without_profile.csv"
)

write_csv(
    field_orphans,
    "03_field_horizons_without_profile.csv"
)

write_csv(
    lab_missing_id,
    "04_lab_horizons_missing_prid.csv"
)

write_csv(
    field_missing_id,
    "05_field_horizons_missing_prid.csv"
)


# ============================================================
# 3. COORDINATE VALIDATION
# ============================================================

print("\nChecking coordinates...")

profiles["_LAT"] = safe_numeric(profiles[profile_lat_col])
profiles["_LON"] = safe_numeric(profiles[profile_lon_col])

profiles["_COORDINATE_STATUS"] = "VALID_NUMERIC"

profiles.loc[
    profiles["_LAT"].isna() | profiles["_LON"].isna(),
    "_COORDINATE_STATUS"
] = "MISSING_OR_NON_NUMERIC"

profiles.loc[
    profiles["_LAT"].notna() &
    ((profiles["_LAT"] < -90) | (profiles["_LAT"] > 90)),
    "_COORDINATE_STATUS"
] = "INVALID_LATITUDE"

profiles.loc[
    profiles["_LON"].notna() &
    ((profiles["_LON"] < -180) | (profiles["_LON"] > 180)),
    "_COORDINATE_STATUS"
] = "INVALID_LONGITUDE"

# Approximate Namibia geographic plausibility check.
# This is NOT a CRS conversion or proof of WGS84.
namibia_plausible = (
    profiles["_LAT"].between(-29, -16) &
    profiles["_LON"].between(11, 26)
)

profiles["_NAMIBIA_RANGE_CHECK"] = np.where(
    namibia_plausible,
    "WITHIN_APPROXIMATE_NAMIBIA_RANGE",
    "OUTSIDE_APPROXIMATE_NAMIBIA_RANGE"
)

coordinate_summary = (
    profiles.groupby(
        ["_COORDINATE_STATUS", "_NAMIBIA_RANGE_CHECK"],
        dropna=False
    )
    .size()
    .reset_index(name="RECORD_COUNT")
)

write_csv(
    coordinate_summary,
    "06_coordinate_quality_summary.csv"
)

write_csv(
    profiles[
        profiles["_COORDINATE_STATUS"] != "VALID_NUMERIC"
    ],
    "07_invalid_or_missing_coordinates.csv"
)

write_csv(
    profiles[
        profiles["_NAMIBIA_RANGE_CHECK"] !=
        "WITHIN_APPROXIMATE_NAMIBIA_RANGE"
    ],
    "08_profiles_outside_namibia_range.csv"
)


# ============================================================
# 4. DATASET-SPECIFIC CRS METADATA
# ============================================================

print("\nChecking coordinate reference system metadata...")

metadata = pd.read_excel(
    WORKBOOK,
    sheet_name="Metadata",
    dtype=str
)

metadata_dataset_col = find_column(metadata, ["DATASET"])
metadata_crs_col = find_column(metadata, ["CRS"])

if metadata_dataset_col and metadata_crs_col:
    metadata["_DATASET_KEY"] = (
        metadata[metadata_dataset_col]
        .fillna("")
        .map(normalize)
    )

    metadata["_CRS_VALUE"] = (
        metadata[metadata_crs_col]
        .fillna("NOT SPECIFIED")
        .str.strip()
    )

    crs_map = (
        metadata.drop_duplicates("_DATASET_KEY")
        .set_index("_DATASET_KEY")["_CRS_VALUE"]
        .to_dict()
    )
else:
    crs_map = {}

if profile_dataset_col:
    profiles["_DATASET_KEY"] = (
        profiles[profile_dataset_col]
        .fillna("")
        .map(normalize)
    )

    profiles["_CRS_METADATA"] = profiles[
        "_DATASET_KEY"
    ].map(crs_map).fillna("NO MATCHING METADATA")

    crs_summary = (
        profiles.groupby(
            [profile_dataset_col, "_CRS_METADATA"],
            dropna=False
        )
        .size()
        .reset_index(name="PROFILE_COUNT")
    )
else:
    profiles["_CRS_METADATA"] = "DATASET COLUMN NOT FOUND"

    crs_summary = pd.DataFrame({
        "STATUS": ["DATASET COLUMN NOT FOUND"],
        "PROFILE_COUNT": [len(profiles)]
    })

write_csv(
    crs_summary,
    "09_dataset_crs_summary.csv"
)


# ============================================================
# 5. LABORATORY MEASUREMENT COVERAGE
# ============================================================

print("\nChecking laboratory measurement coverage...")

# These are candidate measurements based on workbook inspection.
# The script searches for likely column names and reports coverage.
candidate_terms = [
    "PHCA",
    "PHAQ",
    "PHKC",
    "EL25",
    "ELCO",
    "CLPC",
    "SDTO",
    "STPC",
    "CEC",
    "CA",
    "MG",
    "NA",
    "K",
    "CL",
    "SO4",
    "NITROGEN",
    "PHOSPHORUS",
    "ORGANICCARBON",
    "BULKDENSITY",
    "TEXTURE"
]

lab_coverage = []

for col in lab.columns:
    if col == "_PRID":
        continue

    normalized_col = normalize(col)

    if not any(
        term in normalized_col
        for term in candidate_terms
    ):
        continue

    numeric_values = safe_numeric(lab[col])

    non_missing = lab[col].notna() & (
        lab[col].astype(str).str.strip() != ""
    )

    numeric_count = int(numeric_values.notna().sum())
    missing_count = int(len(lab) - non_missing.sum())

    lab_coverage.append({
        "COLUMN": col,
        "TOTAL_HORIZON_ROWS": len(lab),
        "NON_MISSING_RAW_VALUES": int(non_missing.sum()),
        "NUMERIC_VALUES": numeric_count,
        "MISSING_OR_BLANK_VALUES": missing_count,
        "NUMERIC_COVERAGE_PERCENT": round(
            numeric_count / len(lab) * 100, 2
        ) if len(lab) else 0
    })

lab_coverage_df = pd.DataFrame(lab_coverage)

if not lab_coverage_df.empty:
    lab_coverage_df = lab_coverage_df.sort_values(
        "NUMERIC_COVERAGE_PERCENT",
        ascending=False
    )

write_csv(
    lab_coverage_df,
    "10_laboratory_measurement_coverage.csv"
)


# ============================================================
# 6. FIELD HORIZON MEASUREMENT COVERAGE
# ============================================================

print("\nChecking field horizon coverage...")

field_coverage = []

for col in field.columns:
    if col == "_PRID":
        continue

    non_missing = field[col].notna() & (
        field[col].astype(str).str.strip() != ""
    )

    field_coverage.append({
        "COLUMN": col,
        "TOTAL_HORIZON_ROWS": len(field),
        "NON_MISSING_VALUES": int(non_missing.sum()),
        "MISSING_OR_BLANK_VALUES": int(
            len(field) - non_missing.sum()
        ),
        "COVERAGE_PERCENT": round(
            non_missing.sum() / len(field) * 100, 2
        ) if len(field) else 0
    })

field_coverage_df = pd.DataFrame(field_coverage)

if not field_coverage_df.empty:
    field_coverage_df = field_coverage_df.sort_values(
        "COVERAGE_PERCENT",
        ascending=False
    )

write_csv(
    field_coverage_df,
    "11_field_measurement_coverage.csv"
)


# ============================================================
# 7. HORIZON DEPTH VALIDATION
# ============================================================

print("\nChecking soil horizon depth fields...")

depth_candidates = [
    "Upper Depth cm",
    "Lower Depth cm",
    "Hor thickness cm"
]

depth_results = []

for description, df in [
    ("LABORATORY", lab),
    ("FIELD", field)
]:
    for candidate in depth_candidates:
        col = find_column(df, [candidate])

        if not col:
            continue

        numeric = safe_numeric(df[col])

        depth_results.append({
            "TABLE": description,
            "COLUMN": col,
            "TOTAL_ROWS": len(df),
            "NUMERIC_VALUES": int(numeric.notna().sum()),
            "MISSING_OR_UNPARSED": int(numeric.isna().sum()),
            "MINIMUM": numeric.min(),
            "MAXIMUM": numeric.max()
        })

depth_summary = pd.DataFrame(depth_results)

write_csv(
    depth_summary,
    "12_horizon_depth_summary.csv"
)


# ============================================================
# 8. PROFILE-LEVEL HORIZON COUNTS
# ============================================================

lab_counts = (
    lab[lab["_PRID"] != ""]
    .groupby("_PRID")
    .size()
    .rename("LAB_HORIZON_COUNT")
)

field_counts = (
    field[field["_PRID"] != ""]
    .groupby("_PRID")
    .size()
    .rename("FIELD_HORIZON_COUNT")
)

profile_horizon_counts = profiles[
    [profile_id_col]
].copy()

profile_horizon_counts["PRID"] = (
    profile_horizon_counts[profile_id_col]
    .fillna("")
    .str.strip()
)

profile_horizon_counts = profile_horizon_counts.merge(
    lab_counts,
    left_on="PRID",
    right_index=True,
    how="left"
)

profile_horizon_counts = profile_horizon_counts.merge(
    field_counts,
    left_on="PRID",
    right_index=True,
    how="left"
)

profile_horizon_counts[
    ["LAB_HORIZON_COUNT", "FIELD_HORIZON_COUNT"]
] = profile_horizon_counts[
    ["LAB_HORIZON_COUNT", "FIELD_HORIZON_COUNT"]
].fillna(0).astype(int)

profile_horizon_counts["HORIZON_COUNT_MATCH"] = (
    profile_horizon_counts["LAB_HORIZON_COUNT"] ==
    profile_horizon_counts["FIELD_HORIZON_COUNT"]
)

write_csv(
    profile_horizon_counts,
    "13_horizons_per_profile.csv"
)


# ============================================================
# 9. SUMMARY REPORT
# ============================================================

valid_coordinate_count = int(
    (profiles["_COORDINATE_STATUS"] == "VALID_NUMERIC").sum()
)

namibia_range_count = int(namibia_plausible.sum())

matched_lab_count = len(lab) - len(lab_orphans) - len(lab_missing_id)
matched_field_count = (
    len(field) - len(field_orphans) - len(field_missing_id)
)

report = []

report.append("=" * 70)
report.append("BOREMAN OKAHANDJA SOIL DATA VALIDATION REPORT")
report.append("=" * 70)
report.append(f"Generated: {datetime.now()}")
report.append("")

report.append("1. DATASET COUNTS")
report.append("-" * 70)
report.append(f"Soil profiles: {len(profiles):,}")
report.append(f"Laboratory horizons: {len(lab):,}")
report.append(f"Field horizons: {len(field):,}")
report.append("")

report.append("2. PROFILE IDENTIFIERS")
report.append("-" * 70)
report.append(f"Profiles with usable PRID: {len(valid_profiles):,}")
report.append(
    f"Profiles with duplicate PRID rows: {len(duplicate_profile_rows):,}"
)
report.append(f"Lab horizons without matching profile: {len(lab_orphans):,}")
report.append(
    f"Field horizons without matching profile: {len(field_orphans):,}"
)
report.append(f"Lab horizons missing PRID: {len(lab_missing_id):,}")
report.append(f"Field horizons missing PRID: {len(field_missing_id):,}")
report.append("")

report.append("3. COORDINATE QUALITY")
report.append("-" * 70)
report.append(
    f"Profiles with valid numeric coordinates: {valid_coordinate_count:,}"
)
report.append(
    f"Profiles within approximate Namibia coordinate range: "
    f"{namibia_range_count:,}"
)
report.append(
    "Note: Namibia range check is a plausibility check, not CRS verification."
)
report.append("")

report.append("4. COORDINATE REFERENCE SYSTEMS")
report.append("-" * 70)

for _, row in crs_summary.iterrows():
    report.append(" | ".join(str(value) for value in row.values))

report.append("")
report.append("5. HORIZON LINKAGE")
report.append("-" * 70)
report.append(
    f"Lab horizons with a matching profile: {matched_lab_count:,}"
)
report.append(
    f"Field horizons with a matching profile: {matched_field_count:,}"
)
report.append(
    "Profiles with equal lab and field horizon counts: "
    f"{int(profile_horizon_counts['HORIZON_COUNT_MATCH'].sum()):,}"
)
report.append("")

report.append("6. LABORATORY MEASUREMENT COVERAGE")
report.append("-" * 70)

if not lab_coverage_df.empty:
    for _, row in lab_coverage_df.iterrows():
        report.append(
            f"{row['COLUMN']}: {row['NUMERIC_VALUES']:,} numeric "
            f"({row['NUMERIC_COVERAGE_PERCENT']}%)"
        )
else:
    report.append("No candidate laboratory measurement columns found.")

report.append("")
report.append("7. VALIDATION NOTES")
report.append("-" * 70)
report.append(
    "- Coordinate values have not been transformed between CRS systems."
)
report.append(
    "- Dataset metadata must be reviewed before spatial distance calculations."
)
report.append(
    "- Soil measurements must not be treated as groundwater measurements."
)
report.append(
    "- Horizon rows are not independent soil profiles."
)
report.append(
    "- Numeric coverage does not establish scientific suitability."
)
report.append(
    "- No records have been deleted or chemically modified by this script."
)
report.append("")
report.append("OUTPUT FOLDER")
report.append(str(OUTPUT_DIR))
report.append("=" * 70)

report_text = "\n".join(report)

report_path = OUTPUT_DIR / "BOREMAN_SOIL_VALIDATION_REPORT.txt"
report_path.write_text(report_text, encoding="utf-8")

print("\n" + report_text)

print("\nVALIDATION COMPLETED")
print(f"Reports saved to: {OUTPUT_DIR}")