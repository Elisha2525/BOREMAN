
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOIL_DIR = PROJECT_DIR / "outputs" / "soil"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "soil_validation"

LAB_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_HORIZONS.csv"
FIELD_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_FIELD_HORIZONS.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_column(value):
    return "".join(
        char for char in str(value).upper()
        if char.isalnum()
    )


def find_column(df, candidates):
    lookup = {
        normalize_column(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column(candidate)
        if key in lookup:
            return lookup[key]

    return None


def normalize_id(series):
    return series.fillna("").astype(str).str.strip().str.upper()


def normalize_honu(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""

    value = str(value).strip()

    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return str(number)
    except ValueError:
        return value.upper()


print("\n" + "=" * 65)
print("BOREMAN LABORATORY-FIELD HORIZON PAIRING VALIDATION")
print("=" * 65)

lab = pd.read_csv(LAB_FILE, dtype=str, low_memory=False)
field = pd.read_csv(FIELD_FILE, dtype=str, low_memory=False)

print(f"Laboratory horizons: {len(lab):,}")
print(f"Field horizons: {len(field):,}")

# Identify columns
lab_prid = find_column(lab, ["PRID"])
lab_honu = find_column(lab, ["HONU"])

field_prid = find_column(field, ["PRID"])
field_honu = find_column(field, ["HONU"])

required = [
    lab_prid, lab_honu,
    field_prid, field_honu
]

if any(col is None for col in required):
    raise ValueError(
        "Could not find PRID and HONU in both datasets."
    )

# Create normalized matching keys
lab["_PRID_KEY"] = normalize_id(lab[lab_prid])
field["_PRID_KEY"] = normalize_id(field[field_prid])

lab["_HONU_KEY"] = lab[lab_honu].apply(normalize_honu)
field["_HONU_KEY"] = field[field_honu].apply(normalize_honu)

lab["_PAIR_KEY"] = (
    lab["_PRID_KEY"] + "|" + lab["_HONU_KEY"]
)

field["_PAIR_KEY"] = (
    field["_PRID_KEY"] + "|" + field["_HONU_KEY"]
)

# Identify missing identifiers
lab_missing = lab[
    (lab["_PRID_KEY"] == "") |
    (lab["_HONU_KEY"] == "")
].copy()

field_missing = field[
    (field["_PRID_KEY"] == "") |
    (field["_HONU_KEY"] == "")
].copy()

# Only complete keys participate in matching
lab_valid = lab[
    (lab["_PRID_KEY"] != "") &
    (lab["_HONU_KEY"] != "")
].copy()

field_valid = field[
    (field["_PRID_KEY"] != "") &
    (field["_HONU_KEY"] != "")
].copy()

# Find duplicate profile-horizon combinations
lab_duplicates = lab_valid[
    lab_valid.duplicated("_PAIR_KEY", keep=False)
].copy()

field_duplicates = field_valid[
    field_valid.duplicated("_PAIR_KEY", keep=False)
].copy()

# Unique key sets
lab_keys = set(lab_valid["_PAIR_KEY"])
field_keys = set(field_valid["_PAIR_KEY"])

matched_keys = lab_keys & field_keys
lab_only_keys = lab_keys - field_keys
field_only_keys = field_keys - lab_keys

lab_only = lab_valid[
    lab_valid["_PAIR_KEY"].isin(lab_only_keys)
].copy()

field_only = field_valid[
    field_valid["_PAIR_KEY"].isin(field_only_keys)
].copy()

# Save detailed outputs
lab_missing.to_csv(
    OUTPUT_DIR / "14_lab_missing_prid_or_honu.csv",
    index=False
)

field_missing.to_csv(
    OUTPUT_DIR / "15_field_missing_prid_or_honu.csv",
    index=False
)

lab_duplicates.to_csv(
    OUTPUT_DIR / "16_duplicate_lab_horizon_keys.csv",
    index=False
)

field_duplicates.to_csv(
    OUTPUT_DIR / "17_duplicate_field_horizon_keys.csv",
    index=False
)

lab_only.to_csv(
    OUTPUT_DIR / "18_lab_horizons_without_field_match.csv",
    index=False
)

field_only.to_csv(
    OUTPUT_DIR / "19_field_horizons_without_lab_match.csv",
    index=False
)

# Build key comparison table
comparison = pd.DataFrame({
    "PAIR_KEY": sorted(matched_keys)
})

comparison.to_csv(
    OUTPUT_DIR / "20_matched_lab_field_horizon_keys.csv",
    index=False
)

# Report
report = [
    "=" * 70,
    "BOREMAN HORIZON PAIRING VALIDATION REPORT",
    "=" * 70,
    f"Generated: {datetime.now()}",
    "",
    "INPUT COUNTS",
    "-" * 70,
    f"Laboratory rows: {len(lab):,}",
    f"Field rows: {len(field):,}",
    "",
    "IDENTIFIER QUALITY",
    "-" * 70,
    f"Lab rows missing PRID or HONU: {len(lab_missing):,}",
    f"Field rows missing PRID or HONU: {len(field_missing):,}",
    "",
    "DUPLICATE HORIZON KEYS",
    "-" * 70,
    f"Lab duplicate rows: {len(lab_duplicates):,}",
    f"Field duplicate rows: {len(field_duplicates):,}",
    "",
    "EXACT HORIZON MATCHING",
    "-" * 70,
    f"Unique lab horizon keys: {len(lab_keys):,}",
    f"Unique field horizon keys: {len(field_keys):,}",
    f"Matched horizon keys: {len(matched_keys):,}",
    f"Lab-only horizon keys: {len(lab_only_keys):,}",
    f"Field-only horizon keys: {len(field_only_keys):,}",
    "",
    "INTERPRETATION",
    "-" * 70,
    (
        "EXACT KEY SETS MATCH"
        if lab_keys == field_keys
        else "KEY SETS DO NOT MATCH - REVIEW UNMATCHED FILES"
    ),
    (
        "No duplicate horizon keys found."
        if len(lab_duplicates) == 0 and len(field_duplicates) == 0
        else "Duplicate horizon keys require investigation."
    ),
    "",
    "IMPORTANT",
    "-" * 70,
    "- Matching uses normalized PRID and HONU.",
    "- No measurements have been merged or modified.",
    "- Duplicate keys are reported, not automatically removed.",
    "- Equal key sets do not prove that measurements are scientifically equivalent.",
    "",
    "OUTPUT FOLDER",
    str(OUTPUT_DIR),
    "=" * 70
]

report_text = "\n".join(report)

report_file = OUTPUT_DIR / "HORIZON_PAIRING_REPORT.txt"
report_file.write_text(report_text, encoding="utf-8")

print("\n" + report_text)
print("\nHORIZON PAIRING VALIDATION COMPLETED")
