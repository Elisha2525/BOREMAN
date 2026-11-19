
from pathlib import Path
import pandas as pd
import numpy as np

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOIL_FILE = Path(
    r"C:\Users\mayan\dataset\02_NAMIBIA_SOIL\Namibia_Soil_Profile_2025.xlsx"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "soil"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OKAHANDJA_LAT = -21.98
OKAHANDJA_LON = 16.92
SEARCH_RADIUS_KM = 100

# ==================================================
# DISTANCE CALCULATION
# ==================================================

def haversine(lat1, lon1, lat2, lon2):

    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * earth_radius
        * np.arcsin(np.sqrt(a))
    )

# ==================================================
# LOAD WORKBOOK
# ==================================================

if not SOIL_FILE.exists():
    raise FileNotFoundError(SOIL_FILE)

print("=" * 65)
print("BOREMAN OKAHANDJA SOIL EXTRACTION")
print("=" * 65)

sites = pd.read_excel(
    SOIL_FILE,
    sheet_name="Reg&Site"
)

lab = pd.read_excel(
    SOIL_FILE,
    sheet_name="Hor_lab"
)

field = pd.read_excel(
    SOIL_FILE,
    sheet_name="Hor_field"
)

# Remove accidental empty columns.
for frame in [sites, lab, field]:
    frame.drop(
        columns=[
            col for col in frame.columns
            if str(col).strip() == ""
        ],
        inplace=True,
        errors="ignore"
    )

print("\nWORKBOOK COUNTS")
print("Soil profiles:", len(sites))
print("Laboratory horizons:", len(lab))
print("Field horizons:", len(field))

# ==================================================
# VALIDATE REQUIRED COLUMNS
# ==================================================

required_site_columns = [
    "PRID",
    "Latitude",
    "Longitude"
]

for col in required_site_columns:
    if col not in sites.columns:
        raise ValueError(
            f"Missing site column: {col}"
        )

for frame, name in [
    (lab, "Hor_lab"),
    (field, "Hor_field")
]:
    if "PRID" not in frame.columns:
        raise ValueError(
            f"PRID missing in {name}"
        )

# ==================================================
# CLEAN COORDINATES
# ==================================================

sites["Latitude"] = pd.to_numeric(
    sites["Latitude"],
    errors="coerce"
)

sites["Longitude"] = pd.to_numeric(
    sites["Longitude"],
    errors="coerce"
)

valid_coordinates = (
    sites["Latitude"].between(-90, 90)
    & sites["Longitude"].between(-180, 180)
)

sites = sites.loc[
    valid_coordinates
].copy()

# ==================================================
# CALCULATE DISTANCE
# ==================================================

sites["DISTANCE_FROM_OKAHANDJA_KM"] = haversine(
    OKAHANDJA_LAT,
    OKAHANDJA_LON,
    sites["Latitude"].to_numpy(),
    sites["Longitude"].to_numpy()
)

local_sites = sites[
    sites["DISTANCE_FROM_OKAHANDJA_KM"]
    <= SEARCH_RADIUS_KM
].copy()

local_ids = set(
    local_sites["PRID"].dropna().astype(str)
)

# ==================================================
# FILTER HORIZON DATA
# ==================================================

lab_local = lab[
    lab["PRID"].astype(str).isin(local_ids)
].copy()

field_local = field[
    field["PRID"].astype(str).isin(local_ids)
].copy()

# ==================================================
# SAVE DATASETS
# ==================================================

local_sites.to_csv(
    OUTPUT_DIR / "BOREMAN_OKAHANDJA_SOIL_PROFILES.csv",
    index=False,
    encoding="utf-8-sig"
)

lab_local.to_csv(
    OUTPUT_DIR / "BOREMAN_OKAHANDJA_SOIL_HORIZONS.csv",
    index=False,
    encoding="utf-8-sig"
)

field_local.to_csv(
    OUTPUT_DIR / "BOREMAN_OKAHANDJA_SOIL_FIELD_HORIZONS.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# REPORT
# ==================================================

report = []

report.append("BOREMAN OKAHANDJA SOIL EXTRACTION")
report.append("=" * 55)

report.append(
    f"Original site rows: {len(sites)}"
)

report.append(
    f"Local soil profile rows: {len(local_sites)}"
)

report.append(
    f"Unique local profile IDs: {local_sites['PRID'].nunique()}"
)

report.append(
    f"Local laboratory horizon rows: {len(lab_local)}"
)

report.append(
    f"Local field horizon rows: {len(field_local)}"
)

report.append("")

report.append("SOIL PROFILES BY DATASET")

report.append(
    local_sites["Dataset"]
    .value_counts(dropna=False)
    .to_string()
)

report.append("")

report.append("DISTANCE SUMMARY (KM)")

report.append(
    local_sites[
        "DISTANCE_FROM_OKAHANDJA_KM"
    ].describe().to_string()
)

report.append("")

report.append("PROFILE LINKAGE")

report.append(
    "Laboratory profiles not found in local site IDs: "
    + str(
        len(
            set(lab_local["PRID"].dropna().astype(str))
            - local_ids
        )
    )
)

report.append(
    "Field profiles not found in local site IDs: "
    + str(
        len(
            set(field_local["PRID"].dropna().astype(str))
            - local_ids
        )
    )
)

(OUTPUT_DIR / "SOIL_EXTRACTION_REPORT.txt").write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("SOIL EXTRACTION COMPLETED")
print("=" * 65)

print("Local soil profile rows:", len(local_sites))
print("Unique local profiles:", local_sites["PRID"].nunique())
print("Local lab horizons:", len(lab_local))
print("Local field horizons:", len(field_local))

print("\nProfiles by dataset:")
print(
    local_sites["Dataset"]
    .value_counts(dropna=False).to_string()
)

print("\nOutput folder:")
print(OUTPUT_DIR)
