
from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import Transformer
from datetime import datetime

# ============================================================
# BOREMAN - SOIL COORDINATE STANDARDISATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOIL_DIR = PROJECT_DIR / "outputs" / "soil"
VALIDATION_DIR = PROJECT_DIR / "outputs" / "soil_validation"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "soil_wgs84"

WORKBOOK = Path(
    r"C:\Users\mayan\dataset\02_NAMIBIA_SOIL"
    r"\Namibia_Soil_Profile_2025.xlsx"
)

PROFILE_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_PROFILES.csv"
LAB_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_HORIZONS.csv"
FIELD_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_FIELD_HORIZONS.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Okahandja reference point
OKAHANDJA_LAT = -21.98
OKAHANDJA_LON = 16.92
SEARCH_RADIUS_KM = 100


def normalize(value):
    if pd.isna(value):
        return ""
    return "".join(
        char for char in str(value).upper()
        if char.isalnum()
    )


def find_column(df, candidates):
    lookup = {normalize(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize(candidate)
        if key in lookup:
            return lookup[key]

    return None


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0088

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(np.sqrt(a))


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("BOREMAN SOIL COORDINATE STANDARDISATION")
print("=" * 65)

profiles = pd.read_csv(PROFILE_FILE, dtype=str, low_memory=False)
lab = pd.read_csv(LAB_FILE, dtype=str, low_memory=False)
field = pd.read_csv(FIELD_FILE, dtype=str, low_memory=False)

metadata = pd.read_excel(
    WORKBOOK,
    sheet_name="Metadata",
    dtype=str
)

print(f"Soil profiles: {len(profiles):,}")
print(f"Lab horizons: {len(lab):,}")
print(f"Field horizons: {len(field):,}")


# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================

prid_col = find_column(profiles, ["PRID"])
dataset_col = find_column(profiles, ["Dataset"])
lat_col = find_column(profiles, ["Latitude"])
lon_col = find_column(profiles, ["Longitude"])

meta_dataset_col = find_column(metadata, ["DATASET"])
meta_crs_col = find_column(metadata, ["CRS"])

if not all([prid_col, dataset_col, lat_col, lon_col]):
    raise ValueError(
        "Required profile columns PRID, Dataset, Latitude, "
        "or Longitude are missing."
    )

if not all([meta_dataset_col, meta_crs_col]):
    raise ValueError(
        "Metadata sheet must contain DATASET and CRS columns."
    )


# ============================================================
# 3. BUILD DATASET CRS LOOKUP
# ============================================================

metadata["_DATASET_KEY"] = (
    metadata[meta_dataset_col]
    .fillna("")
    .map(normalize)
)

metadata["_CRS_TEXT"] = (
    metadata[meta_crs_col]
    .fillna("")
    .str.strip()
)

crs_lookup = (
    metadata.drop_duplicates("_DATASET_KEY")
    .set_index("_DATASET_KEY")["_CRS_TEXT"]
    .to_dict()
)

profiles["_DATASET_KEY"] = (
    profiles[dataset_col]
    .fillna("")
    .map(normalize)
)

profiles["CRS_METADATA"] = (
    profiles["_DATASET_KEY"]
    .map(crs_lookup)
    .fillna("")
)

profiles["CRS_SOURCE"] = "UNKNOWN"

# Only assign a CRS when the metadata explicitly supports it.
for index, row in profiles.iterrows():
    crs_text = normalize(row["CRS_METADATA"])

    if "WGS84" in crs_text:
        profiles.at[index, "CRS_SOURCE"] = "EPSG:4326"

    elif "SCHWARZECK" in crs_text and "GEOGRAPHIC" in crs_text:
        profiles.at[index, "CRS_SOURCE"] = "EPSG:4293"


# ============================================================
# 4. PREPARE COORDINATE TRANSFORMER
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:4293",
    "EPSG:4326",
    always_xy=True
)

print("\nCoordinate transformation:")
print("Schwarzeck geographic EPSG:4293 -> WGS84 EPSG:4326")
print(f"Transformation description: {transformer.description}")
print(f"Reported accuracy (metres): {transformer.accuracy}")


# ============================================================
# 5. TRANSFORM COORDINATES
# ============================================================

profiles["LATITUDE_ORIGINAL"] = profiles[lat_col]
profiles["LONGITUDE_ORIGINAL"] = profiles[lon_col]

profiles["LATITUDE_NUMERIC"] = pd.to_numeric(
    profiles[lat_col], errors="coerce"
)

profiles["LONGITUDE_NUMERIC"] = pd.to_numeric(
    profiles[lon_col], errors="coerce"
)

profiles["LATITUDE_WGS84"] = np.nan
profiles["LONGITUDE_WGS84"] = np.nan

profiles["COORDINATE_TRANSFORM_STATUS"] = "UNKNOWN_CRS"

for index, row in profiles.iterrows():

    lat = row["LATITUDE_NUMERIC"]
    lon = row["LONGITUDE_NUMERIC"]
    crs = row["CRS_SOURCE"]

    if pd.isna(lat) or pd.isna(lon):
        profiles.at[
            index, "COORDINATE_TRANSFORM_STATUS"
        ] = "INVALID_OR_MISSING_COORDINATES"
        continue

    if crs == "EPSG:4326":
        profiles.at[index, "LATITUDE_WGS84"] = lat
        profiles.at[index, "LONGITUDE_WGS84"] = lon

        profiles.at[
            index, "COORDINATE_TRANSFORM_STATUS"
        ] = "WGS84_UNCHANGED"

    elif crs == "EPSG:4293":

        # always_xy=True means input order is longitude, latitude.
        lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)

        profiles.at[index, "LATITUDE_WGS84"] = lat_wgs84
        profiles.at[index, "LONGITUDE_WGS84"] = lon_wgs84

        profiles.at[
            index, "COORDINATE_TRANSFORM_STATUS"
        ] = "SCHWARZECK_TO_WGS84"

    else:
        profiles.at[
            index, "COORDINATE_TRANSFORM_STATUS"
        ] = "UNVERIFIED_CRS_NOT_TRANSFORMED"


# ============================================================
# 6. VALIDATE TRANSFORMED COORDINATES
# ============================================================

profiles["WGS84_COORDINATES_VALID"] = (
    profiles["LATITUDE_WGS84"].between(-90, 90)
    & profiles["LONGITUDE_WGS84"].between(-180, 180)
)

profiles["NAMIBIA_RANGE_CHECK"] = (
    profiles["LATITUDE_WGS84"].between(-29, -16)
    & profiles["LONGITUDE_WGS84"].between(11, 26)
)

profiles["DISTANCE_FROM_OKAHANDJA_KM"] = haversine_km(
    profiles["LATITUDE_WGS84"].to_numpy(dtype=float),
    profiles["LONGITUDE_WGS84"].to_numpy(dtype=float),
    OKAHANDJA_LAT,
    OKAHANDJA_LON
)

profiles["WITHIN_100KM_OKAHANDJA"] = (
    profiles["DISTANCE_FROM_OKAHANDJA_KM"]
    <= SEARCH_RADIUS_KM
)

# Unknown CRS records cannot be considered spatially verified.
profiles.loc[
    profiles["COORDINATE_TRANSFORM_STATUS"]
    == "UNVERIFIED_CRS_NOT_TRANSFORMED",
    "WITHIN_100KM_OKAHANDJA"
] = False


# ============================================================
# 7. CREATE OUTPUTS
# ============================================================

# Preserve all original fields and add normalised coordinates.
profiles.to_csv(
    OUTPUT_DIR / "BOREMAN_SOIL_PROFILES_WGS84.csv",
    index=False
)

# Profiles with verified WGS84 coordinates.
verified = profiles[
    profiles["COORDINATE_TRANSFORM_STATUS"].isin([
        "WGS84_UNCHANGED",
        "SCHWARZECK_TO_WGS84"
    ])
].copy()

verified.to_csv(
    OUTPUT_DIR / "BOREMAN_SOIL_PROFILES_VERIFIED_WGS84.csv",
    index=False
)

# Verified profiles within the 100 km study radius.
local_verified = verified[
    verified["WITHIN_100KM_OKAHANDJA"]
].copy()

local_verified.to_csv(
    OUTPUT_DIR / "BOREMAN_SOIL_PROFILES_VERIFIED_100KM.csv",
    index=False
)

# Profiles requiring CRS investigation.
unknown = profiles[
    profiles["COORDINATE_TRANSFORM_STATUS"]
    == "UNVERIFIED_CRS_NOT_TRANSFORMED"
].copy()

unknown.to_csv(
    OUTPUT_DIR / "BOREMAN_SOIL_PROFILES_UNKNOWN_CRS.csv",
    index=False
)

# Coordinate transformation summary by dataset.
summary = (
    profiles.groupby(
        [dataset_col, "CRS_SOURCE",
         "COORDINATE_TRANSFORM_STATUS"],
        dropna=False
    )
    .size()
    .reset_index(name="PROFILE_COUNT")
)

summary.to_csv(
    OUTPUT_DIR / "COORDINATE_TRANSFORMATION_SUMMARY.csv",
    index=False
)


# ============================================================
# 8. FINAL REPORT
# ============================================================

status_counts = profiles[
    "COORDINATE_TRANSFORM_STATUS"
].value_counts(dropna=False)

report = [
    "=" * 70,
    "BOREMAN SOIL COORDINATE STANDARDISATION REPORT",
    "=" * 70,
    f"Generated: {datetime.now()}",
    "",
    "INPUT COUNTS",
    "-" * 70,
    f"Total profiles: {len(profiles):,}",
    "",
    "TRANSFORMATION STATUS",
    "-" * 70
]

for status, count in status_counts.items():
    report.append(f"{status}: {count:,}")

report.extend([
    "",
    "SPATIAL VALIDATION",
    "-" * 70,
    f"Verified WGS84 profiles: {len(verified):,}",
    f"Verified profiles within 100 km: {len(local_verified):,}",
    f"Unknown CRS profiles: {len(unknown):,}",
    f"Valid WGS84 coordinate pairs: "
    f"{int(profiles['WGS84_COORDINATES_VALID'].sum()):,}",
    f"Within approximate Namibia range: "
    f"{int(profiles['NAMIBIA_RANGE_CHECK'].sum()):,}",
    "",
    "IMPORTANT LIMITATIONS",
    "-" * 70,
    "- Original coordinates are preserved.",
    "- Schwarzeck geographic coordinates were transformed using EPSG:4293.",
    "- WGS84 coordinates were retained without transformation.",
    "- Unknown CRS records were not transformed.",
    "- Unknown CRS records are excluded from verified spatial results.",
    "- The 100 km filter is recalculated using normalised coordinates.",
    "- Review the CRS metadata and transformation accuracy before modelling.",
    "",
    "OUTPUT FOLDER",
    str(OUTPUT_DIR),
    "=" * 70
])

report_path = OUTPUT_DIR / "BOREMAN_COORDINATE_REPORT.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n" + "\n".join(report))
print("\nCOORDINATE STANDARDISATION COMPLETED")
