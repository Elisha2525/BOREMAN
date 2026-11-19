
from pathlib import Path
import zipfile
import shapefile
import math
import csv

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

ZIP_FILE = Path(
    r"C:\Users\mayan\dataset\03_NAMIBIA_ATLAS\Atlas_Groundwater_Quality_4_15.zip"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# THIS VARIABLE WAS MISSING
EXTRACT_DIR = PROCESSED_DIR / "atlas_4_15"

OKAHANDJA_LAT = -21.98
OKAHANDJA_LON = 16.92
SEARCH_RADIUS_KM = 100

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Boreman extraction started")
print("Project:", PROJECT_DIR)
print("Source:", ZIP_FILE)
print("Output:", OUTPUT_DIR)
# ==================================================
# STEP 2: EXTRACT ATLAS SHAPEFILE
# ==================================================

print("\nExtracting Atlas groundwater data...")

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

required_files = {
    "wq_boreholes.shp",
    "wq_boreholes.shx",
    "wq_boreholes.dbf",
    "wq_boreholes.prj",
    "wq_boreholes.cpg"
}

with zipfile.ZipFile(ZIP_FILE, "r") as archive:
    for member in archive.namelist():
        if Path(member).name.lower() in required_files:
            target = (EXTRACT_DIR / member).resolve()

            if not target.is_relative_to(EXTRACT_DIR.resolve()):
                raise ValueError("Unsafe archive path")

            archive.extract(member, EXTRACT_DIR)

# Locate extracted Shapefile
shapefiles = list(EXTRACT_DIR.rglob("WQ_boreholes.shp"))

if not shapefiles:
    raise FileNotFoundError("WQ_boreholes.shp not found")

SHP_FILE = shapefiles[0]

print("Shapefile found:", SHP_FILE)

# ==================================================
# STEP 3: READ GROUNDWATER RECORDS
# ==================================================

sf = shapefile.Reader(
    str(SHP_FILE),
    encoding="utf-8"
)

fields = [field[0] for field in sf.fields[1:]]

print("\nOriginal records:", sf.numRecords)
print("Total fields:", len(fields))
print("Available fields:", fields)

# ==================================================
# STEP 4: CALCULATE DISTANCE FROM OKAHANDJA
# ==================================================

def calculate_distance(lat1, lon1, lat2, lon2):

    radius = 6371

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

# ==================================================
# STEP 5: PROCESS ALL RECORDS
# ==================================================

national_records = []
local_records = []
invalid_records = 0

print("\nProcessing groundwater records...")

for record in sf.iterRecords():

    row = dict(zip(fields, list(record)))

    try:
        lat = float(row["LATITUDE"])
        lon = float(row["LONGITUDE"])

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid coordinates")

    except (ValueError, TypeError):
        invalid_records += 1
        continue

    distance = calculate_distance(
        OKAHANDJA_LAT,
        OKAHANDJA_LON,
        lat,
        lon
    )

    row["distance_km_from_okahandja"] = round(distance, 3)

    national_records.append(row)

    if distance <= SEARCH_RADIUS_KM:
        local_records.append(row)

# ==================================================
# STEP 6: EXPORT CSV FILES
# ==================================================

def export_csv(filepath, records):

    if not records:
        raise ValueError(
            f"No records available for export: {filepath}"
        )

    columns = list(records[0].keys())

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(records)


national_csv = OUTPUT_DIR / "BOREMAN_NAMIBIA_GROUNDWATER.csv"

local_csv = OUTPUT_DIR / "BOREMAN_OKAHANDJA_100KM_GROUNDWATER.csv"

export_csv(national_csv, national_records)
export_csv(local_csv, local_records)

# ==================================================
# STEP 7: CREATE AUDIT REPORT
# ==================================================

report = OUTPUT_DIR / "BOREMAN_ATLAS_AUDIT.txt"

with open(report, "w", encoding="utf-8") as file:

    file.write("BOREMAN GROUNDWATER DATA AUDIT\n")
    file.write("=" * 50 + "\n")

    file.write(f"\nOriginal records: {sf.numRecords}")
    file.write(f"\nValid coordinate records: {len(national_records)}")
    file.write(f"\nInvalid coordinate records: {invalid_records}")
    file.write(f"\nRecords within 100 km: {len(local_records)}")

    file.write("\n\nAvailable fields:\n")

    for field in fields:
        file.write(f"\n- {field}")

    file.write("\n\nData limitations:")
    file.write("\n- Units need verification.")
    file.write("\n- Sampling dates need verification.")
    file.write("\n- Censored measurements must be preserved.")
    file.write("\n- Repeated borehole records need review.")

# ==================================================
# STEP 8: FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("BOREMAN DATA EXTRACTION COMPLETED")
print("=" * 60)

print("Original records:", sf.numRecords)
print("Valid records:", len(national_records))
print("Invalid coordinate records:", invalid_records)
print("Okahandja records:", len(local_records))

print("\nGenerated files:")

print(national_csv)
print(local_csv)
print(report)

print("\nNext step: groundwater data quality assessment.")
