
from pathlib import Path
from openpyxl import load_workbook
import pandas as pd

# ==================================================
# 1. PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOIL_FILE = Path(
    r"C:\Users\mayan\dataset\02_NAMIBIA_SOIL\Namibia_Soil_Profile_2025.xlsx"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "soil_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "SOIL_DATASET_INSPECTION.txt"

# ==================================================
# 2. CHECK FILE
# ==================================================

if not SOIL_FILE.exists():
    raise FileNotFoundError(
        f"Soil workbook not found: {SOIL_FILE}"
    )

print("=" * 65)
print("BOREMAN - NAMIBIA SOIL DATASET INSPECTION")
print("=" * 65)

# ==================================================
# 3. OPEN WORKBOOK
# ==================================================

workbook = load_workbook(
    SOIL_FILE,
    read_only=True,
    data_only=True
)

report = []

report.append("BOREMAN SOIL DATASET INSPECTION")
report.append("=" * 65)
report.append(f"Workbook: {SOIL_FILE}")
report.append("")

report.append("WORKSHEETS")
report.append("-" * 50)

for sheet_name in workbook.sheetnames:
    sheet = workbook[sheet_name]

    print(f"\nSheet: {sheet_name}")
    print("Dimensions:", sheet.max_row, "rows x", sheet.max_column, "columns")

    report.append(
        f"{sheet_name}: {sheet.max_row} rows x "
        f"{sheet.max_column} columns"
    )

report.append("")

# ==================================================
# 4. INSPECT HEADERS AND SAMPLE ROWS
# ==================================================

coordinate_keywords = [
    "latitude",
    "longitude",
    "lat",
    "lon",
    "east",
    "north",
    "coord",
    "x",
    "y"
]

identifier_keywords = [
    "prid",
    "profile",
    "site",
    "id",
    "sample",
    "horizon"
]

for sheet_name in workbook.sheetnames:

    sheet = workbook[sheet_name]

    print("\n" + "=" * 65)
    print("WORKSHEET:", sheet_name)
    print("=" * 65)

    report.append("")
    report.append("=" * 65)
    report.append(f"WORKSHEET: {sheet_name}")
    report.append("=" * 65)

    rows = sheet.iter_rows(
        min_row=1,
        max_row=min(sheet.max_row, 6),
        values_only=True
    )

    sample_rows = list(rows)

    if not sample_rows:
        report.append("Empty worksheet")
        continue

    headers = [
        str(value).strip() if value is not None else ""
        for value in sample_rows[0]
    ]

    report.append("\nCOLUMN HEADERS:")

    print("\nCOLUMN HEADERS:")

    for index, header in enumerate(headers):

        if header:
            print(f"{index + 1}. {header}")
            report.append(f"{index + 1}. {header}")

    # Identify likely coordinates and identifiers.
    coordinate_columns = []
    identifier_columns = []

    for header in headers:

        lower = header.lower()

        if any(
            keyword in lower
            for keyword in coordinate_keywords
        ):
            coordinate_columns.append(header)

        if any(
            keyword in lower
            for keyword in identifier_keywords
        ):
            identifier_columns.append(header)

    print("\nPossible coordinate columns:")
    print(coordinate_columns)

    print("\nPossible identifier columns:")
    print(identifier_columns)

    report.append("")
    report.append(
        "Possible coordinate columns: "
        + str(coordinate_columns)
    )

    report.append(
        "Possible identifier columns: "
        + str(identifier_columns)
    )

    report.append("")
    report.append("SAMPLE RECORDS:")

    print("\nSAMPLE RECORDS:")

    for row_number, row in enumerate(
        sample_rows[1:],
        start=2
    ):

        record = dict(zip(headers, row))

        print(f"\nRow {row_number}:")
        print(record)

        report.append(f"\nRow {row_number}:")
        report.append(str(record))

# ==================================================
# 5. SAVE REPORT
# ==================================================

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

workbook.close()

print("\n" + "=" * 65)
print("SOIL INSPECTION COMPLETED")
print("=" * 65)

print("Report saved to:")
print(REPORT_FILE)
