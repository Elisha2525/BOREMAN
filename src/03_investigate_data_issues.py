
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT = (
    PROJECT_DIR
    / "outputs"
    / "BOREMAN_OKAHANDJA_100KM_GROUNDWATER.csv"
)

OUT = PROJECT_DIR / "outputs" / "quality_check"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)

print("=" * 60)
print("BOREMAN DATA ISSUE INVESTIGATION")
print("=" * 60)

# 1. Sampling date frequency
print("\n1. MOST COMMON RAW SAMPLING DATES")

dates = (
    df["SAMPLING_D"]
    .value_counts(dropna=False)
    .head(20)
)

print(dates.to_string())

dates.rename_axis("sampling_date").reset_index(
    name="record_count"
).to_csv(OUT / "08_date_frequency.csv", index=False)

# 2. Negative chemical measurements
print("\n2. NEGATIVE CHEMICAL MEASUREMENTS")

parameters = [
    "TOTAL_DISS",
    "SULPHATE_A",
    "FLUORIDE_A",
    "NITRATE_AS"
]

negative_records = []

for col in parameters:
    values = pd.to_numeric(df[col], errors="coerce")
    mask = values.lt(0).fillna(False)

    selected = df.loc[mask].copy()

    for _, row in selected.iterrows():
        negative_records.append({
            "parameter": col,
            "value": row[col],
            "Source": row.get("Source", ""),
            "ID": row.get("ID", ""),
            "SAMPLE_NAM": row.get("SAMPLE_NAM", ""),
            "SAMPLING_D": row.get("SAMPLING_D", ""),
            "LATITUDE": row.get("LATITUDE", ""),
            "LONGITUDE": row.get("LONGITUDE", "")
        })

negative_df = pd.DataFrame(negative_records)

negative_df.to_csv(
    OUT / "09_negative_measurements.csv",
    index=False
)

print(negative_df.to_string(index=False))

# 3. Nitrate censored values
print("\n3. CENSORED NITRATE VALUES")

censored = df[
    df["NITRATE_AS"].str.match(
        r"^\s*<\s*\d",
        na=False
    )
]

print(censored[
    ["Source", "ID", "SAMPLE_NAM", "SAMPLING_D", "NITRATE_AS"]
].to_string(index=False))

censored.to_csv(
    OUT / "10_censored_nitrate_records.csv",
    index=False
)

# 4. Borehole identifier quality
print("\n4. IDENTIFIER QUALITY")

id_values = df["ID"].str.strip()

print("Total records:", len(df))
print("Blank IDs:", (id_values == "").sum())
print("Unknown IDs:", (id_values == "?").sum())

valid = df[
    (df["ID"].str.strip() != "")
    & (df["ID"].str.strip() != "?")
]

print("Records with usable IDs:", len(valid))
print("Distinct IDs:", valid["ID"].nunique())

repeated = (
    valid.groupby(["Source", "ID"])
    .size()
    .reset_index(name="count")
)

repeated = repeated[repeated["count"] > 1]

print("Repeated Source/ID combinations:", len(repeated))

repeated.to_csv(
    OUT / "11_identifier_review.csv",
    index=False
)

print("\nInvestigation completed.")
print("Reports saved in:", OUT)
