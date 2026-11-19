
import pandas as pd
from pathlib import Path

file = Path("outputs/step25B_ai_feature_engineering/01_all_records_with_engineered_features.csv")

df = pd.read_csv(file, low_memory=False)

primary = df[
    df["modelling_readiness_group"]
    == "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS"
].copy()

primary["parsed_time"] = pd.to_datetime(
    primary["timestamp_namibia"],
    errors="coerce",
    utc=True,
    format="mixed"
)

primary["parsed_time"] = primary["parsed_time"].dt.tz_convert(
    "Africa/Windhoek"
)

print("\nTOTAL PRIMARY RECORDS:", len(primary))
print("INVALID TIMESTAMPS:", primary["parsed_time"].isna().sum())

print("\nTIMESTAMP EXAMPLES:")
print(primary[["timestamp_namibia", "parsed_time"]].head(10).to_string(index=False))

print("\nSPLIT COUNTS:")
print(primary["proposed_temporal_split"].value_counts())

print("\nSEGMENT COUNTS:")
print(primary.groupby(
    ["proposed_temporal_split", "feature_segment_id"],
    dropna=False
).size().describe())

for split, group in primary.groupby("proposed_temporal_split"):
    group = group.sort_values("parsed_time")

    intervals = (
        group["parsed_time"].diff().dt.total_seconds()
    )

    print("\nSPLIT:", split)
    print("First timestamp:", group["parsed_time"].min())
    print("Last timestamp:", group["parsed_time"].max())
    print("Median interval:", intervals.median(), "seconds")
    print("Minimum interval:", intervals.min(), "seconds")
    print("Maximum interval:", intervals.max(), "seconds")

print("\nLARGEST SEGMENTS:")

groups = primary.groupby(
    ["proposed_temporal_split", "feature_segment_id"]
)

for key, group in groups:
    if len(group) >= 100:
        group = group.sort_values("parsed_time")
        intervals = group["parsed_time"].diff().dt.total_seconds()

        print(
            key,
            "| Rows:", len(group),
            "| First:", group["parsed_time"].min(),
            "| Last:", group["parsed_time"].max(),
            "| Median interval:", intervals.median()
        )

print("\nDIAGNOSTIC COMPLETED")
