from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FILE = (
    Path.cwd()
    / "outputs"
    / "step25B_ai_feature_engineering"
    / "01_all_records_with_engineered_features.csv"
)

PRIMARY_GROUP = "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS"

print("=" * 65)
print("BOREMAN STEP 27 - INPUT DIAGNOSTIC")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"\nInput file: {INPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

needed = [
    "timestamp_namibia",
    "modelling_readiness_group",
    "proposed_temporal_split",
    "feature_segment_id",
]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("\nReadiness groups:")
print(df["modelling_readiness_group"].value_counts(dropna=False).to_string())

print("\nSplit labels:")
print(df["proposed_temporal_split"].value_counts(dropna=False).to_string())

primary = df.loc[
    df["modelling_readiness_group"].astype(str).str.strip()
    == PRIMARY_GROUP
].copy()

print(f"\nPrimary rows: {len(primary):,}")

if primary.empty:
    print("\nERROR: No primary candidate rows matched the expected label.")
    print("Actual group labels:")
    for value in df["modelling_readiness_group"].dropna().unique():
        print(repr(value))
    raise SystemExit(1)

primary["timestamp_parsed"] = pd.to_datetime(
    primary["timestamp_namibia"],
    errors="coerce",
    utc=True,
    format="mixed",
)

bad_ts = int(primary["timestamp_parsed"].isna().sum())
print(f"Invalid primary timestamps: {bad_ts:,}")

print("\nPrimary split counts:")
print(primary["proposed_temporal_split"].value_counts(dropna=False).to_string())

print("\nFeature segment ID:")
print("Missing IDs:", int(primary["feature_segment_id"].isna().sum()))
print("Unique IDs:", int(primary["feature_segment_id"].nunique(dropna=False)))

if bad_ts == 0:
    primary["timestamp_parsed"] = primary["timestamp_parsed"].dt.tz_convert(
        "Africa/Windhoek"
    )
    group_sizes = primary.groupby(
        ["proposed_temporal_split", "feature_segment_id"],
        dropna=False
    ).size()

    print("\nGroups by split + feature_segment_id:")
    print("Number of groups:", len(group_sizes))
    print("Groups with at least 2 rows:", int((group_sizes >= 2).sum()))
    print("Largest group:", int(group_sizes.max()))

    tolerance_ns = 90 * 1_000_000_000
    pair_counts = {}

    for split_name, split_df in primary.groupby(
        "proposed_temporal_split", dropna=False
    ):
        split_pairs = 0

        for segment_id, group in split_df.groupby(
            "feature_segment_id", dropna=False
        ):
            group = group.sort_values("timestamp_parsed")
            ts = group["timestamp_parsed"].astype("int64").to_numpy()

            if len(ts) < 2:
                continue

            for horizon in [5, 15, 30, 60]:
                desired = ts + horizon * 60 * 1_000_000_000
                right = np.searchsorted(ts, desired, side="left")
                left = right - 1
                origin = np.arange(len(ts))

                right_ok = (right < len(ts)) & (right > origin)
                left_ok = (left > origin) & (left < len(ts))

                rd = np.full(len(ts), np.iinfo(np.int64).max, dtype=np.int64)
                ld = np.full(len(ts), np.iinfo(np.int64).max, dtype=np.int64)

                rd[right_ok] = np.abs(ts[right[right_ok]] - desired[right_ok])
                ld[left_ok] = np.abs(ts[left[left_ok]] - desired[left[left_ok]])

                use_left = left_ok & (~right_ok | (ld <= rd))
                target = np.where(use_left, left, right)
                distance = np.where(use_left, ld, rd)

                valid = (
                    (target > origin)
                    & (target < len(ts))
                    & (distance <= tolerance_ns)
                )
                split_pairs += int(valid.sum())

        pair_counts[str(split_name)] = split_pairs

    print("\nPotential future-pair counts (all four horizons combined):")
    for split_name, count in pair_counts.items():
        print(f"{split_name}: {count:,}")

print("\nDiagnostic completed.")
