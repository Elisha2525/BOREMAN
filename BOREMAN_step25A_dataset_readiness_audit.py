"""
BOREMAN Step 25A - AI Dataset Readiness Audit
Read-only audit of the approved Step 23 dataset.
No source records are changed, deleted, imputed, or overwritten.
"""

from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    r"outputs\step23_modelling_readiness\01_all_records_with_readiness_labels.csv"
)
OUTPUT_DIR = Path(r"outputs\step25_dataset_readiness_audit")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
            "Run this script from the BOREMAN project root directory."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print("\n========== BOREMAN STEP 25A: DATASET READINESS AUDIT ==========")
    print(f"Input: {INPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # 1. Column inventory
    column_inventory = pd.DataFrame({
        "column_name": df.columns,
        "data_type": [str(df[c].dtype) for c in df.columns],
        "non_null_count": [int(df[c].notna().sum()) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [
            round(float(df[c].isna().mean() * 100), 4) for c in df.columns
        ],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    column_inventory.to_csv(
        OUTPUT_DIR / "01_column_inventory.csv", index=False
    )

    # 2. Identify likely timestamp and source-row columns without assuming names
    timestamp_candidates = [
        c for c in df.columns
        if any(token in c.lower() for token in
               ["timestamp_namibia", "timestamp", "datetime", "date_time"])
    ]
    source_row_candidates = [
        c for c in df.columns
        if c.lower() in ["source_row_number", "source_row", "row_number"]
    ]

    print("\nTimestamp-like columns:", timestamp_candidates or "None identified")
    print("Source-row columns:", source_row_candidates or "None identified")

    # 3. Timestamp coverage and parse diagnostics
    timestamp_reports = []
    for col in timestamp_candidates:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        valid = parsed.dropna()
        timestamp_reports.append({
            "column": col,
            "parseable_count": int(parsed.notna().sum()),
            "unparseable_count": int(parsed.isna().sum()),
            "earliest_utc": str(valid.min()) if len(valid) else "",
            "latest_utc": str(valid.max()) if len(valid) else "",
            "unique_timestamps": int(parsed.nunique(dropna=True)),
            "duplicate_timestamp_rows": int(
                parsed.duplicated(keep=False).sum()
            ),
        })
    pd.DataFrame(timestamp_reports).to_csv(
        OUTPUT_DIR / "02_timestamp_audit.csv", index=False
    )

    # 4. Duplicate source-row and full-row checks
    duplicate_reports = []
    for col in source_row_candidates:
        duplicate_reports.append({
            "check": f"duplicate values in {col}",
            "count": int(df[col].duplicated(keep=False).sum()),
        })
    duplicate_reports.extend([
        {
            "check": "exact duplicate full rows",
            "count": int(df.duplicated(keep=False).sum()),
        },
        {
            "check": "duplicate timestamp rows (first detected timestamp column)",
            "count": (
                int(pd.to_datetime(
                    df[timestamp_candidates[0]], errors="coerce", utc=True
                ).duplicated(keep=False).sum())
                if timestamp_candidates else -1
            ),
        },
    ])
    pd.DataFrame(duplicate_reports).to_csv(
        OUTPUT_DIR / "03_duplicate_audit.csv", index=False
    )

    # 5. Numeric sensor summary
    numeric = df.select_dtypes(include="number")
    numeric_summary = numeric.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).transpose()
    numeric_summary.to_csv(OUTPUT_DIR / "04_numeric_summary.csv")

    # 6. Readiness / grouping columns
    group_candidates = [
        c for c in df.columns
        if any(token in c.lower() for token in
               ["readiness", "group", "split", "plateau", "flag", "eligible"])
    ]
    group_rows = []
    for col in group_candidates:
        counts = df[col].value_counts(dropna=False)
        for value, count in counts.items():
            group_rows.append({
                "column": col,
                "value": str(value),
                "record_count": int(count),
                "percentage": round(float(count / len(df) * 100), 4)
                if len(df) else 0,
            })
    pd.DataFrame(group_rows).to_csv(
        OUTPUT_DIR / "05_readiness_and_flag_distributions.csv", index=False
    )

    # 7. Candidate sensor columns
    sensor_candidates = [
        c for c in df.columns
        if c.lower() in ["tds", "humidity", "temperature"]
    ]
    sensor_rows = []
    for col in sensor_candidates:
        s = pd.to_numeric(df[col], errors="coerce")
        sensor_rows.append({
            "sensor": col,
            "numeric_count": int(s.notna().sum()),
            "missing_or_non_numeric": int(s.isna().sum()),
            "minimum": s.min(),
            "maximum": s.max(),
            "mean": s.mean(),
            "standard_deviation": s.std(),
            "unique_values": int(s.nunique(dropna=True)),
        })
    pd.DataFrame(sensor_rows).to_csv(
        OUTPUT_DIR / "06_sensor_readiness.csv", index=False
    )

    # 8. Plain-text report
    report = [
        "BOREMAN STEP 25A - DATASET READINESS AUDIT",
        "=" * 55,
        f"Input file: {INPUT_FILE}",
        f"Total rows: {len(df):,}",
        f"Total columns: {len(df.columns):,}",
        f"Timestamp candidates: {timestamp_candidates}",
        f"Source-row candidates: {source_row_candidates}",
        f"Sensor columns detected: {sensor_candidates}",
        "",
        "COLUMN NAMES:",
        *[f"- {c} ({df[c].dtype})" for c in df.columns],
        "",
        "IMPORTANT:",
        "- This audit is read-only.",
        "- No observations were deleted or modified.",
        "- No missing values were imputed.",
        "- No model was trained.",
        "- Timestamp parsing in this audit is diagnostic only.",
        "- Review the CSV outputs before approving feature preparation.",
    ]
    (OUTPUT_DIR / "07_step25_dataset_readiness_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nAudit outputs saved to:", OUTPUT_DIR)
    print("1. 01_column_inventory.csv")
    print("2. 02_timestamp_audit.csv")
    print("3. 03_duplicate_audit.csv")
    print("4. 04_numeric_summary.csv")
    print("5. 05_readiness_and_flag_distributions.csv")
    print("6. 06_sensor_readiness.csv")
    print("7. 07_step25_dataset_readiness_report.txt")
    print("\nNo source data was changed. No model was trained.")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
