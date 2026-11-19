"""
BOREMAN STEP 23 - CHRONOLOGICAL SPLIT CORRECTION

Reads the already generated Step 23 record-level readiness file and revises
only the proposed temporal split labels and split summary.

It does not alter sensor values, quality flags, readiness groups, or remove rows.
Whole Namibia dates remain together to prevent date-level leakage.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("outputs/step23_modelling_readiness")
INPUT_FILE = OUTPUT_DIR / "01_all_records_with_readiness_labels.csv"
RECORD_OUTPUT = OUTPUT_DIR / "01_all_records_with_readiness_labels.csv"
SPLIT_OUTPUT = OUTPUT_DIR / "05_proposed_chronological_split.csv"
REPORT_OUTPUT = OUTPUT_DIR / "08_step23_split_correction_report.txt"

PRIMARY = "PRIMARY_CANDIDATE_NO_CURRENT_REVIEW_FLAGS"


def stop(message):
    raise ValueError(message)


def main():
    if not INPUT_FILE.exists():
        stop(f"Required Step 23 file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "source_row_number",
        "namibia_date",
        "modelling_readiness_group",
        "proposed_temporal_split",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        stop(f"Missing required columns: {missing}")

    if df["source_row_number"].isna().any():
        stop("Missing source_row_number values.")
    if df["source_row_number"].duplicated().any():
        stop("source_row_number is not unique.")

    original_count = len(df)
    original_ids = set(df["source_row_number"].tolist())

    primary = df[df["modelling_readiness_group"] == PRIMARY]
    if primary.empty:
        stop("No primary candidate records found.")

    # Use only dates containing primary candidates. Keep each date indivisible.
    daily_counts = (
        primary.groupby("namibia_date")
        .size()
        .sort_index()
    )
    dates = daily_counts.index.tolist()
    counts = daily_counts.to_numpy(dtype=int)

    if len(dates) < 3:
        stop("Fewer than three primary-candidate dates; cannot form three splits.")

    total = int(counts.sum())
    targets = np.array([0.70, 0.15, 0.15])

    # Evaluate all chronological two-cut combinations. Choose the split whose
    # record proportions are closest to 70/15/15 while preserving whole dates.
    best = None
    for i in range(1, len(dates) - 1):
        for j in range(i + 1, len(dates)):
            split_counts = np.array([
                counts[:i].sum(),
                counts[i:j].sum(),
                counts[j:].sum(),
            ], dtype=float)
            proportions = split_counts / total
            # Squared proportional error; no shuffling or date interleaving.
            score = float(np.sum((proportions - targets) ** 2))
            if best is None or score < best["score"]:
                best = {
                    "i": i,
                    "j": j,
                    "counts": split_counts.astype(int),
                    "proportions": proportions,
                    "score": score,
                }

    i, j = best["i"], best["j"]
    blocks = [
        ("TRAIN", dates[:i]),
        ("VALIDATION", dates[i:j]),
        ("TEST", dates[j:]),
    ]

    # Reset all rows; only primary candidates receive a split.
    df["proposed_temporal_split"] = "NOT_ASSIGNED"
    split_rows = []
    for split_name, block_dates in blocks:
        mask = (
            (df["modelling_readiness_group"] == PRIMARY)
            & df["namibia_date"].isin(block_dates)
        )
        df.loc[mask, "proposed_temporal_split"] = split_name
        part = df.loc[mask]
        split_rows.append({
            "split": split_name,
            "date_count": len(block_dates),
            "start_date": block_dates[0],
            "end_date": block_dates[-1],
            "record_count": len(part),
            "percentage_of_primary_candidates": round(
                len(part) / total * 100, 3
            ),
        })

    # Confirm that the split is strictly chronological and date-isolated.
    order = {"TRAIN": 0, "VALIDATION": 1, "TEST": 2}
    assigned = df[df["proposed_temporal_split"] != "NOT_ASSIGNED"].copy()
    assigned["_split_order"] = assigned["proposed_temporal_split"].map(order)
    per_date = assigned.groupby("namibia_date")["_split_order"].nunique()
    if (per_date > 1).any():
        stop("A Namibia date appears in more than one split.")
    date_order = assigned.groupby("namibia_date")["_split_order"].first()
    if not date_order.is_monotonic_increasing:
        stop("Chronological order violation detected.")

    if len(df) != original_count:
        stop("Record count changed.")
    if set(df["source_row_number"].tolist()) != original_ids:
        stop("Source record identities changed.")
    if df["source_row_number"].duplicated().any():
        stop("Duplicate source_row_number detected.")

    df.to_csv(RECORD_OUTPUT, index=False)
    pd.DataFrame(split_rows).to_csv(SPLIT_OUTPUT, index=False)

    report = [
        "BOREMAN STEP 23 - CHRONOLOGICAL SPLIT CORRECTION",
        "=" * 58,
        f"Input records: {original_count:,}",
        f"Primary candidate records: {total:,}",
        f"Primary candidate dates: {len(dates)}",
        "Method: chronological date blocks; whole dates stay together.",
        "Target proportions: 70% train, 15% validation, 15% test.",
        "Cut points selected by minimizing squared difference from target",
        "record proportions across all possible chronological date boundaries.",
        "",
        "CORRECTED SPLIT",
    ]
    for row in split_rows:
        report.append(
            f"{row['split']}: {row['record_count']:,} records "
            f"({row['percentage_of_primary_candidates']:.3f}%), "
            f"{row['start_date']} to {row['end_date']} "
            f"({row['date_count']} dates)"
        )
    report.extend([
        "",
        "PRESERVATION CHECKS",
        "PASS: all source records preserved.",
        "PASS: source_row_number remains unique.",
        "PASS: only primary-candidate split labels were assigned.",
        "PASS: no Namibia date crosses split boundaries.",
        "PASS: split order is chronological.",
        "",
        "CAUTION",
        "The test period contains sparse primary-candidate dates after 18 October.",
        "The record proportions are improved, but this does not establish",
        "representativeness, independent validation, or model readiness.",
        "Ground-truth/reference labels and calibration evidence are still required.",
    ])
    REPORT_OUTPUT.write_text("\n".join(report), encoding="utf-8")

    print("\nSTEP 23 SPLIT CORRECTION COMPLETED")
    for row in split_rows:
        print(
            f"{row['split']}: {row['record_count']:,} records "
            f"({row['percentage_of_primary_candidates']:.3f}%) "
            f"| {row['start_date']} to {row['end_date']}"
        )
    print(f"All {original_count:,} records preserved.")
    print("Date-level leakage checks passed.")
    print(f"Updated files: {RECORD_OUTPUT} and {SPLIT_OUTPUT}")
    print(f"Correction report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSPLIT CORRECTION STOPPED: {exc}")
        sys.exit(1)
