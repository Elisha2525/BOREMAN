
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from pyproj import CRS

# ============================================================
# BOREMAN
# GROUNDWATER TO SOIL SPATIAL MATCHING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

GROUNDWATER_FILE = (
    PROJECT_DIR
    / "outputs"
    / "cleaned"
    / "BOREMAN_OKAHANDJA_QUALITY_FLAGGED.csv"
)

SOIL_FILE = (
    PROJECT_DIR
    / "outputs"
    / "soil_wgs84"
    / "BOREMAN_SOIL_PROFILES_VERIFIED_WGS84.csv"
)

ATLAS_EXTRACT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "atlas_4_15"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "spatial_matching"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 5
EARTH_RADIUS_KM = 6371.0088

# ============================================================
# 1. VERIFY GROUNDWATER COORDINATE REFERENCE SYSTEM
# ============================================================

def verify_groundwater_crs():

    print("\nChecking groundwater coordinate reference system...")

    if not ATLAS_EXTRACT_DIR.exists():
        raise FileNotFoundError(
            f"Atlas directory not found:\n{ATLAS_EXTRACT_DIR}"
        )

    prj_files = list(
        ATLAS_EXTRACT_DIR.rglob("WQ_boreholes.prj")
    )

    if not prj_files:
        raise FileNotFoundError(
            "WQ_boreholes.prj was not found. "
            "Cannot verify groundwater CRS."
        )

    prj_file = prj_files[0]

    wkt = prj_file.read_text(
        encoding="utf-8",
        errors="replace"
    )

    source_crs = CRS.from_wkt(wkt)

    detected_epsg = source_crs.to_epsg()

    print(f"PRJ file: {prj_file}")
    print(f"Detected CRS: {source_crs.name}")
    print(f"Detected EPSG: {detected_epsg}")

    if not source_crs.is_geographic:
        raise ValueError(
            "Groundwater CRS is not geographic. "
            "Distance calculation stopped."
        )

    if detected_epsg != 4326:
        raise ValueError(
            f"Expected EPSG:4326 but detected EPSG:{detected_epsg}. "
            "Distance calculation stopped."
        )

    print(
        "Groundwater CRS verified as WGS84 (EPSG:4326)."
    )


# ============================================================
# 2. HAVERSINE DISTANCE CALCULATION
# ============================================================

def calculate_distance_matrix(
    groundwater_lat,
    groundwater_lon,
    soil_lat,
    soil_lon
):

    gw_lat = np.radians(
        groundwater_lat.astype(float)
    )[:, None]

    gw_lon = np.radians(
        groundwater_lon.astype(float)
    )[:, None]

    soil_lat_rad = np.radians(
        soil_lat.astype(float)
    )[None, :]

    soil_lon_rad = np.radians(
        soil_lon.astype(float)
    )[None, :]

    dlat = soil_lat_rad - gw_lat
    dlon = soil_lon_rad - gw_lon

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(gw_lat)
        * np.cos(soil_lat_rad)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    distances = (
        2
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )

    return distances


# ============================================================
# 3. LOAD GROUNDWATER AND SOIL DATA
# ============================================================

def load_datasets():

    print("\nLoading datasets...")

    if not GROUNDWATER_FILE.exists():
        raise FileNotFoundError(
            f"Groundwater file not found:\n{GROUNDWATER_FILE}"
        )

    if not SOIL_FILE.exists():
        raise FileNotFoundError(
            f"Verified soil file not found:\n{SOIL_FILE}"
        )

    groundwater = pd.read_csv(
        GROUNDWATER_FILE,
        dtype=str,
        low_memory=False
    )

    soil = pd.read_csv(
        SOIL_FILE,
        dtype=str,
        low_memory=False
    )

    print(
        f"Groundwater records loaded: {len(groundwater):,}"
    )

    print(
        f"Verified soil profiles loaded: {len(soil):,}"
    )

    return groundwater, soil


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_columns(groundwater, soil):

    print("\nChecking required columns...")

    groundwater_required = [
        "LATITUDE",
        "LONGITUDE"
    ]

    soil_required = [
        "PRID",
        "LATITUDE_WGS84",
        "LONGITUDE_WGS84"
    ]

    for column in groundwater_required:

        if column not in groundwater.columns:
            raise ValueError(
                f"Groundwater column missing: {column}"
            )

    for column in soil_required:

        if column not in soil.columns:
            raise ValueError(
                f"Soil column missing: {column}"
            )

    if "BOREMAN_RECORD_ID" not in groundwater.columns:

        groundwater["BOREMAN_RECORD_ID"] = [
            f"GW_ROW_{i + 1:05d}"
            for i in range(len(groundwater))
        ]

    print("All required columns found.")

    return groundwater, soil


# ============================================================
# 5. VALIDATE COORDINATES
# ============================================================

def validate_coordinates(groundwater, soil):

    print("\nValidating coordinates...")

    groundwater["_LAT"] = pd.to_numeric(
        groundwater["LATITUDE"],
        errors="coerce"
    )

    groundwater["_LON"] = pd.to_numeric(
        groundwater["LONGITUDE"],
        errors="coerce"
    )

    soil["_LAT"] = pd.to_numeric(
        soil["LATITUDE_WGS84"],
        errors="coerce"
    )

    soil["_LON"] = pd.to_numeric(
        soil["LONGITUDE_WGS84"],
        errors="coerce"
    )

    valid_gw_mask = (
        groundwater["_LAT"].between(-90, 90)
        & groundwater["_LON"].between(-180, 180)
    )

    valid_soil_mask = (
        soil["_LAT"].between(-90, 90)
        & soil["_LON"].between(-180, 180)
    )

    valid_gw = groundwater[
        valid_gw_mask
    ].copy().reset_index(drop=True)

    invalid_gw = groundwater[
        ~valid_gw_mask
    ].copy()

    valid_soil = soil[
        valid_soil_mask
    ].copy().reset_index(drop=True)

    invalid_soil = soil[
        ~valid_soil_mask
    ].copy()

    if len(valid_gw) == 0:
        raise ValueError(
            "No valid groundwater coordinates found."
        )

    if len(valid_soil) == 0:
        raise ValueError(
            "No valid soil coordinates found."
        )

    if valid_soil["PRID"].duplicated().any():
        raise ValueError(
            "Duplicate PRID found in verified soil profiles."
        )

    invalid_gw.to_csv(
        OUTPUT_DIR / "INVALID_GROUNDWATER_COORDINATES.csv",
        index=False
    )

    invalid_soil.to_csv(
        OUTPUT_DIR / "INVALID_SOIL_COORDINATES.csv",
        index=False
    )

    print(
        f"Valid groundwater coordinates: {len(valid_gw):,}"
    )

    print(
        f"Invalid groundwater coordinates: {len(invalid_gw):,}"
    )

    print(
        f"Valid soil coordinates: {len(valid_soil):,}"
    )

    print(
        f"Invalid soil coordinates: {len(invalid_soil):,}"
    )

    return valid_gw, valid_soil


# ============================================================
# 6. CALCULATE DISTANCES
# ============================================================

def calculate_distances(valid_gw, valid_soil):

    print("\nCalculating distances between boreholes and soil...")

    distance_matrix = calculate_distance_matrix(
        valid_gw["_LAT"].to_numpy(),
        valid_gw["_LON"].to_numpy(),
        valid_soil["_LAT"].to_numpy(),
        valid_soil["_LON"].to_numpy()
    )

    print("Distance matrix created.")

    print(
        f"Distance matrix dimensions: "
        f"{distance_matrix.shape[0]} x "
        f"{distance_matrix.shape[1]}"
    )

    number_to_select = min(
        TOP_N,
        len(valid_soil)
    )

    nearest_indices = np.argsort(
        distance_matrix,
        axis=1
    )[:, :number_to_select]

    nearest_distances = np.take_along_axis(
        distance_matrix,
        nearest_indices,
        axis=1
    )

    print(
        f"Nearest {number_to_select} soil profiles "
        "identified for each groundwater record."
    )

    return (
        nearest_indices,
        nearest_distances,
        number_to_select
    )


# ============================================================
# 7. CREATE TOP FIVE SOIL MATCHES
# ============================================================

def create_candidates(
    valid_gw,
    valid_soil,
    nearest_indices,
    nearest_distances,
    number_to_select
):

    print("\nCreating soil candidate dataset...")

    gw_indices = np.repeat(
        np.arange(len(valid_gw)),
        number_to_select
    )

    soil_indices = nearest_indices.flatten()

    candidate_groundwater = (
        valid_gw.iloc[gw_indices]
        .reset_index(drop=True)
        .add_prefix("GW_")
    )

    candidate_soil = (
        valid_soil.iloc[soil_indices]
        .reset_index(drop=True)
        .add_prefix("SOIL_")
    )

    candidates = pd.concat(
        [
            candidate_groundwater,
            candidate_soil
        ],
        axis=1
    )

    candidates["SOIL_RANK"] = np.tile(
        np.arange(1, number_to_select + 1),
        len(valid_gw)
    )

    candidates["DISTANCE_KM"] = (
        nearest_distances.flatten().round(4)
    )

    candidates["DISTANCE_BAND"] = pd.cut(
        candidates["DISTANCE_KM"],
        bins=[
            -0.001,
            10,
            25,
            50,
            100,
            np.inf
        ],
        labels=[
            "0-10 KM",
            "10-25 KM",
            "25-50 KM",
            "50-100 KM",
            "OVER 100 KM"
        ]
    )

    candidates = candidates.sort_values(
        [
            "GW_BOREMAN_RECORD_ID",
            "SOIL_RANK"
        ]
    )

    output_file = (
        OUTPUT_DIR
        / "BOREMAN_BOREHOLE_SOIL_CANDIDATES_TOP5.csv"
    )

    candidates.to_csv(
        output_file,
        index=False
    )

    print(
        f"Top-five candidate dataset saved: {output_file}"
    )

    return candidates


# ============================================================
# 8. CREATE NEAREST SOIL MATCH DATASET
# ============================================================

def create_nearest_dataset(candidates):

    print("\nCreating nearest soil profile dataset...")

    nearest = candidates[
        candidates["SOIL_RANK"] == 1
    ].copy()

    output_file = (
        OUTPUT_DIR
        / "BOREMAN_BOREHOLE_NEAREST_SOIL.csv"
    )

    nearest.to_csv(
        output_file,
        index=False
    )

    print(
        f"Nearest soil dataset saved: {output_file}"
    )

    return nearest


# ============================================================
# 9. DISTANCE SUMMARIES
# ============================================================

def create_distance_summaries(nearest):

    print("\nPreparing distance summaries...")

    distance_summary = (
        nearest["DISTANCE_BAND"]
        .value_counts(sort=False, dropna=False)
        .rename_axis("DISTANCE_BAND")
        .reset_index(name="BOREHOLE_COUNT")
    )

    distance_summary.to_csv(
        OUTPUT_DIR
        / "BOREHOLE_NEAREST_SOIL_DISTANCE_SUMMARY.csv",
        index=False
    )

    thresholds = []

    total = len(nearest)

    for threshold in [10, 25, 50, 100]:

        count = int(
            (
                nearest["DISTANCE_KM"]
                <= threshold
            ).sum()
        )

        percentage = (
            count / total * 100
            if total
            else 0
        )

        thresholds.append({
            "MAX_DISTANCE_KM": threshold,
            "BOREHOLES_WITHIN_DISTANCE": count,
            "TOTAL_VALID_BOREHOLES": total,
            "PERCENT_OF_VALID_BOREHOLES": round(
                percentage,
                2
            )
        })

    threshold_df = pd.DataFrame(thresholds)

    threshold_df.to_csv(
        OUTPUT_DIR
        / "BOREHOLES_WITHIN_DISTANCE_THRESHOLDS.csv",
        index=False
    )

    return distance_summary, threshold_df


# ============================================================
# 10. SOIL PROFILE USAGE
# ============================================================

def create_soil_usage_summary(nearest):

    soil_usage = (
        nearest.groupby("SOIL_PRID")
        .size()
        .reset_index(name="NEAREST_BOREHOLE_COUNT")
    )

    soil_usage = soil_usage.sort_values(
        "NEAREST_BOREHOLE_COUNT",
        ascending=False
    )

    soil_usage.to_csv(
        OUTPUT_DIR / "SOIL_PROFILE_NEAREST_USAGE.csv",
        index=False
    )

    return soil_usage


# ============================================================
# 11. GENERATE FINAL REPORT
# ============================================================

def generate_report(
    groundwater,
    valid_gw,
    valid_soil,
    invalid_gw,
    candidates,
    nearest,
    distance_summary,
    threshold_df,
    soil_usage
):

    report = [
        "=" * 70,
        "BOREMAN SPATIAL MATCHING REPORT",
        "=" * 70,
        f"Generated: {datetime.now()}",
        "",
        "INPUT DATA",
        "-" * 70,
        f"Groundwater records loaded: {len(groundwater):,}",
        f"Valid groundwater records: {len(valid_gw):,}",
        f"Invalid groundwater records: {invalid_gw:,}",
        f"Verified soil profiles: {len(valid_soil):,}",
        "",
        "MATCHING RESULTS",
        "-" * 70,
        f"Nearest soil matches: {len(nearest):,}",
        f"Top-five candidate rows: {len(candidates):,}",
        f"Unique soil profiles used as nearest match: "
        f"{soil_usage['SOIL_PRID'].nunique():,}",
        "",
        "NEAREST SOIL DISTANCE DISTRIBUTION",
        "-" * 70
    ]

    for _, row in distance_summary.iterrows():

        report.append(
            f"{row['DISTANCE_BAND']}: "
            f"{int(row['BOREHOLE_COUNT']):,}"
        )

    report.extend([
        "",
        "CUMULATIVE DISTANCE THRESHOLDS",
        "-" * 70
    ])

    for _, row in threshold_df.iterrows():

        report.append(
            f"Within {int(row['MAX_DISTANCE_KM'])} km: "
            f"{int(row['BOREHOLES_WITHIN_DISTANCE']):,} "
            f"({row['PERCENT_OF_VALID_BOREHOLES']}%)"
        )

    report.extend([
        "",
        "SCIENTIFIC LIMITATIONS",
        "-" * 70,
        "- Nearest soil profile does not necessarily represent a borehole.",
        "- Distance alone does not establish geological similarity.",
        "- Soil horizons remain separate from groundwater measurements.",
        "- No original soil or groundwater measurements were modified.",
        "- No spatial matching threshold has been imposed for modelling.",
        "- Spatial proximity does not establish a causal relationship.",
        "",
        "OUTPUT FOLDER",
        str(OUTPUT_DIR),
        "=" * 70
    ])

    report_text = "\n".join(report)

    report_file = (
        OUTPUT_DIR
        / "BOREMAN_SPATIAL_MATCHING_REPORT.txt"
    )

    report_file.write_text(
        report_text,
        encoding="utf-8"
    )

    print("\n" + report_text)

    print("\nSPATIAL MATCHING COMPLETED")
    print(f"Output folder: {OUTPUT_DIR}")


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("BOREMAN GROUNDWATER-SOIL SPATIAL MATCHING")
    print("=" * 70)

    # Verify CRS
    verify_groundwater_crs()

    # Load datasets
    groundwater, soil = load_datasets()

    # Validate columns
    groundwater, soil = validate_columns(
        groundwater,
        soil
    )

    # Validate coordinates
    valid_gw, valid_soil = validate_coordinates(
        groundwater,
        soil
    )

    invalid_gw = len(groundwater) - len(valid_gw)

    # Calculate nearest profiles
    (
        nearest_indices,
        nearest_distances,
        number_to_select
    ) = calculate_distances(
        valid_gw,
        valid_soil
    )

    # Create top-five candidates
    candidates = create_candidates(
        valid_gw,
        valid_soil,
        nearest_indices,
        nearest_distances,
        number_to_select
    )

    # Create nearest-soil dataset
    nearest = create_nearest_dataset(
        candidates
    )

    # Create distance summaries
    distance_summary, threshold_df = (
        create_distance_summaries(nearest)
    )

    # Soil profile usage
    soil_usage = create_soil_usage_summary(
        nearest
    )

    # Final report
    generate_report(
        groundwater,
        valid_gw,
        valid_soil,
        invalid_gw,
        candidates,
        nearest,
        distance_summary,
        threshold_df,
        soil_usage
    )


if __name__ == "__main__":
    main()
