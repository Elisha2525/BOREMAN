
from pathlib import Path
import pandas as pd
import numpy as np
import re

# =========================================================
# BOREMAN - STEP 13
# SOIL FEATURE VALIDATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "outputs" / "engineered_soil"
OUTPUT_DIR = BASE_DIR / "outputs" / "soil_feature_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_FILE = INPUT_DIR / "01_BOREMAN_ENGINEERED_HORIZON_FEATURES.csv"
PROFILE_FILE = INPUT_DIR / "02_BOREMAN_ENGINEERED_PROFILE_FEATURES.csv"
DEPTH_FILE = INPUT_DIR / "03_BOREMAN_DEPTH_RESOLVED_FEATURES.csv"
LINK_FILE = INPUT_DIR / "04_BOREMAN_GROUNDWATER_SOIL_FEATURE_LINK.csv"
COVERAGE_FILE = INPUT_DIR / "05_ENGINEERED_FEATURE_COVERAGE.csv"

# Expected source definitions to verify.
EXPECTED_SOURCES = {
    "sand_percent": "sdto",
    "silt_percent": "stpc",
    "clay_percent": "clpc",
    "silt_clay_percent": "silt + clay",
}

FEATURES_TO_CHECK = [
    "sand_percent",
    "silt_percent",
    "clay_percent",
    "silt_clay_percent",
    "bulk_density",
    "ph_cacl2",
    "ph_water",
    "ec_2soil5_water_us_cm",
    "ec_2soil5_water_ds_m",
    "ec_saturated_paste_us_cm",
    "ec_saturated_paste_ds_m",
    "organic_carbon",
    "total_nitrogen",
    "carbonate_equivalent",
    "exchangeable_calcium",
    "exchangeable_magnesium",
    "exchangeable_potassium",
    "exchangeable_sodium",
    "cec_effective",
    "cec_measured",
    "phosphorus_olsen",
    "chloride",
    "sulphate",
]


# =========================================================
# HELPERS
# =========================================================

def read_csv(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found:\n{path}")
        return None

    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                path,
                low_memory=False,
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read file: {path}")


def normalize(value):
    value = str(value).lower().strip()
    value = value.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", value)


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def add_result(results, check, status, details):
    results.append({
        "check": check,
        "status": status,
        "details": details
    })


def write_csv(df, filename):
    df.to_csv(
        OUTPUT_DIR / filename,
        index=False,
        encoding="utf-8-sig"
    )


# =========================================================
# 1. LOAD FILES
# =========================================================

print("\n" + "=" * 65)
print("BOREMAN - STEP 13: SOIL FEATURE VALIDATION")
print("=" * 65)

horizon = read_csv(HORIZON_FILE)
profile = read_csv(PROFILE_FILE)
depth = read_csv(DEPTH_FILE)
coverage = read_csv(COVERAGE_FILE)
link = read_csv(LINK_FILE, required=False)

print(f"\nHorizon records: {len(horizon):,}")
print(f"Profile records: {len(profile):,}")
print(f"Depth records: {len(depth):,}")
print(f"Feature coverage rows: {len(coverage):,}")

if link is not None:
    print(f"Groundwater-soil link rows: {len(link):,}")
else:
    print("Groundwater-soil link file not found.")

results = []


# =========================================================
# 2. CHECK BASIC STRUCTURE
# =========================================================

for name, df in [
    ("Horizon dataset", horizon),
    ("Profile dataset", profile),
    ("Depth dataset", depth),
]:
    duplicate_names = df.columns[df.columns.duplicated()].tolist()

    if duplicate_names:
        add_result(
            results,
            f"{name}: duplicate column names",
            "WARNING",
            str(duplicate_names)
        )
    else:
        add_result(
            results,
            f"{name}: duplicate column names",
            "PASS",
            "No duplicate column labels detected."
        )

# Check key availability.
required_horizon_keys = ["PRID_LAB", "HONU_LAB"]

for key in required_horizon_keys:
    if key not in horizon.columns:
        add_result(
            results,
            f"Horizon key: {key}",
            "FAIL",
            "Required key column not found."
        )
    else:
        add_result(
            results,
            f"Horizon key: {key}",
            "PASS",
            "Column exists."
        )

if "PRID" in profile.columns:
    profile_prids = profile["PRID"].astype(str).str.strip()
elif "_PRID_KEY" in profile.columns:
    profile_prids = profile["_PRID_KEY"].astype(str).str.strip()
else:
    profile_prids = None

if profile_prids is not None:
    duplicate_prids = int(profile_prids.duplicated().sum())

    add_result(
        results,
        "Profile PRID uniqueness",
        "PASS" if duplicate_prids == 0 else "FAIL",
        f"Duplicate PRID rows: {duplicate_prids}"
    )


# =========================================================
# 3. VERIFY FEATURE SOURCE MAPPING
# =========================================================

coverage_map = {}

for _, row in coverage.iterrows():
    coverage_map[str(row["feature"])] = str(row["source_column"])

source_rows = []

for feature, expected in EXPECTED_SOURCES.items():

    actual = coverage_map.get(feature, "")
    actual_normalized = normalize(actual)

    if expected in actual_normalized:
        status = "PASS"
        details = "Expected source identifier found."
    else:
        status = "REVIEW"
        details = (
            f"Expected source identifier '{expected}', "
            f"actual source column '{actual}'."
        )

    source_rows.append({
        "feature": feature,
        "expected_source_identifier": expected,
        "actual_source_column": actual,
        "status": status,
        "details": details
    })

    add_result(
        results,
        f"Source mapping: {feature}",
        status,
        details
    )

source_validation = pd.DataFrame(source_rows)
write_csv(source_validation, "01_SOURCE_MAPPING_VALIDATION.csv")


# =========================================================
# 4. FEATURE COVERAGE AND NUMERIC RANGE SUMMARY
# =========================================================

range_rows = []

for feature in FEATURES_TO_CHECK:

    if feature not in horizon.columns:
        range_rows.append({
            "feature": feature,
            "status": "NOT_FOUND",
            "numeric_count": 0,
            "missing_count": len(horizon),
            "minimum": np.nan,
            "median": np.nan,
            "maximum": np.nan,
            "negative_count": 0,
            "zero_count": 0
        })
        continue

    values = numeric(horizon[feature])
    valid = values.dropna()

    range_rows.append({
        "feature": feature,
        "status": "AVAILABLE" if len(valid) else "NO_NUMERIC_VALUES",
        "numeric_count": len(valid),
        "missing_count": int(values.isna().sum()),
        "minimum": valid.min() if len(valid) else np.nan,
        "median": valid.median() if len(valid) else np.nan,
        "maximum": valid.max() if len(valid) else np.nan,
        "negative_count": int((valid < 0).sum()),
        "zero_count": int((valid == 0).sum())
    })

range_df = pd.DataFrame(range_rows)
write_csv(range_df, "02_FEATURE_RANGE_SUMMARY.csv")


# =========================================================
# 5. FLAG UNUSUAL VALUES
# =========================================================

anomaly_rows = []

for feature in FEATURES_TO_CHECK:

    if feature not in horizon.columns:
        continue

    values = numeric(horizon[feature])

    for idx, value in values.items():

        if pd.isna(value):
            continue

        reason = None

        if feature.startswith("ph_"):
            if value < 0 or value > 14:
                reason = "pH_OUTSIDE_0_TO_14"

        elif feature in [
            "sand_percent",
            "silt_percent",
            "clay_percent",
            "silt_clay_percent",
            "organic_carbon",
            "chloride",
            "sulphate"
        ]:
            if value < 0:
                reason = "NEGATIVE_PERCENTAGE"
            elif value > 100:
                reason = "PERCENTAGE_OVER_100"

        elif feature == "bulk_density":
            if value <= 0:
                reason = "NONPOSITIVE_BULK_DENSITY"

        elif feature.startswith("ec_"):
            if value < 0:
                reason = "NEGATIVE_EC"

        elif feature in [
            "total_nitrogen",
            "carbonate_equivalent",
            "exchangeable_calcium",
            "exchangeable_magnesium",
            "exchangeable_potassium",
            "exchangeable_sodium",
            "cec_effective",
            "cec_measured",
            "phosphorus_olsen"
        ]:
            if value < 0:
                reason = "NEGATIVE_CHEMICAL_VALUE"

        if reason:
            anomaly_rows.append({
                "horizon_row_index": idx,
                "PRID": horizon.loc[idx, "PRID_LAB"]
                    if "PRID_LAB" in horizon.columns else "",
                "HONU": horizon.loc[idx, "HONU_LAB"]
                    if "HONU_LAB" in horizon.columns else "",
                "feature": feature,
                "value": value,
                "flag": reason
            })

anomalies = pd.DataFrame(anomaly_rows)

if anomalies.empty:
    anomalies = pd.DataFrame(
        columns=[
            "horizon_row_index", "PRID", "HONU",
            "feature", "value", "flag"
        ]
    )

write_csv(anomalies, "03_UNUSUAL_VALUE_FLAGS.csv")

add_result(
    results,
    "Basic unusual-value screening",
    "PASS" if anomalies.empty else "REVIEW",
    f"{len(anomalies):,} values flagged for review. "
    "Flags are screening indicators, not proof of errors."
)


# =========================================================
# 6. EC UNIT CONVERSION CHECKS
# =========================================================

ec_pairs = [
    (
        "ec_2soil5_water_us_cm",
        "ec_2soil5_water_ds_m",
        "2 soil:5 water"
    ),
    (
        "ec_saturated_paste_us_cm",
        "ec_saturated_paste_ds_m",
        "saturated paste"
    )
]

ec_rows = []

for us_col, ds_col, method in ec_pairs:

    if us_col not in horizon.columns or ds_col not in horizon.columns:
        ec_rows.append({
            "method": method,
            "paired_count": 0,
            "median_absolute_difference": np.nan,
            "maximum_absolute_difference": np.nan,
            "status": "MISSING_COLUMN"
        })
        continue

    us = numeric(horizon[us_col])
    ds = numeric(horizon[ds_col])

    paired = pd.DataFrame({
        "us": us,
        "ds": ds
    }).dropna()

    if paired.empty:
        ec_rows.append({
            "method": method,
            "paired_count": 0,
            "median_absolute_difference": np.nan,
            "maximum_absolute_difference": np.nan,
            "status": "NO_PAIRED_VALUES"
        })
        continue

    # 1 dS/m = 1000 µS/cm.
    expected_us = paired["ds"] * 1000
    difference = (paired["us"] - expected_us).abs()

    tolerance = np.maximum(expected_us.abs() * 0.01, 0.1)
    within_tolerance = difference <= tolerance

    status = (
        "PASS"
        if within_tolerance.all()
        else "REVIEW"
    )

    ec_rows.append({
        "method": method,
        "paired_count": len(paired),
        "median_absolute_difference": difference.median(),
        "maximum_absolute_difference": difference.max(),
        "within_1_percent_or_0_1_us_cm": int(
            within_tolerance.sum()
        ),
        "outside_tolerance": int(
            (~within_tolerance).sum()
        ),
        "status": status
    })

    add_result(
        results,
        f"EC unit consistency: {method}",
        status,
        f"{int(within_tolerance.sum())} of {len(paired)} "
        "paired values agree within tolerance."
    )

ec_validation = pd.DataFrame(ec_rows)
write_csv(ec_validation, "04_EC_UNIT_VALIDATION.csv")


# =========================================================
# 7. PARTICLE SIZE / TEXTURE CHECKS
# =========================================================

texture_rows = []

required_texture = [
    "sand_percent",
    "silt_percent",
    "clay_percent",
    "silt_clay_percent"
]

if all(col in horizon.columns for col in required_texture):

    texture = horizon[required_texture].apply(numeric)

    # Check whether silt + clay matches the separately supplied
    # combined silt + clay field where all three exist.
    combined = texture.dropna(
        subset=[
            "silt_percent",
            "clay_percent",
            "silt_clay_percent"
        ]
    ).copy()

    if not combined.empty:

        calculated = (
            combined["silt_percent"]
            + combined["clay_percent"]
        )

        difference = (
            calculated - combined["silt_clay_percent"]
        ).abs()

        for idx in combined.index:
            texture_rows.append({
                "horizon_row_index": idx,
                "PRID": horizon.loc[idx, "PRID_LAB"]
                    if "PRID_LAB" in horizon.columns else "",
                "HONU": horizon.loc[idx, "HONU_LAB"]
                    if "HONU_LAB" in horizon.columns else "",
                "silt_plus_clay_calculated": calculated.loc[idx],
                "silt_plus_clay_reported": combined.loc[
                    idx, "silt_clay_percent"
                ],
                "absolute_difference": difference.loc[idx],
                "within_1_percentage_point": (
                    difference.loc[idx] <= 1
                )
            })

texture_df = pd.DataFrame(texture_rows)

if texture_df.empty:
    texture_df = pd.DataFrame(
        columns=[
            "horizon_row_index", "PRID", "HONU",
            "silt_plus_clay_calculated",
            "silt_plus_clay_reported",
            "absolute_difference",
            "within_1_percentage_point"
        ]
    )

write_csv(texture_df, "05_TEXTURE_CONSISTENCY_CHECK.csv")

if len(texture_df):
    texture_pass = int(
        texture_df["within_1_percentage_point"].sum()
    )

    add_result(
        results,
        "Silt plus clay consistency",
        "PASS" if texture_pass == len(texture_df) else "REVIEW",
        f"{texture_pass} of {len(texture_df)} paired horizons "
        "agree within one percentage point."
    )
else:
    add_result(
        results,
        "Silt plus clay consistency",
        "NOT_ASSESSED",
        "No horizons contained all three measurements."
    )

add_result(
    results,
    "Sand particle-size definition",
    "REVIEW",
    "Source label includes '<53 µm'. Confirm the dataset "
    "definition before treating this as conventional sand "
    "or applying sand+silt+clay closure."
)


# =========================================================
# 8. DEPTH CONSISTENCY
# =========================================================

depth_rows = []

required_depth_columns = [
    "upper_depth_cm_numeric",
    "lower_depth_cm_numeric",
    "horizon_thickness_cm_numeric",
    "depth_midpoint_cm",
    "depth_interval_valid",
    "depth_band"
]

if all(col in depth.columns for col in required_depth_columns):

    for idx, row in depth.iterrows():

        upper = row["upper_depth_cm_numeric"]
        lower = row["lower_depth_cm_numeric"]
        thickness = row["horizon_thickness_cm_numeric"]

        status = "PASS"
        reason = ""

        if pd.notna(upper) and pd.notna(lower):

            if lower <= upper:
                status = "REVIEW"
                reason = "LOWER_DEPTH_NOT_GREATER_THAN_UPPER"

            elif pd.notna(thickness):
                expected = lower - upper

                if abs(expected - thickness) > 1:
                    status = "REVIEW"
                    reason = "THICKNESS_DOES_NOT_MATCH_DEPTH_INTERVAL"

        depth_rows.append({
            "row_index": idx,
            "PRID": row["PRID"],
            "HONU": row["HONU"],
            "upper_depth_cm": upper,
            "lower_depth_cm": lower,
            "reported_thickness_cm": thickness,
            "calculated_thickness_cm": (
                lower - upper
                if pd.notna(upper) and pd.notna(lower)
                else np.nan
            ),
            "status": status,
            "reason": reason
        })

depth_check = pd.DataFrame(depth_rows)
write_csv(depth_check, "06_DEPTH_VALIDATION.csv")

depth_issues = (
    int((depth_check["status"] != "PASS").sum())
    if len(depth_check) else 0
)

add_result(
    results,
    "Depth interval consistency",
    "PASS" if depth_issues == 0 else "REVIEW",
    f"{depth_issues:,} depth records flagged."
)


# =========================================================
# 9. PROFILE AND GROUNDWATER LINK CHECKS
# =========================================================

if link is not None:

    if "_SOIL_PRID_KEY" in link.columns:
        soil_key = link["_SOIL_PRID_KEY"].astype(str).str.strip()
    elif "PRID" in link.columns:
        soil_key = link["PRID"].astype(str).str.strip()
    else:
        soil_key = None

    if soil_key is not None:

        profile_key_set = set(profile_prids) if profile_prids is not None else set()

        matched = soil_key.isin(profile_key_set)

        add_result(
            results,
            "Groundwater-to-soil PRID matching",
            "PASS" if matched.all() else "REVIEW",
            f"{int(matched.sum()):,} of {len(link):,} rows "
            "reference an engineered soil profile."
        )

        link_summary = pd.DataFrame([{
            "groundwater_soil_link_rows": len(link),
            "unique_soil_prids": soil_key.nunique(),
            "matched_soil_prid_rows": int(matched.sum()),
            "unmatched_soil_prid_rows": int((~matched).sum())
        }])

        write_csv(link_summary, "07_GROUNDWATER_SOIL_LINK_CHECK.csv")

    else:
        add_result(
            results,
            "Groundwater-to-soil PRID matching",
            "NOT_ASSESSED",
            "Could not identify soil PRID key in link file."
        )


# =========================================================
# 10. FEATURE MISSINGNESS SUMMARY
# =========================================================

missing_rows = []

for feature in FEATURES_TO_CHECK:

    if feature not in horizon.columns:
        continue

    values = numeric(horizon[feature])

    missing_rows.append({
        "feature": feature,
        "total_horizons": len(values),
        "numeric_values": int(values.notna().sum()),
        "missing_or_unparsed": int(values.isna().sum()),
        "coverage_percent": round(
            values.notna().mean() * 100, 2
        )
    })

missingness = pd.DataFrame(missing_rows)
write_csv(missingness, "08_FEATURE_MISSINGNESS.csv")


# =========================================================
# 11. FINAL VALIDATION REPORT
# =========================================================

results_df = pd.DataFrame(results)
write_csv(results_df, "09_VALIDATION_RESULTS.csv")

status_counts = results_df["status"].value_counts().to_dict()

report = [
    "BOREMAN - STEP 13 SOIL FEATURE VALIDATION REPORT",
    "=" * 70,
    f"Horizon records: {len(horizon):,}",
    f"Profile records: {len(profile):,}",
    f"Depth records: {len(depth):,}",
    "",
    "VALIDATION STATUS COUNTS",
    "-" * 70,
]

for status, count in status_counts.items():
    report.append(f"{status}: {count}")

report.extend([
    "",
    "IMPORTANT SCIENTIFIC NOTES",
    "-" * 70,
    "1. Source mapping is checked against expected source identifiers.",
    "2. Range flags are screening indicators, not proof of errors.",
    "3. EC conversion uses 1 dS/m = 1000 µS/cm.",
    "4. Soil EC methods are kept separate.",
    "5. Sand particle-size definition requires source documentation.",
    "6. Silt + clay consistency is checked independently.",
    "7. No missing values are imputed.",
    "8. No soil measurements are treated as groundwater measurements.",
    "9. No prediction model is trained in this step.",
    "",
    "OUTPUT FILES",
    "-" * 70,
    "01_SOURCE_MAPPING_VALIDATION.csv",
    "02_FEATURE_RANGE_SUMMARY.csv",
    "03_UNUSUAL_VALUE_FLAGS.csv",
    "04_EC_UNIT_VALIDATION.csv",
    "05_TEXTURE_CONSISTENCY_CHECK.csv",
    "06_DEPTH_VALIDATION.csv",
    "07_GROUNDWATER_SOIL_LINK_CHECK.csv (if applicable)",
    "08_FEATURE_MISSINGNESS.csv",
    "09_VALIDATION_RESULTS.csv",
])

report_path = OUTPUT_DIR / "BOREMAN_SOIL_FEATURE_VALIDATION_REPORT.txt"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("BOREMAN SOIL FEATURE VALIDATION COMPLETED")
print("=" * 65)

print("\nValidation status:")
print(results_df["status"].value_counts().to_string())

print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"Report: {report_path}")
