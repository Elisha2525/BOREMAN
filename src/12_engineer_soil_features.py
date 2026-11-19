
from pathlib import Path
import pandas as pd
import numpy as np
import re

# =========================================================
# BOREMAN - STEP 12
# SOIL FEATURE ENGINEERING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOIL_DIR = BASE_DIR / "outputs" / "soil"
MATCH_DIR = BASE_DIR / "outputs" / "spatial_matching"

OUTPUT_DIR = BASE_DIR / "outputs" / "engineered_soil"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_PROFILES.csv"
LAB_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_HORIZONS.csv"
FIELD_FILE = SOIL_DIR / "BOREMAN_OKAHANDJA_SOIL_FIELD_HORIZONS.csv"


# =========================================================
# FEATURE DEFINITIONS
# =========================================================

FEATURE_PATTERNS = {
    "sand_percent": ["sand %"],
    "silt_percent": ["silt %"],
    "clay_percent": ["clay %", "< 2 µm"],
    "silt_clay_percent": ["silt + clay %"],
    "bulk_density": ["bulk density"],
    "ph_cacl2": ["ph cacl2"],
    "ph_water": ["ph water"],
    "ec_2soil5_water_us_cm": [
        "electrical conductivity", "µs/cm", "2 soil:5 water"
    ],
    "ec_2soil5_water_ds_m": [
        "electrical conductivity", "ds/m", "2 soil:5 water"
    ],
    "ec_saturated_paste_us_cm": [
        "electrical conductivity", "µs/cm", "saturated paste"
    ],
    "ec_saturated_paste_ds_m": [
        "electrical conductivity", "ds/m", "saturated paste"
    ],
    "organic_carbon": ["organic carbon"],
    "total_nitrogen": ["total nitrogen"],
    "carbonate_equivalent": ["total carbonate equivalent"],
    "exchangeable_calcium": ["exchangeable calcium"],
    "exchangeable_magnesium": ["exchangeable magnesium"],
    "exchangeable_potassium": ["exchangeable potassium"],
    "exchangeable_sodium": ["exchangeable sodium"],
    "cec_effective": ["cec effective"],
    "cec_measured": ["cec measured"],
    "phosphorus_olsen": ["phosphorus", "olsen"],
    "chloride": ["chloride"],
    "sulphate": ["sulfate"],
}

FIELD_PATTERNS = {
    "field_texture": ["field texture"],
    "diagnostic_material": ["diagnostic material"],
    "horizon_code": ["horizon code"],
    "structure": ["structure"],
    "carbonate_reaction": ["carbonate reaction"],
    "colour_moist": ["colour moist"],
}

PROFILE_PATTERNS = {
    "parent_material": ["parent material"],
    "lithology": ["lithology"],
    "profile_drainage": ["drainage"],
    "soil_depth_class": ["depth class"],
    "slope": ["slope"],
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )

    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                low_memory=False
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Unable to read CSV: {path}")


def normalize(value):
    value = str(value).lower().strip()
    value = value.replace("\n", " ")
    value = value.replace("\r", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def find_key(df, key):
    for col in df.columns:
        if normalize(col) == key.lower():
            return col
    return None


def find_column(df, patterns):
    candidates = []

    for col in df.columns:
        column_name = normalize(col)

        if all(pattern in column_name for pattern in patterns):
            populated = (
                df[col].astype(str).str.strip().ne("").sum()
            )
            candidates.append((populated, col))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[0][1]


def parse_numeric(value):
    """
    Parse exact numeric values only.

    Values such as <0.5 and >60 remain unresolved.
    They are not converted to zero or treated as exact values.
    """

    if value is None:
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    if value.lower() in {"nan", "none", "na", "n/a"}:
        return np.nan

    value = value.replace(",", "")

    if any(symbol in value for symbol in ["<", ">", "≤", "≥"]):
        return np.nan

    if re.fullmatch(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        value
    ):
        return float(value)

    return np.nan


def make_numeric_feature(df, source_df, source_column, feature_name):
    if source_column is not None:
        df[feature_name] = (
            source_df[source_column].map(parse_numeric)
        )


def remove_duplicate_column_names(df):
    return df.loc[
        :,
        ~df.columns.duplicated()
    ].copy()


# =========================================================
# 1. LOAD SOURCE DATA
# =========================================================

print("\n" + "=" * 65)
print("BOREMAN - STEP 12: SOIL FEATURE ENGINEERING")
print("=" * 65)

print("\nLoading soil datasets...")

profiles = read_csv(PROFILE_FILE)
lab = read_csv(LAB_FILE)
field = read_csv(FIELD_FILE)

print(f"Profiles: {len(profiles):,}")
print(f"Laboratory horizons: {len(lab):,}")
print(f"Field horizons: {len(field):,}")


# =========================================================
# 2. IDENTIFY REQUIRED KEYS
# =========================================================

profile_prid = find_key(profiles, "prid")
lab_prid = find_key(lab, "prid")
field_prid = find_key(field, "prid")

lab_honu = find_key(lab, "honu")
field_honu = find_key(field, "honu")

required_keys = {
    "Profile PRID": profile_prid,
    "Lab PRID": lab_prid,
    "Field PRID": field_prid,
    "Lab HONU": lab_honu,
    "Field HONU": field_honu,
}

missing_keys = [
    name for name, column in required_keys.items()
    if column is None
]

if missing_keys:
    raise ValueError(
        f"Missing required columns: {missing_keys}"
    )


# =========================================================
# 3. STANDARDISE JOIN KEYS
# =========================================================

profiles["_PRID_KEY"] = (
    profiles[profile_prid].astype(str).str.strip()
)

lab["_PRID_KEY"] = (
    lab[lab_prid].astype(str).str.strip()
)

field["_PRID_KEY"] = (
    field[field_prid].astype(str).str.strip()
)

lab["_HONU_KEY"] = (
    lab[lab_honu].astype(str).str.strip()
)

field["_HONU_KEY"] = (
    field[field_honu].astype(str).str.strip()
)

for name, df in [
    ("Profiles", profiles),
    ("Lab", lab),
    ("Field", field)
]:
    if df["_PRID_KEY"].eq("").any():
        raise ValueError(f"Blank PRID found in {name} data.")

if lab["_HONU_KEY"].eq("").any():
    raise ValueError("Blank HONU found in laboratory data.")

if field["_HONU_KEY"].eq("").any():
    raise ValueError("Blank HONU found in field data.")


# =========================================================
# 4. VALIDATE KEYS
# =========================================================

if profiles["_PRID_KEY"].duplicated().any():
    raise ValueError("Duplicate soil profile PRIDs found.")

if lab.duplicated(["_PRID_KEY", "_HONU_KEY"]).any():
    raise ValueError("Duplicate laboratory horizon keys found.")

if field.duplicated(["_PRID_KEY", "_HONU_KEY"]).any():
    raise ValueError("Duplicate field horizon keys found.")

lab_keys = set(
    zip(lab["_PRID_KEY"], lab["_HONU_KEY"])
)

field_keys = set(
    zip(field["_PRID_KEY"], field["_HONU_KEY"])
)

if lab_keys != field_keys:
    raise ValueError(
        "Lab and field horizon keys do not match.\n"
        f"Lab only: {len(lab_keys - field_keys)}\n"
        f"Field only: {len(field_keys - lab_keys)}"
    )

print("\nHorizon pairing verified.")
print("Exact lab and field key sets match.")


# =========================================================
# 5. ENGINEER LABORATORY FEATURES
# =========================================================

print("\nIdentifying laboratory features...")

lab_features = lab.copy()

feature_sources = {}

for feature_name, patterns in FEATURE_PATTERNS.items():

    source_column = find_column(lab, patterns)

    if source_column is not None:

        make_numeric_feature(
            lab_features,
            lab,
            source_column,
            feature_name
        )

        feature_sources[feature_name] = source_column

numeric_features = list(feature_sources.keys())

print(
    f"Numeric soil features identified: "
    f"{len(numeric_features)}"
)


# =========================================================
# 6. ENGINEER DEPTH FEATURES
# =========================================================

upper_col = find_column(lab, ["upper depth"])
lower_col = find_column(lab, ["lower depth"])
thickness_col = find_column(lab, ["hor thickness"])

lab_features["upper_depth_cm_numeric"] = (
    lab[upper_col].map(parse_numeric)
    if upper_col else np.nan
)

lab_features["lower_depth_cm_numeric"] = (
    lab[lower_col].map(parse_numeric)
    if lower_col else np.nan
)

lab_features["horizon_thickness_cm_numeric"] = (
    lab[thickness_col].map(parse_numeric)
    if thickness_col else np.nan
)

lab_features["depth_midpoint_cm"] = (
    lab_features["upper_depth_cm_numeric"]
    + lab_features["lower_depth_cm_numeric"]
) / 2

lab_features["depth_interval_valid"] = (
    lab_features["upper_depth_cm_numeric"].notna()
    & lab_features["lower_depth_cm_numeric"].notna()
    & (
        lab_features["lower_depth_cm_numeric"]
        > lab_features["upper_depth_cm_numeric"]
    )
)

lab_features["depth_band"] = np.select(
    [
        lab_features["depth_interval_valid"]
        & (lab_features["depth_midpoint_cm"] <= 30),

        lab_features["depth_interval_valid"]
        & (lab_features["depth_midpoint_cm"] > 30)
        & (lab_features["depth_midpoint_cm"] <= 100),

        lab_features["depth_interval_valid"]
        & (lab_features["depth_midpoint_cm"] > 100)
    ],
    [
        "TOPSOIL_0_30CM",
        "SUBSOIL_30_100CM",
        "DEEP_SOIL_OVER_100CM"
    ],
    default="DEPTH_UNRESOLVED"
)


# =========================================================
# 7. ENGINEER FIELD FEATURES
# =========================================================

print("\nIdentifying field horizon features...")

field_features = field.copy()

field_sources = {}

for feature_name, patterns in FIELD_PATTERNS.items():

    source_column = find_column(field, patterns)

    if source_column is not None:

        field_features[feature_name] = (
            field[source_column].astype(str).str.strip()
        )

        field_sources[feature_name] = source_column


# =========================================================
# 8. JOIN LAB AND FIELD HORIZONS
# =========================================================

print("\nCombining laboratory and field horizons...")

horizons = lab_features.merge(
    field_features,
    on=["_PRID_KEY", "_HONU_KEY"],
    how="inner",
    suffixes=("_LAB", "_FIELD"),
    validate="one_to_one"
)

if len(horizons) != len(lab):
    raise ValueError(
        "Combined horizon count does not match laboratory records."
    )

horizons.to_csv(
    OUTPUT_DIR / "01_BOREMAN_ENGINEERED_HORIZON_FEATURES.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Combined horizons: {len(horizons):,}")


# =========================================================
# 9. PROFILE-LEVEL FEATURE SUMMARIES
# =========================================================

print("\nCreating profile-level features...")

summary_rows = []

for prid, group in lab_features.groupby(
    "_PRID_KEY",
    sort=False
):

    row = {
        "_PRID_KEY": prid,
        "horizon_count": len(group),
        "depth_resolved_horizon_count": int(
            group["depth_interval_valid"].sum()
        )
    }

    for feature in numeric_features:

        values = group[feature].dropna()

        row[f"{feature}_mean"] = (
            values.mean() if not values.empty else np.nan
        )

        row[f"{feature}_median"] = (
            values.median() if not values.empty else np.nan
        )

        row[f"{feature}_min"] = (
            values.min() if not values.empty else np.nan
        )

        row[f"{feature}_max"] = (
            values.max() if not values.empty else np.nan
        )

        row[f"{feature}_measurement_count"] = len(values)

        topsoil = group.loc[
            group["depth_band"] == "TOPSOIL_0_30CM",
            feature
        ].dropna()

        row[f"{feature}_topsoil_mean"] = (
            topsoil.mean() if not topsoil.empty else np.nan
        )

        row[f"{feature}_topsoil_count"] = len(topsoil)

        subsoil = group.loc[
            group["depth_band"] == "SUBSOIL_30_100CM",
            feature
        ].dropna()

        row[f"{feature}_subsoil_mean"] = (
            subsoil.mean() if not subsoil.empty else np.nan
        )

        row[f"{feature}_subsoil_count"] = len(subsoil)

    summary_rows.append(row)

profile_summary = pd.DataFrame(summary_rows)

if "_PRID_KEY" not in profile_summary.columns:
    raise ValueError("Profile summary key was not created.")

if profile_summary["_PRID_KEY"].duplicated().any():
    raise ValueError("Duplicate PRIDs in profile summary.")


# =========================================================
# 10. ADD PROFILE METADATA
# =========================================================

profile_metadata = profiles.copy()

# Remove original PRID column(s) so there is one unambiguous key.
original_prid_columns = [
    col for col in profile_metadata.columns
    if normalize(col) == "prid"
]

profile_metadata = profile_metadata.drop(
    columns=original_prid_columns
)

profile_metadata = remove_duplicate_column_names(
    profile_metadata
)

profile_categorical_sources = {}

for feature_name, patterns in PROFILE_PATTERNS.items():

    source_column = find_column(profiles, patterns)

    if source_column is not None:

        profile_metadata[feature_name] = (
            profiles[source_column].astype(str).str.strip()
        )

        profile_categorical_sources[feature_name] = source_column

profile_features = profile_metadata.merge(
    profile_summary,
    on="_PRID_KEY",
    how="left",
    validate="one_to_one"
)

# Keep the internal key for the spatial join.
# A separate copy will be used for the public output.
profile_output = profile_features.rename(
    columns={"_PRID_KEY": "PRID"}
)

profile_output = remove_duplicate_column_names(profile_output)

profile_output.to_csv(
    OUTPUT_DIR / "02_BOREMAN_ENGINEERED_PROFILE_FEATURES.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Profile records: {len(profile_output):,}")


# =========================================================
# 11. DEPTH-RESOLVED FEATURE DATASET
# =========================================================

depth_columns = [
    "_PRID_KEY",
    "_HONU_KEY",
    "upper_depth_cm_numeric",
    "lower_depth_cm_numeric",
    "horizon_thickness_cm_numeric",
    "depth_midpoint_cm",
    "depth_interval_valid",
    "depth_band"
] + numeric_features

depth_features = lab_features[depth_columns].copy()

depth_features = depth_features.rename(
    columns={
        "_PRID_KEY": "PRID",
        "_HONU_KEY": "HONU"
    }
)

depth_features.to_csv(
    OUTPUT_DIR / "03_BOREMAN_DEPTH_RESOLVED_FEATURES.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 12. GROUNDWATER-TO-SOIL LINK
# =========================================================

print("\nPreparing groundwater-to-soil feature link...")

link_status = "Not created: nearest matching file not found."

nearest_candidates = [
    MATCH_DIR / "BOREMAN_BOREHOLE_NEAREST_SOIL.csv",
    MATCH_DIR / "BOREMAN_BOREHOLE_SOIL_CANDIDATES_TOP5.csv"
]

nearest_file = next(
    (path for path in nearest_candidates if path.exists()),
    None
)

if nearest_file is not None:

    nearest = read_csv(nearest_file)

    prid_candidates = [
        col for col in nearest.columns
        if "prid" in normalize(col)
    ]

    soil_prid_col = None

    for col in prid_candidates:

        col_name = normalize(col)

        if "soil" in col_name or "nearest" in col_name:
            soil_prid_col = col
            break

    if soil_prid_col is None and len(prid_candidates) == 1:
        soil_prid_col = prid_candidates[0]

    if soil_prid_col is None:

        link_status = (
            f"Found {nearest_file.name}, but could not identify "
            "a unique soil PRID column. No link was created."
        )

    else:

        nearest["_SOIL_PRID_KEY"] = (
            nearest[soil_prid_col].astype(str).str.strip()
        )

        # Build a fresh soil lookup with exactly one join key.
        soil_link_features = profile_features.copy()

        soil_link_features = remove_duplicate_column_names(
            soil_link_features
        )

        soil_link_features = soil_link_features.rename(
            columns={"_PRID_KEY": "_SOIL_PRID_KEY"}
        )

        if "_SOIL_PRID_KEY" not in soil_link_features.columns:
            raise ValueError(
                "Soil PRID key missing from engineered profiles."
            )

        if soil_link_features["_SOIL_PRID_KEY"].duplicated().any():
            raise ValueError(
                "Duplicate soil PRID keys in soil lookup."
            )

        groundwater_soil = nearest.merge(
            soil_link_features,
            on="_SOIL_PRID_KEY",
            how="left",
            suffixes=("", "_SOIL"),
            validate="many_to_one"
        )

        groundwater_soil = remove_duplicate_column_names(
            groundwater_soil
        )

        link_path = (
            OUTPUT_DIR
            / "04_BOREMAN_GROUNDWATER_SOIL_FEATURE_LINK.csv"
        )

        groundwater_soil.to_csv(
            link_path,
            index=False,
            encoding="utf-8-sig"
        )

        matched = int(
            groundwater_soil["horizon_count"].notna().sum()
        )

        link_status = (
            f"Created from {nearest_file.name}. "
            f"Matched {matched:,} of "
            f"{len(groundwater_soil):,} rows."
        )

else:

    print("Nearest soil matching file not found.")


# =========================================================
# 13. FEATURE COVERAGE REPORT
# =========================================================

coverage_rows = []

for feature in numeric_features:

    values = lab_features[feature]

    coverage_rows.append({
        "feature": feature,
        "source_column": feature_sources[feature],
        "total_horizons": len(values),
        "numeric_count": int(values.notna().sum()),
        "missing_or_unparsed_count": int(values.isna().sum()),
        "numeric_coverage_percent": round(
            values.notna().mean() * 100,
            2
        ) if len(values) else 0
    })

coverage_df = pd.DataFrame(coverage_rows)

coverage_df.to_csv(
    OUTPUT_DIR / "05_ENGINEERED_FEATURE_COVERAGE.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 14. GENERATE PROCESSING REPORT
# =========================================================

report = [
    "BOREMAN - SOIL FEATURE ENGINEERING REPORT",
    "=" * 65,
    f"Input soil profiles: {len(profiles):,}",
    f"Input laboratory horizons: {len(lab):,}",
    f"Input field horizons: {len(field):,}",
    f"Combined matched horizons: {len(horizons):,}",
    f"Engineered profile records: {len(profile_output):,}",
    f"Numeric features identified: {len(numeric_features):,}",
    "",
    "DEPTH HANDLING",
    "-" * 65,
    "Only exact numeric depth values were parsed.",
    "Censored values such as >60 were not treated as exact depths.",
    "Topsoil: horizon midpoint <= 30 cm.",
    "Subsoil: horizon midpoint >30 and <=100 cm.",
    "Deep soil: horizon midpoint >100 cm.",
    "Unresolved depth intervals remain labelled DEPTH_UNRESOLVED.",
    "",
    "GROUNDWATER LINK",
    "-" * 65,
    link_status,
    "",
    "SCIENTIFIC LIMITATIONS",
    "-" * 65,
    "Soil pH and soil EC are not groundwater pH and groundwater EC.",
    "Profile means are unweighted descriptive summaries.",
    "Nearest soil profiles are spatial proxies, not borehole samples.",
    "No missing measurements were imputed.",
    "No groundwater prediction model was trained.",
    "Original input files were not modified.",
    "",
    "OUTPUT FILES",
    "-" * 65,
    "01_BOREMAN_ENGINEERED_HORIZON_FEATURES.csv",
    "02_BOREMAN_ENGINEERED_PROFILE_FEATURES.csv",
    "03_BOREMAN_DEPTH_RESOLVED_FEATURES.csv",
    "04_BOREMAN_GROUNDWATER_SOIL_FEATURE_LINK.csv (if created)",
    "05_ENGINEERED_FEATURE_COVERAGE.csv",
]

report_path = (
    OUTPUT_DIR / "BOREMAN_SOIL_FEATURE_ENGINEERING_REPORT.txt"
)

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# =========================================================
# 15. FINAL STATUS
# =========================================================

print("\n" + "=" * 65)
print("BOREMAN SOIL FEATURE ENGINEERING COMPLETED")
print("=" * 65)

print(f"Profiles: {len(profile_output):,}")
print(f"Combined horizons: {len(horizons):,}")
print(f"Numeric features: {len(numeric_features):,}")
print(f"Groundwater link: {link_status}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Report: {report_path}")
