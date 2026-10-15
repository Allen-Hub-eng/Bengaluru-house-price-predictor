
from pathlib import Path

import pandas as pd
import numpy as np

# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_houses.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"
REPORT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. LOAD THE CLEANED DATA
# ==========================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run ml/clean_data.py first."
    )

df = pd.read_csv(INPUT_PATH)

original_rows = len(df)

print("=" * 60)
print("STAGE 3: FEATURE ENGINEERING AND PREPROCESSING")
print("=" * 60)

print("Input shape:", df.shape)

# ==========================================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================================

required = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "price",
    "balcony"
]

missing_columns = [
    column for column in required
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ==========================================================
# 4. NORMALIZE LOCATION TEXT
# ==========================================================

df["location"] = (
    df["location"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df = df.dropna(subset=["location"]).copy()

# Convert empty location names to missing and remove them.
df["location"] = df["location"].replace("", pd.NA)
df = df.dropna(subset=["location"]).copy()

# ==========================================================
# 5. CALCULATE AREA PER BEDROOM
# ==========================================================

df["sqft_per_bhk"] = (
    df["total_sqft"] / df["bhk"]
)

# This is a diagnostic feature used to check plausibility.
# It will not be used as a model input in our first version.

# ==========================================================
# 6. FLAG SUSPICIOUS RECORDS
# ==========================================================

# These are conservative starting rules, not universal laws.
# Unusual properties should be reviewed before being excluded.

df["flag_small_area"] = df["total_sqft"] < 300
df["flag_large_area"] = df["total_sqft"] > 20000

df["flag_many_bedrooms"] = df["bhk"] > 10
df["flag_many_bathrooms"] = df["bath"] > 10

df["flag_low_area_per_bedroom"] = (
    df["sqft_per_bhk"] < 200
)

df["flag_high_price_per_sqft"] = (
    df["price_per_sqft"] > 100000
)

df["flag_low_price_per_sqft"] = (
    df["price_per_sqft"] < 500
)

flag_columns = [
    "flag_small_area",
    "flag_large_area",
    "flag_many_bedrooms",
    "flag_many_bathrooms",
    "flag_low_area_per_bedroom",
    "flag_high_price_per_sqft",
    "flag_low_price_per_sqft"
]

# A record is flagged if at least one rule is triggered.
df["flagged_for_review"] = df[flag_columns].any(axis=1)

flagged_rows = df[df["flagged_for_review"]].copy()

# Save the flagged records so that they can be inspected.
flagged_path = (
    OUTPUT_FOLDER / "outlier_review.csv"
)

flagged_rows.to_csv(flagged_path, index=False)

print(
    "\nRecords flagged for outlier review:",
    len(flagged_rows)
)

# ==========================================================
# 7. APPLY DOCUMENTED PLAUSIBILITY FILTERS
# ==========================================================

# Exclude records that fall outside our initial
# residential-property modelling scope.
#
# Price per square foot is NOT used as a hard filter here.
# It can be extreme because of unusual properties or
# because area values may be recorded inconsistently.

keep = (
    (df["total_sqft"] >= 300)
    & (df["total_sqft"] <= 20000)
    & (df["bhk"] >= 1)
    & (df["bhk"] <= 10)
    & (df["bath"] >= 1)
    & (df["bath"] <= 10)
    & (df["sqft_per_bhk"] >= 200)
    & (df["price"] > 0)
)

prepared = df.loc[keep].copy()

rows_removed_by_filters = len(df) - len(prepared)

print(
    "Rows removed by plausibility filters:",
    rows_removed_by_filters
)

# ==========================================================
# 8. GROUP RARE LOCATIONS
# ==========================================================

# A rare location has fewer than 10 records in the
# current prepared dataset.
#
# This reduces the number of separate location categories.
# In the final model pipeline, the category handling and
# encoding must be learned from the training split only.

location_counts = prepared["location"].value_counts()

rare_locations = location_counts[
    location_counts < 10
].index

prepared["location"] = prepared["location"].where(
    ~prepared["location"].isin(rare_locations),
    "other"
)

print(
    "\nNumber of location categories after grouping:",
    prepared["location"].nunique()
)

print(
    "Rows assigned to 'other':",
    int((prepared["location"] == "other").sum())
)

# ==========================================================
# 9. SELECT MODELING COLUMNS
# ==========================================================

# The target is price.
# Do not use price_per_sqft as an input because it is
# calculated directly from price.

model_columns = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony",
    "price"
]

prepared = prepared[model_columns].copy()

# Ensure numerical columns have numerical data types.
for column in [
    "total_sqft",
    "bhk",
    "bath",
    "balcony",
    "price"
]:
    prepared[column] = pd.to_numeric(
        prepared[column],
        errors="coerce"
    )

# Essential columns should never be missing at this point.
prepared = prepared.dropna(
    subset=[
        "location",
        "total_sqft",
        "bhk",
        "bath",
        "price"
    ]
).copy()

# Keep missing balcony values for later imputation.
# We will learn the imputation value from training data
# only, using a scikit-learn preprocessing pipeline.

# ==========================================================
# 10. SAVE THE PREPARED DATASET
# ==========================================================

prepared_path = (
    OUTPUT_FOLDER / "prepared_houses.csv"
)

prepared.to_csv(prepared_path, index=False)

print("\nPrepared dataset shape:", prepared.shape)
print("\nRemaining missing values:")
print(prepared.isnull().sum())

print("\nPrepared data preview:")
print(prepared.head())

# ==========================================================
# 11. SAVE THE STAGE 3 REPORT
# ==========================================================

report_path = (
    REPORT_FOLDER / "stage03_feature_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("STAGE 3: FEATURE ENGINEERING REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Input rows: {original_rows}\n")
    file.write(f"Rows after plausibility filters: {len(prepared)}\n")
    file.write(
        f"Rows removed by plausibility filters: "
        f"{rows_removed_by_filters}\n\n"
    )

    file.write(
        f"Records flagged for review: {len(flagged_rows)}\n"
    )

    file.write(
        f"Location categories after grouping: "
        f"{prepared['location'].nunique()}\n\n"
    )

    file.write("Model input features:\n")
    file.write(
        "location, total_sqft, bhk, bath, balcony\n"
    )

    file.write("\nTarget:\nprice\n")

    file.write("\nRemaining missing values:\n")
    file.write(prepared.isnull().sum().to_string())

    file.write("\n\nNumerical summary:\n")
    file.write(
        prepared[
            [
                "total_sqft",
                "bhk",
                "bath",
                "balcony",
                "price"
            ]
        ].describe().to_string()
    )

print("\nPrepared dataset saved to:")
print(prepared_path)

print("\nOutlier review file saved to:")
print(flagged_path)

print("\nStage 3 report saved to:")
print(report_path)