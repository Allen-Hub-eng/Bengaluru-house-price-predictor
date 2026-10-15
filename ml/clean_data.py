# Cleaning the dataset
from pathlib import Path

import pandas as pd
import numpy as np


# ==========================================================
# 1. DEFINE PROJECT PATHS
# ==========================================================

from pathlib import Path

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

print("Project path:", PROJECT_ROOT)

RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD THE DATASET
# ==========================================================

DATA_PATH = RAW_FOLDER / "Bengaluru_House_Data.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

original_rows = len(df)

print("=" * 60)
print("BENGALURU HOUSE PRICE DATA CLEANING")
print("=" * 60)

print("Original dataset shape:", df.shape)


# ==========================================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "location",
    "size",
    "total_sqft",
    "bath",
    "price"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )


# ==========================================================
# 4. REMOVE EXACT DUPLICATE ROWS
# ==========================================================

duplicates_before = int(df.duplicated().sum())

df = df.drop_duplicates().copy()

duplicates_removed = duplicates_before

print("\nExact duplicate rows removed:", duplicates_removed)


# ==========================================================
# 5. CLEAN TEXT COLUMNS
# ==========================================================

text_columns = [
    "area_type",
    "availability",
    "location",
    "size",
    "society",
    "total_sqft"
]

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].astype("string").str.strip()

# Convert empty strings into missing values.
for column in text_columns:
    if column in df.columns:
        df[column] = df[column].replace("", pd.NA)

# Remove rows without a usable location.
df = df.dropna(subset=["location"]).copy()

# Normalize location names.
df["location"] = (
    df["location"]
    .str.lower()
    .str.strip()
)


# ==========================================================
# 6. EXTRACT BEDROOM COUNT FROM THE SIZE COLUMN
# ==========================================================

# Examples:
# "2 BHK" -> 2
# "3 Bedroom" -> 3
# "4 Bedroom" -> 4

df["bhk"] = pd.to_numeric(
    df["size"]
    .str.extract(r"(\d+)", expand=False),
    errors="coerce"
)

df["bhk"] = df["bhk"].astype("Int64")

print("\nBedroom count created successfully.")


# ==========================================================
# 7. CONVERT TOTAL SQUARE FEET INTO NUMBERS
# ==========================================================

def convert_area(value):
    """
    Convert property area text into a numeric value.

    Examples:
        '1200'       -> 1200
        '1000-1200'  -> 1100
        '1,200'      -> 1200

    Unsupported formats become NaN and are reviewed
    separately instead of being guessed.
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace(",", "")

    # Handle area ranges such as 1000-1200.
    if "-" in value:
        parts = value.split("-")

        if len(parts) == 2:
            try:
                lower = float(parts[0].strip())
                upper = float(parts[1].strip())

                if lower > 0 and upper >= lower:
                    return (lower + upper) / 2

            except ValueError:
                return np.nan

    # Handle a simple numeric area.
    try:
        return float(value)

    except ValueError:
        return np.nan


df["total_sqft"] = df["total_sqft"].apply(convert_area)

print("Area values converted to numbers.")


# ==========================================================
# 8. CONVERT NUMERICAL COLUMNS
# ==========================================================

for column in ["bath", "balcony", "price"]:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ==========================================================
# 9. REMOVE RECORDS MISSING ESSENTIAL INFORMATION
# ==========================================================

rows_before_required_drop = len(df)

df = df.dropna(
    subset=[
        "location",
        "bhk",
        "total_sqft",
        "bath",
        "price"
    ]
).copy()

print(
    "\nRows removed for missing essential fields:",
    rows_before_required_drop - len(df)
)


# ==========================================================
# 10. REMOVE INVALID NON-POSITIVE VALUES
# ==========================================================

rows_before_validity_filter = len(df)

df = df[
    (df["bhk"] > 0)
    & (df["total_sqft"] > 0)
    & (df["bath"] > 0)
    & (df["price"] > 0)
].copy()

print(
    "Rows removed for invalid non-positive values:",
    rows_before_validity_filter - len(df)
)


# ==========================================================
# 11. HANDLE BALCONY VALUES
# ==========================================================

# Missing balcony values will be imputed later by the
# machine learning preprocessing pipeline.
# This avoids unnecessarily discarding those records.

if "balcony" in df.columns:
    df["balcony"] = df["balcony"].where(
        df["balcony"] >= 0,
        np.nan
    )


# ==========================================================
# 12. CREATE PRICE PER SQUARE FOOT
# ==========================================================

# The dataset's price is expected to be in lakhs.
# Multiplying by 100,000 converts lakhs to rupees.

df["price_per_sqft"] = (
    df["price"] * 100000 / df["total_sqft"]
)


# ==========================================================
# 13. CREATE A CLEANING REPORT
# ==========================================================

final_rows = len(df)
rows_removed = original_rows - final_rows

print("\nFinal dataset shape:", df.shape)
print("Original rows:", original_rows)
print("Final rows:", final_rows)
print("Total rows removed:", rows_removed)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nNumerical summary:")
print(
    df[
        [
            "total_sqft",
            "bhk",
            "bath",
            "price",
            "price_per_sqft"
        ]
    ].describe()
)


# ==========================================================
# 14. SAVE THE CLEANED DATASET
# ==========================================================

CLEANED_PATH = (
    PROCESSED_FOLDER / "cleaned_houses.csv"
)

df.to_csv(CLEANED_PATH, index=False)

print("\nCleaned dataset saved to:")
print(CLEANED_PATH)


# ==========================================================
# 15. SAVE A REPORT FOR STAGE 2
# ==========================================================

report_path = OUTPUT_FOLDER / "stage02_cleaning_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("BENGALURU HOUSE PRICE CLEANING REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Original rows: {original_rows}\n")
    file.write(f"Final rows: {final_rows}\n")
    file.write(f"Rows removed: {rows_removed}\n")
    file.write(f"Duplicates removed: {duplicates_removed}\n")

    file.write("\nFinal columns:\n")
    file.write(str(df.columns.tolist()))

    file.write("\n\nMissing values after cleaning:\n")
    file.write(df.isnull().sum().to_string())

    file.write("\n\nNumerical summary:\n")
    file.write(
        df[
            [
                "total_sqft",
                "bhk",
                "bath",
                "price",
                "price_per_sqft"
            ]
        ].describe().to_string()
    )

print("\nCleaning report saved to:")
print(report_path)