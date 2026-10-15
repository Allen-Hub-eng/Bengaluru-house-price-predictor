# ============================================================
# BENGALURU HOUSE PRICE PREDICTION
# STAGE 5.1A: INVESTIGATE EXTREME PREDICTION ERRORS
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPARED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prepared_houses.csv"
)

ERRORS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "stage05_1_all_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ------------------------------------------------------------
# 2. LOAD THE DATASETS
# ------------------------------------------------------------

houses = pd.read_csv(PREPARED_DATA_PATH)

predictions = pd.read_csv(ERRORS_PATH)

print("\n" + "=" * 65)
print("EXTREME PREDICTION ERROR INVESTIGATION")
print("=" * 65)

print("\nPrepared dataset shape:", houses.shape)

print("Prediction dataset shape:", predictions.shape)


# ------------------------------------------------------------
# 3. INSPECT THE LARGEST ERRORS
# ------------------------------------------------------------

print("\nTOP 25 LARGEST ERRORS")
print("-" * 65)

columns_to_display = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony",
    "actual_price_lakhs",
    "predicted_price_lakhs",
    "absolute_error_lakhs",
    "absolute_percentage_error",
]

top_errors = predictions.sort_values(
    "absolute_error_lakhs",
    ascending=False
).head(25)

print(
    top_errors[columns_to_display].to_string(index=False)
)


# ------------------------------------------------------------
# 4. CHECK WHETHER EXTREME ERRORS INVOLVE LARGE PROPERTIES
# ------------------------------------------------------------

print("\nAREA DISTRIBUTION OF THE 25 LARGEST ERRORS")
print("-" * 65)

print(
    top_errors["total_sqft"].describe().round(2)
)

print("\nBEDROOM DISTRIBUTION")
print("-" * 65)

print(
    top_errors["bhk"].value_counts().sort_index()
)

print("\nBATHROOM DISTRIBUTION")
print("-" * 65)

print(
    top_errors["bath"].value_counts().sort_index()
)


# ------------------------------------------------------------
# 5. CHECK WHICH LOCATION CATEGORIES APPEAR MOST OFTEN
# ------------------------------------------------------------

print("\nLOCATION CATEGORIES IN THE LARGEST ERRORS")
print("-" * 65)

print(
    top_errors["location"].value_counts()
)


# ------------------------------------------------------------
# 6. COMPARE ERROR SIZE ACROSS LOCATION CATEGORIES
# ------------------------------------------------------------

location_analysis = (
    predictions
    .groupby("location", dropna=False)
    .agg(
        property_count=(
            "absolute_error_lakhs",
            "count"
        ),
        mean_absolute_error=(
            "absolute_error_lakhs",
            "mean"
        ),
        median_absolute_error=(
            "absolute_error_lakhs",
            "median"
        ),
        maximum_absolute_error=(
            "absolute_error_lakhs",
            "max"
        ),
        average_actual_price=(
            "actual_price_lakhs",
            "mean"
        )
    )
    .reset_index()
)

# Sort by mean absolute error.
# Categories with only a few records can have unstable estimates.

location_analysis = location_analysis.sort_values(
    "mean_absolute_error",
    ascending=False
)

location_report_path = (
    OUTPUT_DIR
    / "stage05_1_location_error_analysis.csv"
)

location_analysis.to_csv(
    location_report_path,
    index=False
)

print("\nLOCATION CATEGORIES WITH THE HIGHEST MAE")
print("-" * 65)

print(
    location_analysis.head(20).round(2).to_string(index=False)
)

print("\nSaved location analysis to:")
print(location_report_path)


# ------------------------------------------------------------
# 7. CHECK PRICE AND PRICE-PER-SQUARE-FOOT DISTRIBUTIONS
# ------------------------------------------------------------

print("\nPREPARED DATASET PRICE DISTRIBUTION")
print("-" * 65)

print(
    houses["price"].describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    ).round(2)
)

print("\nPRICE PER SQUARE FOOT DISTRIBUTION")
print("-" * 65)

if "price_per_sqft" in houses.columns:

    print(
        houses["price_per_sqft"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
        .round(2)
    )

else:

    print("price_per_sqft column not found.")


# ------------------------------------------------------------
# 8. CHECK WHETHER EXTREME ERRORS COME FROM THE 'OTHER' GROUP
# ------------------------------------------------------------

print("\nCOMPARISON: OTHER VS NAMED LOCATIONS")
print("-" * 65)

predictions["is_other_location"] = (
    predictions["location"].astype(str).str.lower() == "other"
)

other_comparison = (
    predictions
    .groupby("is_other_location")
    .agg(
        property_count=(
            "absolute_error_lakhs",
            "count"
        ),
        mean_absolute_error=(
            "absolute_error_lakhs",
            "mean"
        ),
        median_absolute_error=(
            "absolute_error_lakhs",
            "median"
        ),
        average_actual_price=(
            "actual_price_lakhs",
            "mean"
        )
    )
    .reset_index()
)

other_comparison["location_group"] = (
    other_comparison["is_other_location"]
    .map({
        True: "other",
        False: "named location"
    })
)

print(
    other_comparison[
        [
            "location_group",
            "property_count",
            "mean_absolute_error",
            "median_absolute_error",
            "average_actual_price"
        ]
    ].round(2).to_string(index=False)
)


# ------------------------------------------------------------
# 9. SAVE A COMBINED INVESTIGATION REPORT
# ------------------------------------------------------------

report_path = (
    OUTPUT_DIR
    / "stage05_1_extreme_error_investigation.txt"
)

report_lines = [
    "EXTREME PREDICTION ERROR INVESTIGATION",
    "=" * 60,
    "",
    f"Prepared dataset records: {len(houses)}",
    f"Test predictions analysed: {len(predictions)}",
    "",
    "PRICE DISTRIBUTION",
    "-" * 60,
    houses["price"].describe().to_string(),
    "",
    "TOP 25 ERROR RECORDS",
    "-" * 60,
    top_errors[columns_to_display].to_string(index=False),
    "",
    "LOCATION CATEGORY ERROR ANALYSIS",
    "-" * 60,
    location_analysis.head(20).round(2).to_string(index=False),
    "",
    "OTHER VS NAMED LOCATIONS",
    "-" * 60,
    other_comparison.to_string(index=False),
    "",
    "INTERPRETATION NOTE",
    "-" * 60,
    "Large prediction errors do not automatically mean a record",
    "is incorrect. Inspect source data and property plausibility",
    "before excluding any observation.",
    "",
    f"Location analysis CSV: {location_report_path}",
]

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nInvestigation completed.")

print("Report saved to:")
print(report_path)