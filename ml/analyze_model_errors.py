# ============================================================
# BENGALURU HOUSE PRICE PREDICTION
# STAGE 5.1: MODEL ERROR ANALYSIS
# ============================================================

# SECTION 1: IMPORT LIBRARIES

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SECTION 2: DEFINE PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "prepared_houses.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SECTION 3: LOAD THE DATASET
# ============================================================

print("\n" + "=" * 65)
print("BENGALURU HOUSE PRICE PREDICTION - ERROR ANALYSIS")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")

print("Dataset shape:", df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# SECTION 4: SELECT FEATURES AND TARGET
# ============================================================

FEATURES = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony"
]

TARGET = "price"

# X contains the input features.
X = df[FEATURES].copy()

# y contains the actual house prices in lakhs.
y = df[TARGET].copy()

print("\nInput features:")
print(FEATURES)

print("\nTarget variable:", TARGET)


# ============================================================
# SECTION 5: REPRODUCE THE ORIGINAL TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# SECTION 6: BUILD THE PREPROCESSING PIPELINE
# ============================================================

numeric_features = [
    "total_sqft",
    "bhk",
    "bath",
    "balcony"
]

categorical_features = [
    "location"
]

# Numerical preprocessing:
# 1. Fill missing values using the median.
# 2. Standardize numerical values.

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# Categorical preprocessing:
# 1. Fill missing categories using the most frequent value.
# 2. Convert location categories into numerical columns.

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# Combine the numerical and categorical transformations.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# SECTION 7: CREATE THE BASELINE RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ]
)


# ============================================================
# SECTION 8: TRAIN THE MODEL
# ============================================================

print("\nTraining the baseline Random Forest model...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# SECTION 9: GENERATE PREDICTIONS
# ============================================================

predictions = model.predict(X_test)

# Calculate the difference between actual and predicted prices.

errors = y_test.to_numpy() - predictions

# Absolute error ignores whether the prediction was too high
# or too low.

absolute_errors = np.abs(errors)

# Percentage error is meaningful only when actual prices
# are greater than zero.

actual_prices = y_test.to_numpy()

absolute_percentage_errors = (
    absolute_errors / actual_prices
) * 100


# ============================================================
# SECTION 10: EVALUATE THE MODEL
# ============================================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n" + "=" * 65)
print("BASELINE MODEL PERFORMANCE")
print("=" * 65)

print(f"MAE:  {mae:.2f} lakhs")
print(f"RMSE: {rmse:.2f} lakhs")
print(f"R2:   {r2:.4f}")


# ============================================================
# SECTION 11: CREATE A PREDICTION-ERROR DATAFRAME
# ============================================================

results = X_test.copy()

results["actual_price_lakhs"] = actual_prices

results["predicted_price_lakhs"] = predictions

# Positive error: actual price is higher than predicted price.
# Negative error: predicted price is higher than actual price.

results["error_lakhs"] = errors

results["absolute_error_lakhs"] = absolute_errors

results["absolute_percentage_error"] = absolute_percentage_errors

# Restore the original row index from the prepared dataset.
# This helps us identify the corresponding source record.

results["source_row_index"] = X_test.index

# Sort from largest absolute error to smallest.

results = results.sort_values(
    by="absolute_error_lakhs",
    ascending=False
)

# Save all test predictions and their errors.

all_predictions_path = (
    OUTPUT_DIR / "stage05_1_all_predictions.csv"
)

results.to_csv(
    all_predictions_path,
    index=False
)

print("\nSaved all test predictions to:")
print(all_predictions_path)


# ============================================================
# SECTION 12: IDENTIFY THE 25 LARGEST PREDICTION ERRORS
# ============================================================

worst_predictions = results.head(25)

worst_predictions_path = (
    OUTPUT_DIR / "stage05_1_largest_errors.csv"
)

worst_predictions.to_csv(
    worst_predictions_path,
    index=False
)

print("\n" + "=" * 65)
print("25 LARGEST PREDICTION ERRORS")
print("=" * 65)

print(
    worst_predictions[
        [
            "location",
            "total_sqft",
            "bhk",
            "actual_price_lakhs",
            "predicted_price_lakhs",
            "absolute_error_lakhs"
        ]
    ].to_string(index=False)
)

print("\nSaved largest errors to:")
print(worst_predictions_path)


# ============================================================
# SECTION 13: ANALYZE ERRORS BY ACTUAL PRICE RANGE
# ============================================================

# Group properties by their actual prices.
# This helps us understand how the model performs at
# different price levels.

price_bins = [
    0,
    50,
    100,
    200,
    500,
    np.inf
]

price_labels = [
    "0-50 lakhs",
    "50-100 lakhs",
    "100-200 lakhs",
    "200-500 lakhs",
    "500+ lakhs"
]

price_groups = pd.cut(
    actual_prices,
    bins=price_bins,
    labels=price_labels,
    right=False
)

group_analysis = pd.DataFrame(
    {
        "actual_price_lakhs": actual_prices,
        "predicted_price_lakhs": predictions,
        "absolute_error_lakhs": absolute_errors,
        "price_group": price_groups
    }
)

price_range_report = (
    group_analysis
    .groupby(
        "price_group",
        observed=False
    )
    .agg(
        number_of_properties=(
            "actual_price_lakhs",
            "count"
        ),
        average_actual_price=(
            "actual_price_lakhs",
            "mean"
        ),
        average_predicted_price=(
            "predicted_price_lakhs",
            "mean"
        ),
        mean_absolute_error=(
            "absolute_error_lakhs",
            "mean"
        )
    )
    .reset_index()
)

price_range_path = (
    OUTPUT_DIR / "stage05_1_price_range_analysis.csv"
)

price_range_report.to_csv(
    price_range_path,
    index=False
)

print("\n" + "=" * 65)
print("ERROR ANALYSIS BY PRICE RANGE")
print("=" * 65)

print(price_range_report.to_string(index=False))

print("\nSaved price-range analysis to:")
print(price_range_path)


# ============================================================
# SECTION 14: CREATE ACTUAL VS PREDICTED PRICE PLOT
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    actual_prices,
    predictions,
    alpha=0.5
)

# Draw a diagonal reference line representing perfect predictions.

minimum_price = min(
    actual_prices.min(),
    predictions.min()
)

maximum_price = max(
    actual_prices.max(),
    predictions.max()
)

plt.plot(
    [minimum_price, maximum_price],
    [minimum_price, maximum_price],
    linestyle="--"
)

plt.xlabel("Actual Price (lakhs)")

plt.ylabel("Predicted Price (lakhs)")

plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

actual_predicted_path = (
    OUTPUT_DIR / "stage05_1_actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300
)

plt.close()

print("\nSaved actual-versus-predicted plot to:")
print(actual_predicted_path)


# ============================================================
# SECTION 15: CREATE ABSOLUTE ERROR DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    absolute_errors,
    bins=50,
    edgecolor="black"
)

plt.xlabel("Absolute Prediction Error (lakhs)")

plt.ylabel("Number of Properties")

plt.title("Distribution of Absolute Prediction Errors")

plt.tight_layout()

error_distribution_path = (
    OUTPUT_DIR / "stage05_1_error_distribution.png"
)

plt.savefig(
    error_distribution_path,
    dpi=300
)

plt.close()

print("\nSaved error distribution plot to:")
print(error_distribution_path)


# ============================================================
# SECTION 16: SAVE A TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR / "stage05_1_error_analysis_report.txt"
)

report_lines = [
    "BENGALURU HOUSE PRICE PREDICTION",
    "STAGE 5.1: MODEL ERROR ANALYSIS",
    "=" * 55,
    "",
    f"Dataset records: {len(df)}",
    f"Training records: {len(X_train)}",
    f"Testing records: {len(X_test)}",
    "",
    "BASELINE MODEL PERFORMANCE",
    "-" * 55,
    f"MAE: {mae:.4f} lakhs",
    f"RMSE: {rmse:.4f} lakhs",
    f"R2: {r2:.4f}",
    "",
    "ERROR SUMMARY",
    "-" * 55,
    f"Mean absolute error: {absolute_errors.mean():.4f} lakhs",
    f"Median absolute error: {np.median(absolute_errors):.4f} lakhs",
    f"90th percentile absolute error: {np.percentile(absolute_errors, 90):.4f} lakhs",
    f"95th percentile absolute error: {np.percentile(absolute_errors, 95):.4f} lakhs",
    f"Maximum absolute error: {absolute_errors.max():.4f} lakhs",
    "",
    "IMPORTANT NOTE",
    "-" * 55,
    "These results describe performance on the held-out test split.",
    "Large errors require investigation; they do not automatically",
    "mean the corresponding records should be removed.",
    "",
    "FILES GENERATED",
    "-" * 55,
    str(all_predictions_path),
    str(worst_predictions_path),
    str(price_range_path),
    str(actual_predicted_path),
    str(error_distribution_path)
]

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("ERROR ANALYSIS COMPLETED")
print("=" * 65)

print("\nReport saved to:")
print(report_path)

print("\nPlease inspect the generated CSV files and plots.")