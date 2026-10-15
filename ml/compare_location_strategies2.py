# ============================================================
# BENGALURU HOUSE PRICE PREDICTION
# STAGE 5.3: LOCATION STRATEGY EXPERIMENTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_houses.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD THE ORIGINAL CLEANED DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "cleaned_houses.csv not found. "
        "Run ml/clean_data.py first."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STAGE 5.3: LOCATION STRATEGY EXPERIMENTS")
print("=" * 70)

print("\nOriginal cleaned dataset shape:", df.shape)


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony",
    "price"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 4. NORMALIZE AND VALIDATE FEATURES
# ============================================================

# Normalize location names.

df["location"] = (
    df["location"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["location"] = df["location"].replace(
    "",
    pd.NA
)

# Convert numerical columns.

numeric_columns = [
    "total_sqft",
    "bhk",
    "bath",
    "balcony",
    "price"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Keep records with usable essential features.

df = df.dropna(
    subset=[
        "location",
        "total_sqft",
        "bhk",
        "bath",
        "price"
    ]
).copy()

# Apply the same residential-property plausibility rules
# used in Stage 3.

df["sqft_per_bhk"] = (
    df["total_sqft"] / df["bhk"]
)

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

df = df.loc[keep].copy()

print("Records after cleaning and plausibility filters:", len(df))


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

FEATURES = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony"
]

TARGET = "price"

X = df[FEATURES].copy()
y = df[TARGET].copy()

NUMERICAL_FEATURES = [
    "total_sqft",
    "bhk",
    "bath",
    "balcony"
]


# ============================================================
# 6. CREATE ONE CONSISTENT TRAIN/TEST SPLIT
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
# 7. CREATE A PREPROCESSING FUNCTION
# ============================================================

def create_preprocessor():

    numerical_pipeline = Pipeline(
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

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                ["location"]
            )
        ]
    )

    return preprocessor


# ============================================================
# 8. DEFINE LOCATION FREQUENCY THRESHOLDS
# ============================================================

THRESHOLDS = [1, 5, 10, 20]

all_results = []
all_price_range_results = []
all_location_diagnostics = []

prediction_results = {}


# ============================================================
# 9. TRAIN AND EVALUATE EACH LOCATION STRATEGY
# ============================================================

for threshold in THRESHOLDS:

    print("\n" + "=" * 70)

    print(
        f"Testing minimum location frequency: {threshold}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 9A. Learn location frequencies from training data only.
    # --------------------------------------------------------

    training_location_counts = (
        X_train["location"].value_counts()
    )

    frequent_locations = set(
        training_location_counts[
            training_location_counts >= threshold
        ].index
    )

    # --------------------------------------------------------
    # 9B. Group rare locations using the training vocabulary.
    # --------------------------------------------------------

    X_train_model = X_train.copy()
    X_test_model = X_test.copy()

    X_train_model["location"] = (
        X_train_model["location"].where(
            X_train_model["location"].isin(
                frequent_locations
            ),
            "other"
        )
    )

    X_test_model["location"] = (
        X_test_model["location"].where(
            X_test_model["location"].isin(
                frequent_locations
            ),
            "other"
        )
    )

    # Count how many test records have locations unseen
    # in the original training data.

    training_locations = set(
        X_train["location"].unique()
    )

    unseen_test_locations = (
        ~X_test["location"].isin(training_locations)
    )

    unseen_test_count = int(
        unseen_test_locations.sum()
    )

    grouped_test_count = int(
        (
            X_test_model["location"] == "other"
        ).sum()
    )

    # --------------------------------------------------------
    # 9C. Create the model.
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=20,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # --------------------------------------------------------
    # 9D. Train the model.
    # --------------------------------------------------------

    model.fit(
        X_train_model,
        y_train
    )

    # --------------------------------------------------------
    # 9E. Generate test predictions.
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_model
    )

    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------------
    # 9F. Calculate evaluation metrics.
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    median_absolute_error = np.median(
        np.abs(
            y_test.to_numpy() - predictions
        )
    )

    # --------------------------------------------------------
    # 9G. Save results for this strategy.
    # --------------------------------------------------------

    model_name = f"Location_Threshold_{threshold}"

    prediction_results[model_name] = predictions

    all_results.append(
        {
            "Location_Threshold": threshold,
            "Location_Categories_Kept": len(
                frequent_locations
            ),
            "Training_Records": len(X_train),
            "Testing_Records": len(X_test),
            "Test_Locations_Unseen_In_Training": (
                unseen_test_count
            ),
            "Test_Records_Grouped_As_Other": (
                grouped_test_count
            ),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Median_Absolute_Error": (
                median_absolute_error
            )
        }
    )

    print(
        "Location categories retained:",
        len(frequent_locations)
    )

    print(
        "Unseen test-location records:",
        unseen_test_count
    )

    print(
        "Test records grouped as other:",
        grouped_test_count
    )

    print(f"MAE: {mae:.4f} lakhs")
    print(f"RMSE: {rmse:.4f} lakhs")
    print(f"R2: {r2:.4f}")
    print(
        "Median absolute error:",
        f"{median_absolute_error:.4f} lakhs"
    )

    # --------------------------------------------------------
    # 9H. Analyze performance by actual price range.
    # --------------------------------------------------------

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
        y_test,
        bins=price_bins,
        labels=price_labels,
        right=False
    )

    error_table = pd.DataFrame(
        {
            "price_group": price_groups.to_numpy(),
            "absolute_error": np.abs(
                y_test.to_numpy() - predictions
            )
        }
    )

    grouped_errors = (
        error_table.groupby(
            "price_group",
            observed=False
        )
        .agg(
            property_count=(
                "absolute_error",
                "count"
            ),
            MAE=(
                "absolute_error",
                "mean"
            )
        )
        .reset_index()
    )

    grouped_errors["Location_Threshold"] = threshold

    all_price_range_results.append(
        grouped_errors
    )

    # --------------------------------------------------------
    # 9I. Save a diagnostic of location frequencies.
    # --------------------------------------------------------

    location_diagnostic = pd.DataFrame(
        {
            "location": training_location_counts.index,
            "training_count": training_location_counts.values
        }
    )

    location_diagnostic["threshold"] = threshold

    location_diagnostic["retained"] = (
        location_diagnostic["training_count"]
        >= threshold
    )

    all_location_diagnostics.append(
        location_diagnostic
    )


# ============================================================
# 10. SAVE OVERALL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
).sort_values(
    by="MAE",
    ascending=True
)

results_path = (
    OUTPUT_DIR
    / "stage05_3_location_strategy_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 70)
print("LOCATION STRATEGY COMPARISON")
print("=" * 70)

print(
    results_df.round(4).to_string(index=False)
)

print("\nSaved:", results_path)


# ============================================================
# 11. SAVE PRICE-RANGE COMPARISON
# ============================================================

price_range_df = pd.concat(
    all_price_range_results,
    ignore_index=True
)

price_range_path = (
    OUTPUT_DIR
    / "stage05_3_location_price_range_comparison.csv"
)

price_range_df.to_csv(
    price_range_path,
    index=False
)

print("\nPRICE-RANGE COMPARISON")

print(
    price_range_df.round(2).to_string(index=False)
)

print("\nSaved:", price_range_path)


# ============================================================
# 12. SAVE LOCATION FREQUENCY DIAGNOSTICS
# ============================================================

location_diagnostics_df = pd.concat(
    all_location_diagnostics,
    ignore_index=True
)

diagnostics_path = (
    OUTPUT_DIR
    / "stage05_3_location_frequency_diagnostics.csv"
)

location_diagnostics_df.to_csv(
    diagnostics_path,
    index=False
)

print("\nSaved:", diagnostics_path)


# ============================================================
# 13. CREATE MAE COMPARISON GRAPH
# ============================================================

plot_results = results_df.sort_values(
    by="Location_Threshold"
)

plt.figure(figsize=(9, 6))

plt.plot(
    plot_results["Location_Threshold"],
    plot_results["MAE"],
    marker="o"
)

plt.xlabel("Minimum Training Records Per Location")
plt.ylabel("MAE (Lakhs)")

plt.title(
    "Effect of Location Frequency Threshold on MAE"
)

plt.xticks(THRESHOLDS)

plt.tight_layout()

mae_graph_path = (
    OUTPUT_DIR
    / "stage05_3_location_mae_comparison.png"
)

plt.savefig(
    mae_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. CREATE R2 COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    plot_results["Location_Threshold"],
    plot_results["R2"],
    marker="o"
)

plt.xlabel("Minimum Training Records Per Location")
plt.ylabel("R2")

plt.title(
    "Effect of Location Frequency Threshold on R2"
)

plt.xticks(THRESHOLDS)

plt.tight_layout()

r2_graph_path = (
    OUTPUT_DIR
    / "stage05_3_location_r2_comparison.png"
)

plt.savefig(
    r2_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. SAVE THE FINAL REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "stage05_3_location_strategy_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "BENGALURU HOUSE PRICE PREDICTION\n"
    )

    report.write(
        "STAGE 5.3: LOCATION STRATEGY REPORT\n"
    )

    report.write("=" * 65 + "\n\n")

    report.write(
        f"Training records: {len(X_train)}\n"
    )

    report.write(
        f"Testing records: {len(X_test)}\n\n"
    )

    report.write(
        "LOCATION STRATEGY COMPARISON\n\n"
    )

    report.write(
        results_df.round(4).to_string(index=False)
    )

    report.write(
        "\n\nPRICE-RANGE COMPARISON\n\n"
    )

    report.write(
        price_range_df.round(2).to_string(index=False)
    )

    report.write(
        "\n\nInterpretation notes:\n"
    )

    report.write(
        "- Lower MAE and RMSE indicate smaller prediction errors.\n"
    )

    report.write(
        "- Higher R2 generally indicates better predictive fit.\n"
    )

    report.write(
        "- Location thresholds are learned from training data only.\n"
    )

    report.write(
        "- Results are specific to this test split.\n"
    )

    report.write(
        "- The best threshold should be evaluated using several metrics.\n"
    )

print("\n" + "=" * 70)
print("STAGE 5.3 COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(results_path)
print(price_range_path)
print(diagnostics_path)
print(mae_graph_path)
print(r2_graph_path)
print(report_path)
