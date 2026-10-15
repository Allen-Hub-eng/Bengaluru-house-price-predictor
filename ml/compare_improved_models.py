# ============================================================
# BENGALURU HOUSE PRICE PREDICTION
# STAGE 5.2: MODEL IMPROVEMENT EXPERIMENTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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
    / "prepared_houses.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VALIDATE THE DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Prepared dataset not found. "
        "Run ml/prepare_features.py first."
    )

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony"
]

TARGET = "price"

X = df[FEATURES].copy()

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

# Remove invalid target values.

valid_rows = (
    y.notna()
    & np.isfinite(y)
    & (y > 0)
)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

NUMERICAL_FEATURES = [
    "total_sqft",
    "bhk",
    "bath",
    "balcony"
]

# Ensure numerical columns contain numeric values.

for column in NUMERICAL_FEATURES:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

# Standardize location text.

X["location"] = (
    X["location"]
    .astype("string")
    .str.strip()
    .str.lower()
)

X["location"] = X["location"].replace("", pd.NA)

# Remove records without a usable location.

valid_locations = X["location"].notna()

X = X.loc[valid_locations].copy()
y = y.loc[valid_locations].copy()

print("=" * 65)
print("STAGE 5.2: MODEL IMPROVEMENT EXPERIMENTS")
print("=" * 65)

print("\nDataset records:", len(X))


# ============================================================
# 3. CREATE A CONSISTENT TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 4. DEFINE A REUSABLE PREPROCESSING FUNCTION
# ============================================================

def create_preprocessor():
    """
    Create a preprocessing pipeline.

    Numerical features:
        - Fill missing values using the median.
        - Standardize numerical values.

    Location:
        - Fill missing values using the most frequent category.
        - Convert categories into numerical columns.
    """

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
# 5. DEFINE EXPERIMENT A: ORIGINAL RANDOM FOREST
# ============================================================

def create_baseline_model():
    """
    Reproduce the original Random Forest configuration.
    """

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

    return model


# ============================================================
# 6. DEFINE EXPERIMENT B: LESS AGGRESSIVE LOCATION GROUPING
# ============================================================

def create_location_experiment_model():
    """
    Test a model that receives the prepared location categories.

    This experiment uses the same prepared dataset as the baseline.
    It does not change the existing rare-location grouping.

    We will use this as a control before implementing an alternative
    location strategy in a separate experiment.
    """

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=25,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    return model


# ============================================================
# 7. DEFINE EXPERIMENT C: LOG-TRANSFORMED TARGET
# ============================================================

def create_log_target_model():
    """
    Train the Random Forest on log1p(price).

    During prediction, TransformedTargetRegressor automatically
    applies expm1() to return predictions in the original units.

    This experiment uses the same Random Forest hyperparameters
    as the baseline so we can investigate the target transformation.
    """

    base_pipeline = Pipeline(
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

    transformed_model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    return transformed_model


# ============================================================
# 8. REGISTER THE EXPERIMENTS
# ============================================================

experiments = {
    "A_Baseline_Random_Forest": create_baseline_model(),
    "B_Tuned_Random_Forest": create_location_experiment_model(),
    "C_Log_Transformed_Target": create_log_target_model()
}


# ============================================================
# 9. TRAIN AND EVALUATE EACH MODEL
# ============================================================

all_results = []

prediction_results = {}

for model_name, model in experiments.items():

    print("\n" + "=" * 65)
    print("Training:", model_name)
    print("=" * 65)

    # Train using exactly the same training data.

    model.fit(
        X_train,
        y_train
    )

    # Predict on exactly the same test data.

    predictions = model.predict(X_test)

    # Protect against tiny negative numerical predictions.

    predictions = np.maximum(
        predictions,
        0
    )

    # Calculate evaluation metrics.

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

    median_error = np.median(
        np.abs(
            y_test.to_numpy() - predictions
        )
    )

    print(f"MAE: {mae:.4f} lakhs")
    print(f"RMSE: {rmse:.4f} lakhs")
    print(f"R2: {r2:.4f}")
    print(f"Median absolute error: {median_error:.4f} lakhs")

    # Store predictions for later comparison.

    prediction_results[model_name] = predictions

    # Save evaluation metrics.

    all_results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Median_Absolute_Error": median_error
        }
    )


# ============================================================
# 10. CREATE THE MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

comparison_path = (
    OUTPUT_DIR
    / "stage05_2_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    results_df.round(4).to_string(index=False)
)

print("\nSaved model comparison to:")
print(comparison_path)


# ============================================================
# 11. EVALUATE MODELS BY ACTUAL PRICE RANGE
# ============================================================

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

price_range_results = []

for model_name, predictions in prediction_results.items():

    errors = np.abs(
        y_test.to_numpy() - predictions
    )

    temp = pd.DataFrame(
        {
            "price_group": price_groups.to_numpy(),
            "absolute_error": errors
        }
    )

    grouped = (
        temp.groupby(
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

    grouped["Model"] = model_name

    price_range_results.append(
        grouped
    )

price_range_df = pd.concat(
    price_range_results,
    ignore_index=True
)

price_range_path = (
    OUTPUT_DIR
    / "stage05_2_price_range_comparison.csv"
)

price_range_df.to_csv(
    price_range_path,
    index=False
)

print("\nPRICE-RANGE MODEL COMPARISON")
print("=" * 65)

print(
    price_range_df.round(2).to_string(index=False)
)

print("\nSaved price-range comparison to:")
print(price_range_path)


# ============================================================
# 12. CREATE MAE COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (lakhs)")
plt.title("Comparison of Model MAE")

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

mae_graph_path = (
    OUTPUT_DIR
    / "stage05_2_mae_comparison.png"
)

plt.savefig(
    mae_graph_path,
    dpi=300
)

plt.close()


# ============================================================
# 13. CREATE RMSE COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.xlabel("Model")
plt.ylabel("Root Mean Squared Error (lakhs)")
plt.title("Comparison of Model RMSE")

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

rmse_graph_path = (
    OUTPUT_DIR
    / "stage05_2_rmse_comparison.png"
)

plt.savefig(
    rmse_graph_path,
    dpi=300
)

plt.close()


# ============================================================
# 14. SAVE A TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "stage05_2_model_improvement_report.txt"
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
        "STAGE 5.2: MODEL IMPROVEMENT EXPERIMENTS\n"
    )

    report.write("=" * 60 + "\n\n")

    report.write(
        f"Training records: {len(X_train)}\n"
    )

    report.write(
        f"Testing records: {len(X_test)}\n\n"
    )

    report.write(
        "MODEL COMPARISON\n"
    )

    report.write(
        results_df.round(4).to_string(index=False)
    )

    report.write("\n\nPRICE-RANGE COMPARISON\n")

    report.write(
        price_range_df.round(2).to_string(index=False)
    )

    report.write(
        "\n\nInterpretation:\n"
    )

    report.write(
        "Lower MAE and RMSE indicate smaller prediction errors.\n"
    )

    report.write(
        "Higher R2 generally indicates better fit on this test split.\n"
    )

    report.write(
        "The best model should be selected using multiple metrics\n"
    )

    report.write(
        "and price-range performance, not a single metric alone.\n"
    )

print("\n" + "=" * 65)
print("STAGE 5.2 EXPERIMENTS COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print(comparison_path)
print(price_range_path)
print(mae_graph_path)
print(rmse_graph_path)
print(report_path)