
# ============================================================
# BENGALURU HOUSE PRICE PREDICTION
# STAGE 4: TRAIN AND COMPARE MACHINE LEARNING MODELS
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 1. SET PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prepared_houses.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Prepared dataset not found. "
        "Run ml/prepare_features.py first."
    )


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("BENGALURU HOUSE PRICE PREDICTION")
print("STAGE 4: MODEL TRAINING")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nDataset columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. DEFINE INPUT FEATURES AND TARGET
# ------------------------------------------------------------

FEATURES = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony"
]

TARGET = "price"

missing_columns = [
    column
    for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Inputs used to predict the house price.
X = df[FEATURES].copy()

# Actual house prices that the model will learn to predict.
y = pd.to_numeric(df[TARGET], errors="coerce")

# Remove rows with a missing or invalid target.
valid_rows = y.notna() & np.isfinite(y) & (y > 0)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Ensure numerical features have numerical data types.
NUMERICAL_FEATURES = [
    "total_sqft",
    "bhk",
    "bath",
    "balcony"
]

for column in NUMERICAL_FEATURES:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

# Normalize location labels.
X["location"] = (
    X["location"]
    .astype("string")
    .str.strip()
    .str.lower()
)

X["location"] = X["location"].replace("", pd.NA)

# Drop records without a usable location.
valid_locations = X["location"].notna()

X = X.loc[valid_locations].copy()
y = y.loc[valid_locations].copy()

print("\nUsable dataset shape:", X.shape)


# ------------------------------------------------------------
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------------------
# 5. BUILD PREPROCESSING PIPELINE
# ------------------------------------------------------------

# Numerical features:
# 1. Replace missing values with the training median.
# 2. Standardize values for models that benefit from scaling.

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


# Categorical feature:
# 1. Replace missing locations with the most frequent location.
# 2. Convert location categories into numerical columns.
# 3. Ignore unseen categories rather than failing.

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Apply the correct preprocessing to each feature type.

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


# ------------------------------------------------------------
# 6. DEFINE MACHINE LEARNING MODELS
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ------------------------------------------------------------
# 7. TRAIN AND EVALUATE EACH MODEL
# ------------------------------------------------------------

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)

    # Keep preprocessing and model training together.
    # The preprocessing is fitted only on training data.

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train the model.
    pipeline.fit(X_train, y_train)

    # Predict prices for unseen test records.
    predictions = pipeline.predict(X_test)

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

    print(f"MAE:  {mae:.4f} lakhs")
    print(f"RMSE: {rmse:.4f} lakhs")
    print(f"R2:   {r2:.4f}")

    results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )

    trained_models[model_name] = pipeline


# ------------------------------------------------------------
# 8. CREATE MODEL COMPARISON REPORT
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

# Lower MAE is the primary ranking criterion.
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

results_path = OUTPUT_FOLDER / "stage04_model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]

print("\nBest model by test MAE:", best_model_name)


# ------------------------------------------------------------
# 9. SAVE COMPARISON GRAPH: MAE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.title("Model Comparison - Mean Absolute Error")
plt.xlabel("Machine Learning Model")
plt.ylabel("MAE (Price in Lakhs)")
plt.xticks(rotation=20, ha="right")

plt.tight_layout()

mae_graph_path = OUTPUT_FOLDER / "stage04_mae_comparison.png"

plt.savefig(
    mae_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. SAVE COMPARISON GRAPH: R2
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.title("Model Comparison - R2 Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("R2 Score")
plt.xticks(rotation=20, ha="right")

plt.tight_layout()

r2_graph_path = OUTPUT_FOLDER / "stage04_r2_comparison.png"

plt.savefig(
    r2_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 11. SAVE TEXT REPORT
# ------------------------------------------------------------

report_path = OUTPUT_FOLDER / "stage04_training_report.txt"

with open(report_path, "w", encoding="utf-8") as report:

    report.write("BENGALURU HOUSE PRICE PREDICTION\n")
    report.write("STAGE 4: MODEL TRAINING REPORT\n")
    report.write("=" * 60 + "\n\n")

    report.write(f"Usable dataset records: {len(X)}\n")
    report.write(f"Training records: {len(X_train)}\n")
    report.write(f"Testing records: {len(X_test)}\n\n")

    report.write("Input features:\n")
    report.write(", ".join(FEATURES) + "\n\n")

    report.write(f"Target variable: {TARGET}\n\n")

    report.write("Model comparison:\n")
    report.write(results_df.to_string(index=False))
    report.write("\n\n")

    report.write(
        f"Best model by test MAE: {best_model_name}\n"
    )

    report.write(
        "\nNote: The best model will be saved "
        "as a final production artifact in Stage 5.\n"
    )


# ------------------------------------------------------------
# 12. FINISH
# ------------------------------------------------------------

print("\nStage 4 completed successfully!")

print("\nFiles created:")
print(results_path)
print(mae_graph_path)
print(r2_graph_path)
print(report_path)

