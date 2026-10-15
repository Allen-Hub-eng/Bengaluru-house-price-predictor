
# ============================================================
# BENGALURU HOUSE PRICE PREDICTION
# STAGE 5: MODEL EVALUATION AND SAVING
# ============================================================

from pathlib import Path

import joblib
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


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prepared_houses.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"
MODEL_FOLDER = PROJECT_ROOT / "models"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

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
y = pd.to_numeric(df[TARGET], errors="coerce")

valid_rows = y.notna() & np.isfinite(y) & (y > 0)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

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

X["location"] = (
    X["location"]
    .astype("string")
    .str.strip()
    .str.lower()
)

X["location"] = X["location"].replace("", pd.NA)

valid_locations = X["location"].notna()

X = X.loc[valid_locations].copy()
y = y.loc[valid_locations].copy()


# ------------------------------------------------------------
# 3. RECREATE THE SAME TRAIN/TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# ------------------------------------------------------------

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
            OneHotEncoder(handle_unknown="ignore")
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


# ------------------------------------------------------------
# 5. CREATE THE RANDOM FOREST MODEL
# ------------------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ]
)


# ------------------------------------------------------------
# 6. TRAIN THE MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model_pipeline.fit(X_train, y_train)

print("Training completed.")


# ------------------------------------------------------------
# 7. PREDICT TEST PRICES
# ------------------------------------------------------------

predictions = model_pipeline.predict(X_test)

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

print("\nMODEL EVALUATION RESULTS")
print("=" * 45)

print(f"MAE:  {mae:.4f} lakhs")
print(f"RMSE: {rmse:.4f} lakhs")
print(f"R2:   {r2:.4f}")


# ------------------------------------------------------------
# 8. CREATE PREDICTION COMPARISON TABLE
# ------------------------------------------------------------

comparison = X_test.copy()

comparison["actual_price"] = y_test.to_numpy()
comparison["predicted_price"] = predictions

comparison["absolute_error"] = np.abs(
    comparison["actual_price"]
    - comparison["predicted_price"]
)

comparison["percentage_error"] = (
    comparison["absolute_error"]
    / comparison["actual_price"]
) * 100

comparison = comparison.sort_values(
    by="absolute_error",
    ascending=False
)

comparison_path = (
    OUTPUT_FOLDER
    / "stage05_prediction_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\nFive predictions with the largest absolute errors:")

print(
    comparison[
        [
            "location",
            "total_sqft",
            "bhk",
            "actual_price",
            "predicted_price",
            "absolute_error"
        ]
    ].head(5).to_string(index=False)
)


# ------------------------------------------------------------
# 9. ACTUAL VS PREDICTED PRICE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

minimum = min(
    float(y_test.min()),
    float(np.min(predictions))
)

maximum = max(
    float(y_test.max()),
    float(np.max(predictions))
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")

plt.tight_layout()

graph_path = (
    OUTPUT_FOLDER
    / "stage05_actual_vs_predicted.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. SAVE THE TRAINED MODEL
# ------------------------------------------------------------

model_path = MODEL_FOLDER / "house_price_model.joblib"

joblib.dump(
    model_pipeline,
    model_path
)

print("\nSaved model:", model_path)


# ------------------------------------------------------------
# 11. SAVE EVALUATION REPORT
# ------------------------------------------------------------

report_path = (
    OUTPUT_FOLDER
    / "stage05_evaluation_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("BENGALURU HOUSE PRICE PREDICTION\n")
    report.write("STAGE 5: MODEL EVALUATION REPORT\n")
    report.write("=" * 55 + "\n\n")

    report.write(f"Training records: {len(X_train)}\n")
    report.write(f"Testing records: {len(X_test)}\n\n")

    report.write("Evaluation metrics:\n")
    report.write(f"MAE: {mae:.4f} lakhs\n")
    report.write(f"RMSE: {rmse:.4f} lakhs\n")
    report.write(f"R2: {r2:.4f}\n\n")

    report.write(
        "Selected model: Random Forest Regressor\n"
    )

    report.write(
        "\nNote: These metrics evaluate the model "
        "on the held-out test split. They do not "
        "guarantee the same performance on future properties.\n"
    )

print("\nStage 5 evaluation completed!")

print("\nFiles created:")
print(comparison_path)
print(graph_path)
print(report_path)

