from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_houses.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODEL_DIR / "final_house_price_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
LOCATION_THRESHOLD = 10

FEATURES = [
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony",
]

NUMERIC_FEATURES = [
    "total_sqft",
    "bhk",
    "bath",
    "balcony",
]

CATEGORICAL_FEATURES = ["location"]

# Hyperparameters selected in Stage 5.4
BEST_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 0.6,
}

# ==================================================
# 2. DATA CONVERSION FUNCTIONS
# ==================================================

def convert_sqft(value):
    """Convert numeric square-foot values and ranges."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace(",", "")

    numbers = re.findall(r"\d+(?:\.\d+)?", value)

    if not numbers:
        return np.nan

    values = [float(number) for number in numbers]

    return sum(values) / len(values)


def extract_bhk(value):
    """Extract BHK from values such as '3 BHK'."""

    if pd.isna(value):
        return np.nan

    match = re.search(r"(\d+(?:\.\d+)?)", str(value))

    if match:
        return float(match.group(1))

    return np.nan

# ==================================================
# 3. LOCATION FREQUENCY GROUPER
# ==================================================

class LocationFrequencyGrouper(
    BaseEstimator,
    TransformerMixin
):
    """
    Group rare and unseen locations into 'other'.

    The frequency list is learned only when fit() is called.
    """

    def __init__(self, threshold=10):
        self.threshold = threshold

    def fit(self, X, y=None):
        counts = X["location"].value_counts()

        self.frequent_locations_ = set(
            counts[counts >= self.threshold].index
        )

        return self

    def transform(self, X):
        X = X.copy()

        X["location"] = X["location"].where(
            X["location"].isin(self.frequent_locations_),
            "other",
        )

        return X

# ==================================================
# 4. CREATE COMPLETE ML PIPELINE
# ==================================================

def create_pipeline():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    random_forest = RandomForestRegressor(
        **BEST_PARAMETERS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "location_grouper",
                LocationFrequencyGrouper(
                    threshold=LOCATION_THRESHOLD
                ),
            ),
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                random_forest,
            ),
        ]
    )

    return pipeline

# ==================================================
# 5. LOAD DATASET
# ==================================================

print("\nLoading cleaned dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "location",
    "total_sqft",
    "size",
    "bath",
    "balcony",
    "price",
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

df["location"] = (
    df["location"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["location"] = df["location"].replace("", pd.NA)

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

df["bhk"] = df["size"].apply(extract_bhk)

for column in ["bath", "balcony", "price"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

# ==================================================
# 6. APPLY DATA QUALITY FILTERS
# ==================================================

df = df.dropna(
    subset=[
        "location",
        "total_sqft",
        "bhk",
        "bath",
        "price",
    ]
).copy()

df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

df = df[
    (df["total_sqft"] >= 300)
    & (df["total_sqft"] <= 20000)
    & (df["bhk"] >= 1)
    & (df["bhk"] <= 10)
    & (df["bath"] >= 1)
    & (df["bath"] <= 10)
    & (df["sqft_per_bhk"] >= 200)
    & (df["price"] > 0)
].copy()

X = df[FEATURES].copy()
y = df["price"].copy()

print(f"Usable records: {len(df)}")

# ==================================================
# 7. RECREATE THE HELD-OUT TEST EVALUATION
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# Fit a fresh copy of the selected pipeline on the
# training split to verify reproducible performance.

evaluation_model = create_pipeline()

evaluation_model.fit(X_train, y_train)

test_predictions = evaluation_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    test_predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions,
    )
)

r2 = r2_score(
    y_test,
    test_predictions,
)

median_error = median_absolute_error(
    y_test,
    test_predictions,
)

print("\nHELD-OUT TEST METRICS")
print("-" * 40)
print(f"MAE: {mae:.4f} lakhs")
print(f"RMSE: {rmse:.4f} lakhs")
print(f"R2: {r2:.4f}")
print(f"Median absolute error: {median_error:.4f} lakhs")

# ==================================================
# 8. SAVE FINAL TEST METRICS
# ==================================================

metrics = pd.DataFrame(
    [
        {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Median_Absolute_Error": median_error,
            "Training_Records": len(X_train),
            "Testing_Records": len(X_test),
        }
    ]
)

metrics_path = OUTPUT_DIR / "stage05_5_test_metrics.csv"

metrics.to_csv(
    metrics_path,
    index=False,
)

# ==================================================
# 9. RETRAIN THE DEPLOYMENT MODEL ON ALL APPROVED DATA
# ==================================================

print("\nTraining the final deployment model...")

final_model = create_pipeline()

final_model.fit(X, y)

# Save the entire pipeline, including preprocessing.
joblib.dump(
    final_model,
    FINAL_MODEL_PATH,
)

print(f"Final model saved to: {FINAL_MODEL_PATH}")

# ==================================================
# 10. TEST THE SAVED MODEL
# ==================================================

print("\nTesting saved model...")

loaded_model = joblib.load(FINAL_MODEL_PATH)

sample_input = pd.DataFrame(
    [
        {
            "location": "whitefield",
            "total_sqft": 1200,
            "bhk": 2,
            "bath": 2,
            "balcony": 1,
        }
    ]
)

sample_prediction = loaded_model.predict(sample_input)[0]

print("\nSAMPLE PREDICTION")
print("-" * 40)
print(f"Location: Whitefield")
print(f"Area: 1200 sq ft")
print(f"Bedrooms: 2 BHK")
print(f"Bathrooms: 2")
print(f"Balconies: 1")
print(
    f"Predicted price: ₹{sample_prediction:.2f} lakhs"
)

# Check that unseen locations can be processed.
unseen_input = sample_input.copy()

unseen_input["location"] = "example_unseen_location"

unseen_prediction = loaded_model.predict(unseen_input)[0]

print(
    "\nUnseen-location prediction completed successfully."
)
print(
    f"Predicted price: ₹{unseen_prediction:.2f} lakhs"
)

print("\nSTAGE 5.5 COMPLETED SUCCESSFULLY!")
print(f"Final model: {FINAL_MODEL_PATH}")
print(f"Test metrics: {metrics_path}")