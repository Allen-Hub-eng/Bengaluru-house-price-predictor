
from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
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
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_houses.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
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

# --------------------------------------------------
# 2. HELPER FUNCTIONS FOR DATA CLEANING
# --------------------------------------------------

def convert_sqft(value):
    """Convert common square-foot values and ranges to numbers."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace(",", "")

    # Handle ranges such as '2100 - 2850'
    numbers = re.findall(r"\d+(?:\.\d+)?", value)

    if not numbers:
        return np.nan

    values = [float(number) for number in numbers]

    # Use the average for a range
    return sum(values) / len(values)

def extract_bhk(value):
    """Extract bedroom count from values such as '3 BHK'."""

    if pd.isna(value):
        return np.nan

    match = re.search(r"(\d+(?:\.\d+)?)", str(value))

    if match:
        return float(match.group(1))

    return np.nan

# --------------------------------------------------
# 3. LOCATION GROUPING TRANSFORMER
# --------------------------------------------------

class LocationFrequencyGrouper(BaseEstimator, TransformerMixin):
    """
    Keep locations appearing at least 'threshold' times
    in the data used to fit this transformer.

    Rare and previously unseen locations become 'other'.
    During cross-validation, this is fitted separately
    within each training fold to avoid location leakage.
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

# --------------------------------------------------
# 4. PREPROCESSING AND MODEL PIPELINE
# --------------------------------------------------

def create_pipeline(model):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            (
                "location_grouper",
                LocationFrequencyGrouper(
                    threshold=LOCATION_THRESHOLD
                ),
            ),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

# --------------------------------------------------
# 5. LOAD AND PREPARE DATA
# --------------------------------------------------

print("\nLoading cleaned dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that Stage 2 created cleaned_houses.csv."
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
    column for column in required_columns
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
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Match the plausibility filters used in Stage 3.
df = df.dropna(
    subset=["location", "total_sqft", "bhk", "bath", "price"]
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

# Keep only model inputs and target.
X = df[FEATURES].copy()
y = df["price"].copy()

print(f"Usable records: {len(df)}")
print(f"Number of input features: {len(FEATURES)}")

# Split before learning location frequencies.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# --------------------------------------------------
# 6. DEFINE THE BASELINE MODEL
# --------------------------------------------------

baseline_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

baseline_pipeline = create_pipeline(baseline_model)

print("\nTraining baseline model...")

baseline_pipeline.fit(X_train, y_train)
baseline_predictions = baseline_pipeline.predict(X_test)

# --------------------------------------------------
# 7. HYPERPARAMETER TUNING
# --------------------------------------------------

tuning_model = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=1,
)

tuning_pipeline = create_pipeline(tuning_model)

# RandomizedSearchCV samples 10 combinations from
# these available values and evaluates each using
# five-fold cross-validation.
parameter_options = {
    "model__n_estimators": [100, 150, 200, 300],
    "model__max_depth": [15, 20, 25, None],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [1.0, 0.8, 0.6],
}

search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=parameter_options,
    n_iter=10,
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2,
    refit=True,
    return_train_score=True,
)

print("\nStarting hyperparameter tuning...")
print("This may take several minutes on your computer.")

search.fit(X_train, y_train)

best_pipeline = search.best_estimator_

print("\nTuning complete!")
print("Best parameters:")
for name, value in search.best_params_.items():
    print(f"  {name}: {value}")

best_cv_mae = -search.best_score_

print(
    f"\nBest cross-validation MAE: "
    f"{best_cv_mae:.2f} lakhs"
)

# --------------------------------------------------
# 8. EVALUATE ON THE UNTOUCHED TEST SET
# --------------------------------------------------

tuned_predictions = best_pipeline.predict(X_test)

def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "Median_Absolute_Error": median_absolute_error(
            actual, predicted
        ),
    }

baseline_metrics = calculate_metrics(
    y_test, baseline_predictions
)

tuned_metrics = calculate_metrics(
    y_test, tuned_predictions
)

comparison = pd.DataFrame(
    [
        {
            "Model": "Baseline Random Forest",
            **baseline_metrics,
        },
        {
            "Model": "Tuned Random Forest",
            **tuned_metrics,
        },
    ]
)

comparison_path = (
    OUTPUT_DIR / "stage05_4_model_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

print("\nTEST SET MODEL COMPARISON")
print(comparison.round(4).to_string(index=False))

# --------------------------------------------------
# 9. SAVE ALL SEARCH RESULTS
# --------------------------------------------------

search_results = pd.DataFrame(search.cv_results_)

search_results["CV_MAE"] = (
    -search_results["mean_test_score"]
)

search_results = search_results.sort_values("CV_MAE")

search_results.to_csv(
    OUTPUT_DIR / "stage05_4_tuning_search_results.csv",
    index=False,
)

# Save the best experimental model separately.
model_path = (
    MODEL_DIR / "house_price_model_stage05_4.joblib"
)

joblib.dump(best_pipeline, model_path)

# --------------------------------------------------
# 10. ERROR ANALYSIS BY PRICE RANGE
# --------------------------------------------------

error_df = pd.DataFrame(
    {
        "Actual_Price": y_test.to_numpy(),
        "Baseline_Prediction": baseline_predictions,
        "Tuned_Prediction": tuned_predictions,
    }
)

error_df["Baseline_Absolute_Error"] = abs(
    error_df["Actual_Price"]
    - error_df["Baseline_Prediction"]
)

error_df["Tuned_Absolute_Error"] = abs(
    error_df["Actual_Price"]
    - error_df["Tuned_Prediction"]
)

error_df["Price_Group"] = pd.cut(
    error_df["Actual_Price"],
    bins=[0, 50, 100, 200, 500, np.inf],
    labels=[
        "0-50 lakhs",
        "50-100 lakhs",
        "100-200 lakhs",
        "200-500 lakhs",
        "500+ lakhs",
    ],
    include_lowest=True,
)

price_group_report = (
    error_df.groupby("Price_Group", observed=False)
    .agg(
        Property_Count=("Actual_Price", "size"),
        Baseline_MAE=("Baseline_Absolute_Error", "mean"),
        Tuned_MAE=("Tuned_Absolute_Error", "mean"),
    )
    .reset_index()
)

price_group_path = (
    OUTPUT_DIR / "stage05_4_price_range_errors.csv"
)

price_group_report.to_csv(price_group_path, index=False)

error_df.to_csv(
    OUTPUT_DIR / "stage05_4_test_predictions.csv",
    index=False,
)

print("\nERROR BY ACTUAL PRICE RANGE")
print(price_group_report.round(2).to_string(index=False))

# --------------------------------------------------
# 11. CREATE COMPARISON GRAPH
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["MAE"],
)

plt.ylabel("MAE (lakhs)")
plt.title("Stage 5.4: Baseline vs Tuned Random Forest")
plt.tight_layout()

graph_path = (
    OUTPUT_DIR / "stage05_4_model_comparison.png"
)

plt.savefig(graph_path, dpi=150)
plt.close()

# --------------------------------------------------
# 12. WRITE THE REPORT
# --------------------------------------------------

improvement = (
    baseline_metrics["MAE"] - tuned_metrics["MAE"]
)

report_path = (
    OUTPUT_DIR / "stage05_4_tuning_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("STAGE 5.4 - RANDOM FOREST TUNING REPORT\n")
    file.write("=" * 48 + "\n\n")

    file.write(f"Usable records: {len(df)}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n")
    file.write(
        f"Location threshold: {LOCATION_THRESHOLD}\n\n"
    )

    file.write("Best hyperparameters:\n")
    for name, value in search.best_params_.items():
        file.write(f"  {name}: {value}\n")

    file.write(
        f"\nBest 5-fold CV MAE: {best_cv_mae:.4f} lakhs\n\n"
    )

    file.write("Test-set metrics:\n")
    file.write(comparison.round(4).to_string(index=False))
    file.write("\n\n")

    file.write(
        f"MAE improvement over baseline: "
        f"{improvement:.4f} lakhs\n"
    )

    file.write(
        "\nNote: This is an experimental model. "
        "The held-out test metrics are estimates, not "
        "a guarantee of prediction accuracy for every property.\n"
    )

print("\nSTAGE 5.4 COMPLETE!")
print(f"Comparison CSV: {comparison_path}")
print(f"Search results: {OUTPUT_DIR / 'stage05_4_tuning_search_results.csv'}")
print(f"Price-range errors: {price_group_path}")
print(f"Test predictions: {OUTPUT_DIR / 'stage05_4_test_predictions.csv'}")
print(f"Comparison graph: {graph_path}")
print(f"Report: {report_path}")
print(f"Experimental model: {model_path}")