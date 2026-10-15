# --------------------------------------------------
# 1. IMPORTS AND ENVIRONMENT CONFIGURATION
# --------------------------------------------------

from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_house_price_model.joblib"
)

# --------------------------------------------------
# 3. LOAD ENVIRONMENT VARIABLES
# IMPORTANT: Do this before importing security/auth.
# --------------------------------------------------

ENV_PATH = PROJECT_ROOT / "backend" / ".env"

load_dotenv(ENV_PATH)

# --------------------------------------------------
# 4. OTHER IMPORTS
# --------------------------------------------------

import __main__
import joblib
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
)

from fastapi.middleware.cors import CORSMiddleware

from sklearn.base import BaseEstimator, TransformerMixin

from sqlalchemy.orm import Session

from backend.schemas import (
    HousePredictionRequest,
    HousePredictionResponse,
)

from backend.database import get_db

from backend.models import User

from backend.auth import (
    router as auth_router,
    get_current_user,
)

from backend.predictions import (
    router as predictions_router,
    save_prediction,
)


# --------------------------------------------------
# 2. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Bengaluru House Price Predictor",
    description=(
        "An API that predicts Bengaluru house prices "
        "using a trained machine learning model."
    ),
    version="1.0.0",
)

app.include_router(auth_router)

app.include_router(predictions_router)

# --------------------------------------------------
# 3. CONFIGURE CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 4. LOAD THE TRAINED MODEL
# --------------------------------------------------

# --------------------------------------------------
# LOCATION GROUPING TRANSFORMER
# Required to load the saved model
# --------------------------------------------------

class LocationFrequencyGrouper(BaseEstimator, TransformerMixin):

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


# Make the class available under the name
# expected by the saved model.
__main__.LocationFrequencyGrouper = LocationFrequencyGrouper


# Load the trained model.
model = joblib.load(MODEL_PATH)

print("House price prediction model loaded successfully!")



# --------------------------------------------------
# 5. ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to the Bengaluru House Price Predictor API",
        "status": "running",
        "docs": "/docs",
    }


# --------------------------------------------------
# 6. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# 7. HOUSE PRICE PREDICTION ENDPOINT
# --------------------------------------------------

# --------------------------------------------------
# HOUSE PRICE PREDICTION
# AUTHENTICATED USERS ONLY
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=HousePredictionResponse,
)
def predict_price(
    house: HousePredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    input_data = pd.DataFrame(
        [
            {
                "location": house.location.strip().lower(),
                "total_sqft": house.total_sqft,
                "bhk": house.bhk,
                "bath": house.bath,
                "balcony": house.balcony,
            }
        ]
    )

    try:

        predicted_price = float(
            model.predict(input_data)[0]
        )

    except Exception as error:

        print(
            "Prediction error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="The house price prediction failed.",
        ) from error

    if (
        not pd.notna(predicted_price)
        or predicted_price <= 0
    ):

        raise HTTPException(
            status_code=500,
            detail="The model returned an invalid price.",
        )

    # Save the successful prediction to MySQL.
    saved_prediction = save_prediction(
        db=db,
        user=current_user,
        house=house,
        predicted_price=predicted_price,
    )

    return HousePredictionResponse(
        location=saved_prediction.location,
        predicted_price_lakhs=round(
            float(
                saved_prediction.predicted_price_lakhs
            ),
            2,
        ),
        predicted_price_crores=round(
            float(
                saved_prediction.predicted_price_crores
            ),
            4,
        ),
        currency="INR",
    )