# --------------------------------------------------
# PREDICTION HISTORY ROUTES
# --------------------------------------------------

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import (
    User,
    PredictionHistory,
)

from backend.auth import get_current_user

from backend.schemas import (
    HousePredictionRequest,
    HousePredictionResponse,
    PredictionHistoryResponse,
)


router = APIRouter(
    prefix="/predictions",
    tags=["Prediction History"],
)


# --------------------------------------------------
# SAVE A PREDICTION
# --------------------------------------------------

def save_prediction(
    db: Session,
    user: User,
    house: HousePredictionRequest,
    predicted_price: float,
):

    prediction = PredictionHistory(
        user_id=user.id,
        location=house.location.strip(),
        total_sqft=house.total_sqft,
        bhk=house.bhk,
        bath=house.bath,
        balcony=house.balcony,
        predicted_price_lakhs=round(
            predicted_price,
            2,
        ),
        predicted_price_crores=round(
            predicted_price / 100,
            4,
        ),
    )

    try:
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

    except Exception:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Could not save prediction history.",
        )

    return prediction


# --------------------------------------------------
# GET PREDICTION HISTORY
# --------------------------------------------------

@router.get(
    "/history",
    response_model=list[PredictionHistoryResponse],
)
def get_prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    history = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.user_id
            == current_user.id
        )
        .order_by(
            PredictionHistory.created_at.desc(),
            PredictionHistory.id.desc(),
        )
        .limit(100)
        .all()
    )

    return history


# --------------------------------------------------
# DELETE A SAVED PREDICTION
# --------------------------------------------------

@router.delete("/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        get_current_user
    ),
):

    prediction = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.id == prediction_id,
            PredictionHistory.user_id
            == current_user.id,
        )
        .first()
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    db.delete(prediction)
    db.commit()

    return {
        "message": "Prediction deleted successfully."
    }