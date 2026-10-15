# --------------------------------------------------
# PYDANTIC SCHEMAS
# --------------------------------------------------

from datetime import datetime

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    EmailStr,
)


# --------------------------------------------------
# 1. USER REGISTRATION
# --------------------------------------------------

class UserRegisterRequest(BaseModel):

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
    )

    email: EmailStr

    password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )


class UserResponse(BaseModel):

    id: int

    username: str

    email: EmailStr

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


# --------------------------------------------------
# 2. USER LOGIN
# --------------------------------------------------

class UserLoginRequest(BaseModel):

    email: EmailStr

    password: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )


class TokenResponse(BaseModel):

    access_token: str

    token_type: str = "bearer"

    expires_in_minutes: int


# --------------------------------------------------
# 3. HOUSE PRICE PREDICTION REQUEST
# --------------------------------------------------

class HousePredictionRequest(BaseModel):

    location: str = Field(
        ...,
        min_length=1,
        max_length=150,
    )

    total_sqft: float = Field(
        ...,
        gt=0,
        le=20000,
    )

    bhk: int = Field(
        ...,
        ge=1,
        le=10,
    )

    bath: int = Field(
        ...,
        ge=1,
        le=10,
    )

    balcony: int | None = Field(
        default=None,
        ge=0,
        le=3,
    )


# --------------------------------------------------
# 4. HOUSE PRICE PREDICTION RESPONSE
# --------------------------------------------------

class HousePredictionResponse(BaseModel):

    location: str

    predicted_price_lakhs: float

    predicted_price_crores: float

    currency: str


# --------------------------------------------------
# 5. SAVED PREDICTION RESPONSE
# --------------------------------------------------

class PredictionHistoryResponse(BaseModel):

    id: int

    location: str

    total_sqft: float

    bhk: int

    bath: int

    balcony: int | None

    predicted_price_lakhs: float

    predicted_price_crores: float

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )