# --------------------------------------------------
# AUTHENTICATION ROUTES
# --------------------------------------------------

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from sqlalchemy.orm import Session

from sqlalchemy.exc import IntegrityError

import jwt

from backend.database import get_db
from backend.models import User
from backend.schemas import (
    UserRegisterRequest,
    UserResponse,
    UserLoginRequest,
    TokenResponse,
)

from backend.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

bearer_scheme = HTTPBearer(
    auto_error=False,
)


# --------------------------------------------------
# GET THE CURRENT AUTHENTICATED USER
# --------------------------------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> User:

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        payload = decode_access_token(token)

        subject = payload.get("sub")

        if subject is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token.",
            )

        user_id = int(subject)

    except (
        jwt.InvalidTokenError,
        ValueError,
        TypeError,
    ):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(
        User.id == user_id
    ).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
        )

    return user


# --------------------------------------------------
# 1. REGISTER
# --------------------------------------------------

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_user(
    request: UserRegisterRequest,
    db: Session = Depends(get_db),
):

    username = request.username.strip()

    email = str(request.email).strip().lower()

    if not username:
        raise HTTPException(
            status_code=400,
            detail="Username cannot be empty.",
        )

    existing_user = db.query(User).filter(
        User.email == email
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        )

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(
            request.password
        ),
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)

    except IntegrityError:

        db.rollback()

        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        )

    return user


# --------------------------------------------------
# 2. LOGIN
# --------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
)
def login_user(
    request: UserLoginRequest,
    db: Session = Depends(get_db),
):

    email = str(request.email).strip().lower()

    user = db.query(User).filter(
        User.email == email
    ).first()

    if (
        user is None
        or not verify_password(
            request.password,
            user.password_hash,
        )
    ):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token(
        user_id=user.id,
        email=user.email,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES,
    )


# --------------------------------------------------
# 3. CURRENT USER
# --------------------------------------------------

@router.get(
    "/me",
    response_model=UserResponse,
)
def get_my_profile(
    current_user: User = Depends(
        get_current_user
    ),
):

    return current_user