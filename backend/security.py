# --------------------------------------------------
# SECURITY: PASSWORD HASHING AND JWT AUTHENTICATION
# --------------------------------------------------

import os

from datetime import datetime, timedelta, timezone

import jwt

from pwdlib import PasswordHash


# --------------------------------------------------
# 1. SECURITY CONFIGURATION
# --------------------------------------------------

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not JWT_SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY is missing. "
        "Configure it in backend/.env."
    )

JWT_ALGORITHM = os.getenv(
    "JWT_ALGORITHM",
    "HS256",
)

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv(
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        "30",
    )
)


# --------------------------------------------------
# 2. PASSWORD HASHING
# --------------------------------------------------

password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """
    Convert a plaintext password into a secure hash.
    The original password is never stored.
    """

    return password_hasher.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a supplied password against its stored hash.
    """

    return password_hasher.verify(
        plain_password,
        hashed_password,
    )


# --------------------------------------------------
# 3. CREATE JWT ACCESS TOKEN
# --------------------------------------------------

def create_access_token(
    user_id: int,
    email: str,
) -> str:
    """
    Create a signed JWT token for an authenticated user.
    """

    expiration = datetime.now(
        timezone.utc
    ) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expiration,
    }

    token = jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    return token


# --------------------------------------------------
# 4. VERIFY JWT ACCESS TOKEN
# --------------------------------------------------

def decode_access_token(token: str) -> dict:
    """
    Verify the JWT signature and expiration.
    Raises an exception if the token is invalid.
    """

    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
    )