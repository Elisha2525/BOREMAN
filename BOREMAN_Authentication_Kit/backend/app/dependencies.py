import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from .database import get_db
from .models import User
from .security import decode_access_token

bearer = HTTPBearer(auto_error=False)

def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
                 db: Session = Depends(get_db)) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        claims = decode_access_token(credentials.credentials)
        user_id, version = int(claims["sub"]), int(claims["ver"])
    except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired access token.")
    user = db.get(User, user_id)
    if not user or not user.is_active or user.status != "approved" or user.token_version != version:
        raise HTTPException(status_code=401, detail="Account inactive, pending approval, or session revoked.")
    return user

def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "administrator":
        raise HTTPException(status_code=403, detail="Administrator permission required.")
    return user
