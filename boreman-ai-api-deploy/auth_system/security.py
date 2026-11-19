from datetime import datetime, timedelta, timezone
import jwt
from pwdlib import PasswordHash
from .config import settings

password_hash = PasswordHash.recommended()

def hash_password(password: str) -> str:
    return password_hash.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return password_hash.verify(password, hashed)

def create_access_token(user_id: int, token_version: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "ver": token_version,
               "exp": now + timedelta(minutes=settings.access_token_minutes), "iat": now}
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")

def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=["HS256"])
