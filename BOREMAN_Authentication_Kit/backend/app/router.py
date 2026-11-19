from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from .config import settings
from .database import get_db
from .dependencies import current_user, require_admin
from .models import User
from .schemas import SignupIn, LoginIn, BootstrapAdminIn, UserUpdateIn
from .security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

def safe_user(u: User):
    return {"id": u.id, "full_name": u.full_name, "email": u.email,
            "role": u.role, "status": u.status, "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None}

@router.post("/signup", status_code=201)
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    user = User(full_name=payload.full_name.strip(), email=email,
                password_hash=hash_password(payload.password), role="viewer",
                status="pending", is_active=True)
    db.add(user); db.commit(); db.refresh(user)
    return {"message": "Signup submitted. An administrator must approve your account.",
            "user": safe_user(user)}

@router.post("/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email.lower().strip()))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account has been deactivated.")
    if user.status != "approved":
        raise HTTPException(status_code=403, detail="Your account is awaiting administrator approval.")
    return {"access_token": create_access_token(user.id, user.token_version),
            "token_type": "bearer", "user": safe_user(user)}

@router.post("/logout")
def logout(user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.token_version += 1
    db.commit()
    return {"message": "Logged out; previous tokens revoked."}

@router.get("/me")
def me(user: User = Depends(current_user)):
    return safe_user(user)

@router.post("/bootstrap-admin")
def bootstrap_admin(payload: BootstrapAdminIn, db: Session = Depends(get_db)):
    if not settings.bootstrap_admin_secret or payload.secret != settings.bootstrap_admin_secret:
        raise HTTPException(status_code=403, detail="Bootstrap is not configured or secret is incorrect.")
    if db.scalar(select(User.id).where(User.role == "administrator")):
        raise HTTPException(status_code=409, detail="An administrator already exists.")
    email = payload.email.lower().strip()
    configured = settings.bootstrap_admin_email.lower().strip()
    if configured and email != configured:
        raise HTTPException(status_code=403, detail="Email does not match BOOTSTRAP_ADMIN_EMAIL.")
    user = db.scalar(select(User).where(User.email == email))
    if user:
        user.password_hash = hash_password(payload.password)
        user.role, user.status, user.is_active = "administrator", "approved", True
        user.token_version += 1
    else:
        user = User(full_name="BOREMAN Administrator", email=email,
                    password_hash=hash_password(payload.password), role="administrator",
                    status="approved", is_active=True)
        db.add(user)
    db.commit(); db.refresh(user)
    return {"message": "Administrator created. Remove BOOTSTRAP_ADMIN_SECRET now.",
            "user": safe_user(user)}

@router.get("/users")
def users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return [safe_user(u) for u in db.scalars(select(User).order_by(User.created_at.desc())).all()]

@router.patch("/users/{user_id}")
def update_user(user_id: int, payload: UserUpdateIn, admin: User = Depends(require_admin),
                db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.id == admin.id and payload.is_active is False:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own administrator account.")
    if payload.role is not None: user.role = payload.role
    if payload.status is not None: user.status = payload.status
    if payload.is_active is not None: user.is_active = payload.is_active
    user.token_version += 1
    user.updated_at = datetime.now(timezone.utc)
    db.commit(); db.refresh(user)
    return {"message": "User updated; existing sessions revoked.", "user": safe_user(user)}
