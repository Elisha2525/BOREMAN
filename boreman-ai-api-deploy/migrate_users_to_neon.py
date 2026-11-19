
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from auth_system.database import engine as neon_engine, Base
from auth_system.models import User

# Existing local SQLite database
sqlite_engine = create_engine("sqlite:///./boreman_auth.db")

# Ensure target tables exist
Base.metadata.create_all(bind=neon_engine)

with Session(sqlite_engine) as source, Session(neon_engine) as target:
    local_users = source.scalars(select(User)).all()

    migrated = 0
    skipped = 0

    for user in local_users:
        existing = target.scalar(
            select(User).where(User.email == user.email)
        )

        if existing:
            print(f"Already exists, skipped: {user.email}")
            skipped += 1
            continue

        new_user = User(
            full_name=user.full_name,
            email=user.email,
            password_hash=user.password_hash,
            role=user.role,
            status=user.status,
            is_active=user.is_active,
            token_version=user.token_version,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

        target.add(new_user)
        migrated += 1

    target.commit()

print("Migration completed.")
print("Migrated:", migrated)
print("Skipped:", skipped)
