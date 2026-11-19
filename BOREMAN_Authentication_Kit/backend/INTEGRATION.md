# Mount the auth router into the existing BOREMAN API

Do not replace your existing `boreman-ai-api/main.py` or Adafruit routes.

1. Copy the files inside `backend/app/` into `boreman-ai-api/auth_system/`.
2. Add to your existing `main.py` after the FastAPI `app` is created:

```python
from auth_system.database import Base as AuthBase, engine as auth_engine
from auth_system.router import router as auth_router

AuthBase.metadata.create_all(bind=auth_engine)
app.include_router(auth_router)
```

3. Merge `backend/requirements.txt` packages into your existing `requirements.txt`.
4. Configure `DATABASE_URL`, `SECRET_KEY`, `ACCESS_TOKEN_MINUTES`, `CORS_ORIGINS`, `BOOTSTRAP_ADMIN_EMAIL`, and `BOOTSTRAP_ADMIN_SECRET`.
5. Keep current Adafruit endpoints. Do not replace the current app object.
6. Add `Depends(current_user)` to routes that require login and `Depends(require_admin)` to admin-only routes.

If the project already has a SQLAlchemy engine/session, consolidate the auth models onto that database setup rather than creating a second engine. Use migrations for production schema changes.
