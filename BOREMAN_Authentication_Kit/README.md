# BOREMAN Authentication Kit

A local-first authentication starter for the existing BOREMAN React + FastAPI project.

Includes email/password signup and login, pending approval, administrator/researcher/viewer roles, user activation/deactivation, logout, Argon2 password hashing, JWT authentication, and database-backed token revocation.

## A. Test the backend locally

Requires Python 3.11+.

PowerShell, from the extracted `backend` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`: set a unique `SECRET_KEY`, `BOOTSTRAP_ADMIN_EMAIL`, and `BOOTSTRAP_ADMIN_SECRET`. Then run:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 9001 --reload
```

Open `http://127.0.0.1:9001/docs`.

SQLite is used by default for the local smoke test. To connect PostgreSQL, replace `DATABASE_URL` in `.env` with your provider's connection string, e.g.:

```env
DATABASE_URL=postgresql+psycopg://USERNAME:PASSWORD@HOST:5432/DATABASE
```

## B. Create the first administrator

Call `POST /api/auth/bootstrap-admin` once in Swagger (`/docs`) with:

```json
{
  "email": "admin@example.com",
  "secret": "the-value-from-BOOTSTRAP_ADMIN_SECRET",
  "password": "A-unique-password-at-least-12-chars"
}
```

Use the same email as `BOOTSTRAP_ADMIN_EMAIL`. After success, remove `BOOTSTRAP_ADMIN_SECRET` from the environment.

## C. Connect the React frontend

Copy `frontend/src/auth/AuthPanel.jsx` and `frontend/src/auth/auth.css` into your Vite project's `src/auth/` folder.

Add `.env.local` in the Vite project root:

```env
VITE_AUTH_API_URL=http://127.0.0.1:9000
```

For an isolated UI test, import and render `<AuthPanel />`. To wrap the current dashboard:

```jsx
import AuthPanel from "./auth/AuthPanel";
import ExistingBoremanDashboard from "./ExistingBoremanDashboard";

export default function App() {
  return (
    <AuthPanel>
      <ExistingBoremanDashboard />
    </AuthPanel>
  );
}
```

Replace the example dashboard import with your actual existing component. The kit does not overwrite your current dashboard.

## D. Connect to the existing FastAPI API

Do not replace your current `boreman-ai-api/main.py` or Adafruit routes.

Copy the files in `backend/app/` into `boreman-ai-api/auth_system/`. In your existing `main.py`, add:

```python
from auth_system.database import Base as AuthBase, engine as auth_engine
from auth_system.router import router as auth_router

AuthBase.metadata.create_all(bind=auth_engine)
app.include_router(auth_router)
```

Merge the packages from `backend/requirements.txt` into your existing `requirements.txt`. Set the variables in `.env.example` in the local backend environment and later in Catalyst AppSail Environment Variables.

Protect backend routes explicitly using `Depends(current_user)` or `Depends(require_admin)`. Merely hiding the React dashboard is not backend security. See `backend/INTEGRATION.md`.

## E. Catalyst

Use PostgreSQL for production; AppSail local disk/SQLite is not a durable production database. Set `DATABASE_URL`, `SECRET_KEY`, `ACCESS_TOKEN_MINUTES`, `CORS_ORIGINS`, `BOOTSTRAP_ADMIN_EMAIL`, and the one-time bootstrap secret in AppSail environment variables. Never put database credentials, JWT secrets, Adafruit keys, or bootstrap secrets in Vite variables.

Deploy backend first. Verify `/health`, `/docs`, signup, bootstrap admin, login, `/me`, approval, deactivation, and logout before pointing the frontend at the deployed API.

## Scope and production checklist

This is a starter integration kit, not yet connected to your actual source files or Catalyst account. Before production, add password reset, email verification, login rate limiting, audit logging, database backups/migrations, and test the actual Catalyst CORS and database configuration. The one-time bootstrap secret must be removed after first-admin creation.
