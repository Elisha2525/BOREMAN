from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite:///./boreman_auth.db"
    secret_key: str = "CHANGE_ME"
    access_token_minutes: int = 60
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    bootstrap_admin_email: str = ""
    bootstrap_admin_secret: str = ""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self):
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

settings = Settings()
