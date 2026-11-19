from typing import Literal
from pydantic import BaseModel, EmailStr, Field

Role = Literal["administrator", "researcher", "viewer"]

class SignupIn(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)

class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

class BootstrapAdminIn(BaseModel):
    email: EmailStr
    secret: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=12, max_length=128)

class UserUpdateIn(BaseModel):
    role: Role | None = None
    is_active: bool | None = None
    status: Literal["pending", "approved", "rejected"] | None = None
