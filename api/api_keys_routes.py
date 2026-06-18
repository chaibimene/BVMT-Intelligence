from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import get_db, APIKey, User, UserRole
from .auth import get_current_active_user, require_admin
from pydantic import BaseModel
from typing import Optional
import os
from datetime import datetime

router = APIRouter(prefix="/api/admin", tags=["admin"])


class APIKeyCreate(BaseModel):
    provider: str
    key_name: str
    encrypted_key: str
    is_active: bool = True


class APIKeyUpdate(BaseModel):
    provider: Optional[str] = None
    key_name: Optional[str] = None
    encrypted_key: Optional[str] = None
    is_active: Optional[bool] = None


class APIKeyResponse(BaseModel):
    id: int
    provider: str
    key_name: str
    masked_key: str
    is_active: bool
    created_at: str
    updated_at: str


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "••••••••"
    return f"{key[:4]}••••••••••••••••••••••{key[-4:]}"


@router.get("/api-keys", response_model=list[APIKeyResponse])
def get_api_keys(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    keys = db.query(APIKey).all()
    return [
        APIKeyResponse(
            id=k.id,
            provider=k.provider,
            key_name=k.key_name,
            masked_key=mask_key(k.encrypted_key),
            is_active=k.is_active,
            created_at=k.created_at.isoformat(),
            updated_at=k.updated_at.isoformat()
        ) for k in keys
    ]


@router.post("/api-keys", response_model=APIKeyResponse)
def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    key = APIKey(
        provider=key_data.provider,
        key_name=key_data.key_name,
        encrypted_key=key_data.encrypted_key,
        is_active=key_data.is_active
    )
    db.add(key)
    db.commit()
    db.refresh(key)
    
    # Update environment variable if possible
    env_var_name = f"{key_data.provider.upper()}_API_KEY"
    os.environ[env_var_name] = key_data.encrypted_key
    
    return APIKeyResponse(
        id=key.id,
        provider=key.provider,
        key_name=key.key_name,
        masked_key=mask_key(key.encrypted_key),
        is_active=key.is_active,
        created_at=key.created_at.isoformat(),
        updated_at=key.updated_at.isoformat()
    )


@router.put("/api-keys/{key_id}", response_model=APIKeyResponse)
def update_api_key(
    key_id: int,
    key_data: APIKeyUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    if key_data.provider is not None:
        key.provider = key_data.provider
    if key_data.key_name is not None:
        key.key_name = key_data.key_name
    if key_data.encrypted_key is not None:
        key.encrypted_key = key_data.encrypted_key
        # Update environment variable
        env_var_name = f"{key.provider.upper()}_API_KEY"
        os.environ[env_var_name] = key_data.encrypted_key
    if key_data.is_active is not None:
        key.is_active = key_data.is_active
    
    key.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(key)
    
    return APIKeyResponse(
        id=key.id,
        provider=key.provider,
        key_name=key.key_name,
        masked_key=mask_key(key.encrypted_key),
        is_active=key.is_active,
        created_at=key.created_at.isoformat(),
        updated_at=key.updated_at.isoformat()
    )


@router.delete("/api-keys/{key_id}")
def delete_api_key(
    key_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    db.delete(key)
    db.commit()
    return {"message": "API key deleted successfully"}