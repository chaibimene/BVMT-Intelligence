from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import get_db, UserSettings, User
from .auth import get_current_active_user
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    language: Optional[str] = None
    theme: Optional[str] = None
    notifications_enabled: Optional[bool] = None
    auto_save_conversations: Optional[bool] = None
    show_confidence_scores: Optional[bool] = None
    show_source_citations: Optional[bool] = None


class SettingsResponse(BaseModel):
    language: str
    theme: str
    notifications_enabled: bool
    auto_save_conversations: bool
    show_confidence_scores: bool
    show_source_citations: bool


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


@router.get("/", response_model=SettingsResponse)
def get_settings(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    settings = db.query(UserSettings).filter(UserSettings.user_id == current_user.id).first()
    
    if not settings:
        # Create default settings
        settings = UserSettings(user_id=current_user.id)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    
    return SettingsResponse(
        language=settings.language,
        theme=settings.theme,
        notifications_enabled=settings.notifications_enabled,
        auto_save_conversations=settings.auto_save_conversations,
        show_confidence_scores=settings.show_confidence_scores,
        show_source_citations=settings.show_source_citations
    )


@router.put("/", response_model=SettingsResponse)
def update_settings(
    settings_data: SettingsUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    settings = db.query(UserSettings).filter(UserSettings.user_id == current_user.id).first()
    
    if not settings:
        settings = UserSettings(user_id=current_user.id)
        db.add(settings)
    
    if settings_data.language is not None:
        settings.language = settings_data.language
    if settings_data.theme is not None:
        settings.theme = settings_data.theme
    if settings_data.notifications_enabled is not None:
        settings.notifications_enabled = settings_data.notifications_enabled
    if settings_data.auto_save_conversations is not None:
        settings.auto_save_conversations = settings_data.auto_save_conversations
    if settings_data.show_confidence_scores is not None:
        settings.show_confidence_scores = settings_data.show_confidence_scores
    if settings_data.show_source_citations is not None:
        settings.show_source_citations = settings_data.show_source_citations
    
    db.commit()
    db.refresh(settings)
    
    return SettingsResponse(
        language=settings.language,
        theme=settings.theme,
        notifications_enabled=settings.notifications_enabled,
        auto_save_conversations=settings.auto_save_conversations,
        show_confidence_scores=settings.show_confidence_scores,
        show_source_citations=settings.show_source_citations
    )


@router.put("/profile")
def update_profile(
    profile_data: ProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if profile_data.name is not None:
        current_user.name = profile_data.name
    if profile_data.email is not None:
        # Check if email already exists
        existing = db.query(User).filter(User.email == profile_data.email, User.id != current_user.id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        current_user.email = profile_data.email
    
    db.commit()
    db.refresh(current_user)
    
    return {"message": "Profile updated", "name": current_user.name, "email": current_user.email}


@router.put("/password")
def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    from .auth import verify_password, get_password_hash
    
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()
    
    return {"message": "Password changed successfully"}