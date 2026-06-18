from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import get_db, Conversation, Message, User
from .auth import get_current_active_user
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import json

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: int
    title: Optional[str]
    created_at: str
    updated_at: str
    message_count: int


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[List[dict]]
    confidence: Optional[float]
    timestamp: str


class ConversationDetailResponse(BaseModel):
    id: int
    title: Optional[str]
    created_at: str
    updated_at: str
    messages: List[MessageResponse]


def get_time_group(dt: datetime) -> str:
    now = datetime.utcnow()
    diff = now - dt
    
    if diff.days == 0:
        return "Aujourd'hui"
    elif diff.days == 1:
        return "Hier"
    elif diff.days < 7:
        return "Cette semaine"
    elif diff.days < 30:
        return "Ce mois"
    else:
        return "Plus ancien"


@router.post("/", response_model=ConversationResponse)
def create_conversation(
    conv_data: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    conversation = Conversation(
        user_id=current_user.id,
        title=conv_data.title or "Nouvelle conversation"
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
        message_count=0
    )


@router.get("/", response_model=List[dict])
def list_conversations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()
    
    result = []
    for conv in conversations:
        message_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        result.append({
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "message_count": message_count,
            "time_group": get_time_group(conv.updated_at)
        })
    
    return result


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp).all()
    
    message_responses = []
    for msg in messages:
        sources = None
        if msg.sources:
            try:
                sources = json.loads(msg.sources)
            except:
                sources = None
        
        message_responses.append(MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            sources=sources,
            confidence=msg.confidence,
            timestamp=msg.timestamp.isoformat()
        ))
    
    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
        messages=message_responses
    )


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted successfully"}


@router.post("/{conversation_id}/messages")
def add_message(
    conversation_id: int,
    role: str,
    content: str,
    sources: Optional[List[dict]] = None,
    confidence: Optional[float] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=json.dumps(sources) if sources else None,
        confidence=confidence
    )
    db.add(message)
    
    # Update conversation timestamp
    conversation.updated_at = datetime.utcnow()
    
    # Auto-update title from first user message
    if role == "user" and db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.role == "user"
    ).count() == 1:
        conversation.title = content[:100]
    
    db.commit()
    db.refresh(message)
    
    return {"message": "Message added", "id": message.id}


@router.post("/search")
def search_conversations(
    query: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.title.ilike(f"%{query}%")
    ).order_by(Conversation.updated_at.desc()).all()
    
    result = []
    for conv in conversations:
        message_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        result.append({
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "message_count": message_count,
            "time_group": get_time_group(conv.updated_at)
        })
    
    return result


@router.delete("/")
def clear_all_conversations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).all()
    
    for conv in conversations:
        db.delete(conv)
    
    db.commit()
    return {"message": "All conversations deleted"}