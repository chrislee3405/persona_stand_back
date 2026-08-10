from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import conversation as models


router = APIRouter(prefix="/api", tags=["dialogue"])



@router.get("/guestchat")
async def get_chat_dialogues():
    return {"reach guestchat"}


@router.post("/save-conversation")
async def save_conversation(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    new_entry = models.UserInput(
        content=body.get("content", []),
        code=body.get("code") or "GUEST"
    )
    db.add(new_entry)
    db.commit()
    return {"status": "saved"}

