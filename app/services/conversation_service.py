from sqlalchemy.orm import Session
from fastapi import Depends
from app.database import get_db
from app.models import conversation as conversation_models

class ConversationService:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    async def save_conversation(self, content: list, code: str | None) -> None:
        new_entry = conversation_models.UserInput(
            content=content,
            code=code or "GUEST"
        )
        self.db.add(new_entry)
        self.db.commit()