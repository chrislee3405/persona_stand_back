from typing import Optional
from fastapi import Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import code as code_models   

class CodeService:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    async def process_invite_code(self, input_code: str) -> Optional[str]:
        if not input_code.strip():
            return None

        result = (
            self.db.query(code_models.InviteCode)
            .filter(code_models.InviteCode.code == input_code)
            .first()
        )

        if result is None:
            return 'Code not found, you can continue as guest'

        return result.code