from fastapi import Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import code as code_models
from app.services.conversation_manage_service import ConversationService


class InvalidCodeError(Exception):
    """Raised when a submitted invite code doesn't match any stored code."""
    pass


class CodeService:
    def __init__(
        self,
        db: Session = Depends(get_db),
        conversation_service: ConversationService = Depends()
    ):
        self.db = db
        self.conversation_service = conversation_service

    async def match_code(self, input_code: str, conversation_id: str | None, session_id: str) -> str:
        """
        Verifies an invite code, and if it's valid and tied to an existing
        conversation, updates that conversation's code in place instead of
        leaving it stuck on "GUEST". Raises InvalidCodeError if the code
        doesn't match any stored row, or ConversationAccessDeniedError
        (propagated from ConversationService) if the caller's session
        didn't create the conversation it's trying to upgrade.
        """
        if not input_code.strip():
            raise InvalidCodeError(input_code)

        result = (
            self.db.query(code_models.InviteCode)
            .filter(code_models.InviteCode.code == input_code)
            .first()
        )

        if result is None:
            raise InvalidCodeError(input_code)

        processed_result = result.code

        if conversation_id:
            await self.conversation_service.update_conversation_code(
                conversation_id=conversation_id,
                code=processed_result,
                session_id=session_id
            )

        return processed_result