from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import code as code_models
from app.services.conversation_manage_service import ConversationService


class InvalidCodeError(Exception):
    """Raised when a submitted invite code doesn't match any stored code."""
    pass


class CodeService:
    def __init__(self, db: AsyncSession = Depends(get_db), conversation_service: ConversationService = Depends()):
        """
        Stores the injected database session and ConversationService.

        Parameters:
        - db (AsyncSession): SQLAlchemy async session — injected by FastAPI via get_db
        - conversation_service (ConversationService): handles conversation lookups/updates — injected by FastAPI

        Returns:
        - None: sets self.db and self.conversation_service
        """
        self.db = db
        self.conversation_service = conversation_service

    async def match_code(self, input_code: str, conversation_id: str | None, session_id: str) -> str:
        """
        Verifies a submitted invite code and, if tied to a conversation, upgrades that conversation off the guest code.

        Parameters:
        - input_code (str): the code submitted by the client — comes from codes_router.verify_code
        - conversation_id (str | None): conversation to upgrade, if any — comes from codes_router.verify_code
        - session_id (str): the caller's session — comes from codes_router.verify_code

        Returns:
        - str: the matched code — goes back to codes_router.verify_code, then to the client and into the session cookie

        Does NOT commit. codes_router wraps this call, the session-id rotation
        and the ownership transfer in one transaction, so the conversation is
        only linked to the code if the whole rotation succeeds.

        Raises:
        - InvalidCodeError: input_code doesn't match any stored code
        - ConversationAccessDeniedError: caller's session doesn't own conversation_id (propagated from ConversationService)
        - ConversationCodeAlreadyLinkedError: conversation_id is already linked to a different code (propagated from ConversationService)
        """
        if not input_code.strip():
            raise InvalidCodeError(input_code)

        result = await self.db.execute(
            select(code_models.InviteCode).where(code_models.InviteCode.code == input_code)
        )
        matched = result.scalar_one_or_none()

        if matched is None:
            raise InvalidCodeError(input_code)

        processed_result = matched.code

        if conversation_id:
            await self.conversation_service.update_conversation_code(
                conversation_id=conversation_id,
                code=processed_result,
                session_id=session_id
            )

        return processed_result
