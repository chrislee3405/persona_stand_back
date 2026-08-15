import uuid
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import Depends
from app.database import get_db
from app.models import conversation as conversation_models


class ConversationNotFoundError(Exception):
    """Raised when a client supplies a conversation_id that doesn't exist."""
    pass


class ConversationService:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    def _get_conversation_locked(self, conversation_id: str) -> conversation_models.Conversation | None:
        """
        Looks up an existing conversation and locks the row for the rest of
        this transaction, so two concurrent appends to the same conversation
        can't race on order_index assignment.
        """
        return (
            self.db.query(conversation_models.Conversation)
            .filter(conversation_models.Conversation.conversation_id == conversation_id)
            .with_for_update()
            .first()
        )

    def _create_conversation(self, code: str | None) -> conversation_models.Conversation:
        """Backend is the sole authority on conversation_id generation now."""
        entry = conversation_models.Conversation(
            conversation_id=str(uuid.uuid4()),
            code=code or "GUEST"
        )
        self.db.add(entry)
        self.db.flush()  # visible in this transaction without committing yet
        return entry

    async def append_message(
        self,
        conversation_id: str | None,
        code: str | None,
        sender: str,
        text: str
    ) -> tuple[conversation_models.Message, str]:
        """
        Persists a message. If conversation_id is None, this is treated as
        the first message of a brand new conversation — one gets created
        and its id is returned. If conversation_id is provided but doesn't
        match any row, raises ConversationNotFoundError (this should not
        happen in normal use, since only the backend ever mints ids).

        Returns (message, conversation_id) — conversation_id is echoed back
        explicitly so callers always know which conversation the message
        landed in, whether it was newly created or not.
        """
        if conversation_id:
            conversation = self._get_conversation_locked(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
        else:
            conversation = self._create_conversation(code)

        next_index = (
            self.db.query(
                func.coalesce(func.max(conversation_models.Message.order_index), -1)
            )
            .filter(conversation_models.Message.conversation_id == conversation.conversation_id)
            .scalar()
        ) + 1

        message = conversation_models.Message(
            conversation_id=conversation.conversation_id,
            order_index=next_index,
            sender=sender,
            text=text
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message, conversation.conversation_id

    async def get_conversation(self, conversation_id: str) -> list[conversation_models.Message]:
        """Admin retrieval — always returns messages ordered oldest to newest."""
        return (
            self.db.query(conversation_models.Message)
            .filter(conversation_models.Message.conversation_id == conversation_id)
            .order_by(conversation_models.Message.order_index.asc())
            .all()
        )

    async def update_conversation_code(self, conversation_id: str, code: str) -> None:
        """
        Called when a user verifies an invite code mid-conversation — i.e.
        they already sent some messages as a guest, and the existing
        conversation row's code is currently "GUEST". Updates it in place
        to the verified code, rather than creating a new conversation.
 
        If conversation_id doesn't match any row (e.g. user verifies a
        code before sending any message at all, so no conversation exists
        yet), this is a no-op — there's nothing to update, and that's not
        an error. The code will simply be sent correctly on their first
        message via invitechat once a conversation does get created.
        """
        conversation = self._get_conversation_locked(conversation_id)
        if conversation is None:
            return
        conversation.code = code
        self.db.commit()



    async def summarize_and_append(self, conversation_id: str, code: str | None, pairs: list) -> None:
        conversation = self._get_conversation_locked(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        new_chunk = await self._generate_summary(pairs, previous_summary=conversation.summary)
        conversation.summary = (
            f"{conversation.summary}\n{new_chunk}" if conversation.summary else new_chunk
        )
        self.db.commit()

    async def _generate_summary(self, pairs: list, previous_summary: str | None) -> str:
        """Plug in your actual summarization call here (e.g. Claude API)."""
        transcript = "\n".join(
            f"User: {p['user']}\nAssistant: {p['backend']}" for p in pairs
        )
        # TODO: replace with a real LLM call
        raise NotImplementedError("Wire up your summarization model here")
