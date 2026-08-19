import uuid
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import Depends
from app.database import get_db
from app.models import conversation as conversation_models


class ConversationNotFoundError(Exception):
    """Raised when a client supplies a conversation_id that doesn't exist."""
    pass


class ConversationAccessDeniedError(Exception):
    """Raised when the caller's session doesn't own the conversation."""
    pass


class ConversationService:
    """
    Owns conversation/message storage only — creating conversations,
    appending messages, retrieving history, updating metadata like the
    invite code. No AI logic lives here.
    """

    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    def get_conversation_locked(self, conversation_id: str) -> conversation_models.Conversation | None:
        """
        Looks up an existing conversation and locks the row for the rest of
        this transaction, so two concurrent appends to the same conversation
        can't race on order_index assignment. Public — other services
        (e.g. SummarizationService) reuse this rather than duplicating it.
        """
        return (
            self.db.query(conversation_models.Conversation)
            .filter(conversation_models.Conversation.conversation_id == conversation_id)
            .with_for_update()
            .first()
        )

    def assert_ownership(
        self,
        conversation: conversation_models.Conversation,
        session_id: str
    ) -> None:
        """
        Ownership is strictly per-session, for both guest and code-owned
        conversations: only the session that created a conversation may
        read or write it. Verifying an invite code only gates whether a
        session may use /api/invitechat at all (checked at the router
        level) — it no longer grants access to a conversation created by a
        *different* session, even one that verified the exact same code.

        Practical consequence: an invite-code user who switches browsers,
        clears cookies, or lets their session cookie expire loses access
        to their prior conversations, same as a guest would. If you want
        that back later, reintroduce a branch here that matches on
        conversation.code against the session's verified_code instead.
        """
        if conversation.owner_session_id != session_id:
            raise ConversationAccessDeniedError()

    def _create_conversation(self, code: str | None, session_id: str) -> conversation_models.Conversation:
        """Backend is the sole authority on conversation_id generation."""
        entry = conversation_models.Conversation(
            conversation_id=str(uuid.uuid4()),
            code=code or "GUEST",
            owner_session_id=session_id  # recorded even for code-owned convos, for audit/creator tracking
        )
        self.db.add(entry)
        self.db.flush()  # visible in this transaction without committing yet
        return entry

    async def append_message(
        self,
        conversation_id: str | None,
        code: str | None,
        session_id: str,
        sender: str,
        text: str
    ) -> tuple[conversation_models.Message, str]:
        """
        Persists a message. If conversation_id is None, this is treated as
        the first message of a brand new conversation — one gets created
        and its id is returned. If conversation_id is provided, it must
        both exist AND belong to this exact session (see assert_ownership)
        — otherwise this raises rather than silently reading/writing
        someone else's conversation, guest or code-owned alike.
        """
        if conversation_id:
            conversation = self.get_conversation_locked(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            self.assert_ownership(conversation, session_id)
        else:
            conversation = self._create_conversation(code, session_id)

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
        """
        Admin retrieval — always returns messages ordered oldest to newest.
        NOT gated by session ownership by design (admins need to read any
        conversation). Do not expose this on a router without separate
        admin authentication/authorization in front of it.
        """
        return (
            self.db.query(conversation_models.Message)
            .filter(conversation_models.Message.conversation_id == conversation_id)
            .order_by(conversation_models.Message.order_index.asc())
            .all()
        )

    async def update_conversation_code(
        self,
        conversation_id: str,
        code: str,
        session_id: str
    ) -> None:
        """
        Called when a user verifies an invite code mid-conversation. Only
        the session that created the (guest) conversation may upgrade it —
        otherwise anyone who verifies any valid code could hijack an
        arbitrary conversation_id belonging to someone else, since
        conversation_id alone was never proof of anything.
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            return
        if conversation.owner_session_id != session_id:
            raise ConversationAccessDeniedError()
        conversation.code = code
        self.db.commit()