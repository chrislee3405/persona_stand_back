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

    def __init__(self, db: Session = Depends(get_db)):
        """
        Store the injected database session onto the ConversationService instance attribute.
        Params: db (Session)
        Returns: None
        """
        self.db = db

    def get_conversation_unlocked(self, conversation_id: str) -> conversation_models.Conversation | None:
        """
        Use the given conversation id to look up the matching row from the database conversation table without locking it.
        Params: conversation_id (str)
        Returns: Conversation | None - the row, or None if it doesn't exist
        """
        return (
            self.db.query(conversation_models.Conversation)
            .filter(conversation_models.Conversation.conversation_id == conversation_id)
            .first()
        )

    def get_conversation_locked(self, conversation_id: str) -> conversation_models.Conversation | None:
        """
        Use the given conversation id to look up and lock the matching row in the database conversation table for the rest of the current transaction.
        Params: conversation_id (str)
        Returns: Conversation | None - the row, or None if it doesn't exist
        """
        return (
            self.db.query(conversation_models.Conversation)
            .filter(conversation_models.Conversation.conversation_id == conversation_id)
            .with_for_update()
            .first()
        )

    def assert_ownership(self, conversation: conversation_models.Conversation, session_id: str) -> None:
        """
        Compare the given browser session id against the owner_session_id field of the given conversation row to confirm ownership, raising an error if they don't match.
        Params: conversation (Conversation), session_id (str)
        Returns: None
        """
        if conversation.owner_session_id != session_id:
            raise ConversationAccessDeniedError()

    def _create_conversation(self, code: str | None, session_id: str) -> conversation_models.Conversation:
        """
        Use the given invite code and browser session id to insert a new row into the database conversation table, owned by that session.
        Params: code (str | None), session_id (str)
        Returns: Conversation - the newly created row
        """
        entry = conversation_models.Conversation(
            conversation_id=str(uuid.uuid4()),
            code=code or "GUEST",
            owner_session_id=session_id
        )
        self.db.add(entry)
        self.db.flush()
        return entry

    async def append_message(self, conversation_id: str | None, code: str | None, session_id: str, sender: str, text: str) -> tuple[conversation_models.Message, str]:
        """
        Use the given sender and text to insert a new row into the database message table, first creating a new row in the database conversation table if conversation_id is None, or verifying the given browser session id owns the existing conversation row in the database if conversation_id is provided.
        Params: conversation_id (str | None), code (str | None), session_id (str), sender (str), text (str)
        Returns: tuple[Message, str] - the created message and its conversation's id
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

    async def get_recent_messages(self, conversation_id: str) -> list[conversation_models.Message]:
        """
        Use the given conversation id to fetch every row from the database message table whose order_index is greater than that conversation row's last_summarized_index checkpoint in the database conversation table.
        Params: conversation_id (str)
        Returns: list[Message] - messages with order_index greater than last_summarized_index, oldest first
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        return (
            self.db.query(conversation_models.Message)
            .filter(
                conversation_models.Message.conversation_id == conversation_id,
                conversation_models.Message.order_index > conversation.last_summarized_index
            )
            .order_by(conversation_models.Message.order_index.asc())
            .all()
        )

    async def mark_summarized_up_to(self, conversation_id: str, order_index: int, summary_text: str | None = None) -> None:
        """
        Use the given order_index and optional summary_text to update the last_summarized_index and summary fields of the matching row in the database conversation table.
        Params: conversation_id (str), order_index (int), summary_text (str | None)
        Returns: None
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            return
        conversation.last_summarized_index = order_index
        if summary_text is not None:
            conversation.summary = summary_text
        self.db.commit()

    async def get_conversation(self, conversation_id: str) -> list[conversation_models.Message]:
        """
        Use the given conversation id to fetch every row from the database message table, regardless of browser session ownership.
        Params: conversation_id (str)
        Returns: list[Message] - all messages, oldest first
        """
        return (
            self.db.query(conversation_models.Message)
            .filter(conversation_models.Message.conversation_id == conversation_id)
            .order_by(conversation_models.Message.order_index.asc())
            .all()
        )

    async def list_conversations_for_session(self, session_id: str) -> list[conversation_models.Conversation]:
        """
        Use the given browser session id to fetch every matching row from the database conversation table.
        Params: session_id (str)
        Returns: list[Conversation] - matching conversations, newest first
        """
        return (
            self.db.query(conversation_models.Conversation)
            .filter(conversation_models.Conversation.owner_session_id == session_id)
            .order_by(conversation_models.Conversation.created_at.desc())
            .all()
        )

    async def get_conversation_for_session(self, conversation_id: str, session_id: str) -> list[conversation_models.Message]:
        """
        Use the given conversation id and browser session id to fetch every row from the database message table after verifying the session owns the matching row in the database conversation table.
        Params: conversation_id (str), session_id (str)
        Returns: list[Message] - all messages, oldest first
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        self.assert_ownership(conversation, session_id)

        return (
            self.db.query(conversation_models.Message)
            .filter(conversation_models.Message.conversation_id == conversation_id)
            .order_by(conversation_models.Message.order_index.asc())
            .all()
        )

    async def update_conversation_code(self, conversation_id: str, code: str, session_id: str) -> None:
        """
        Use the user-verified input code to update the invite code field in the database conversation table after verifying the browser session id owns that row.
        Params: conversation_id (str), code (str), session_id (str)
        Returns: None
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            return
        if conversation.owner_session_id != session_id:
            raise ConversationAccessDeniedError()
        conversation.code = code
        self.db.commit()