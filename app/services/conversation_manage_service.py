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


class ConversationCodeAlreadyLinkedError(Exception):
    """Raised when a conversation already has a non-GUEST code linked and the caller tries to link a different one."""
    pass


class ConversationService:

    def __init__(self, db: Session = Depends(get_db)):
        """
        Stores the injected database session.

        Parameters:
        - db (Session): SQLAlchemy session — injected by FastAPI via get_db

        Returns:
        - None: sets self.db
        """
        self.db = db

    def get_conversation_unlocked(self, conversation_id: str) -> conversation_models.Conversation | None:
        """
        Looks up a conversation row by id without locking it.

        Parameters:
        - conversation_id (str): the conversation to look up — comes from the caller

        Returns:
        - Conversation | None: the matching row, or None if it doesn't exist — goes to the caller
        """
        return (
            self.db.query(conversation_models.Conversation)
            .filter(conversation_models.Conversation.conversation_id == conversation_id)
            .first()
        )

    def get_conversation_locked(self, conversation_id: str) -> conversation_models.Conversation | None:
        """
        Looks up a conversation row by id and locks it for the rest of the current transaction.

        Parameters:
        - conversation_id (str): the conversation to look up — comes from the caller

        Returns:
        - Conversation | None: the matching row, or None if it doesn't exist — goes to the caller
        """
        return (
            self.db.query(conversation_models.Conversation)
            .filter(conversation_models.Conversation.conversation_id == conversation_id)
            .with_for_update()
            .first()
        )

    def assert_ownership(self, conversation: conversation_models.Conversation, session_id: str) -> None:
        """
        Verifies that a session owns a conversation, raising if it doesn't.

        Parameters:
        - conversation (Conversation): the row to check — comes from the caller
        - session_id (str): the session to check against — comes from the caller

        Returns:
        - None: raises ConversationAccessDeniedError if ownership doesn't match
        """
        if conversation.owner_session_id != session_id:
            raise ConversationAccessDeniedError()

    def _create_conversation(self, code: str | None, session_id: str) -> conversation_models.Conversation:
        """
        Inserts a new conversation row owned by the given session.

        Parameters:
        - code (str | None): invite code to store, defaults to "GUEST" — comes from the caller
        - session_id (str): the owning session — comes from the caller

        Returns:
        - Conversation: the newly created row — goes to append_message
        """
        entry = conversation_models.Conversation(
            conversation_id=str(uuid.uuid4()),
            code=code or "GUEST",
            owner_session_id=session_id
        )
        self.db.add(entry)
        self.db.flush()
        return entry

    async def append_message(self, conversation_id: str | None, code: str | None, session_id: str, sender: str, text: str, selected_scenario: str | None = None, selected_document: str | None = None) -> tuple[conversation_models.Message, str]:
        """
        Inserts a new message row, creating the conversation first if conversation_id is None, or verifying ownership if it isn't.

        Parameters:
        - conversation_id (str | None): conversation to append to, or None to create one — comes from the router
        - code (str | None): invite code for a newly created conversation — comes from the router
        - session_id (str): the caller's session — comes from the router
        - sender (str): "user" or "backend" — comes from the router
        - text (str): the message content — comes from the router
        - selected_scenario (str | None): scenario topics used to generate this message, backend messages only — comes from the router
        - selected_document (str | None): document topics used to generate this message, backend messages only — comes from the router

        Returns:
        - tuple[Message, str]: the created message and its conversation's id — goes to the router
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
            text=text,
            selected_scenario=selected_scenario,
            selected_document=selected_document
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message, conversation.conversation_id

    async def get_recent_messages(self, conversation_id: str, exclude_last: bool = True) -> list[conversation_models.Message]:
        """
        Fetches messages after the conversation's last_summarized_index checkpoint, excluding sender="error" and sender="regen" rows.

        Parameters:
        - conversation_id (str): the conversation to fetch messages for — comes from the caller
        - exclude_last (bool): drop the most recently persisted message, defaults to True — comes from the caller

        Returns:
        - list[Message]: non-error, non-regen messages after last_summarized_index, oldest first — goes to the caller (e.g. ModelCollaborateService, SummarizationService). sender="error" rows (logged when model_orchestration fails, or when ResponseGate.check exhausts its retries) and sender="regen" rows (rejected intermediate attempts logged by ResponseGate.check) are always excluded, both from live prompt history and from summarization, so a failed or discarded turn never shows up as a fake prior reply.
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        recent_messages = (
            self.db.query(conversation_models.Message)
            .filter(
                conversation_models.Message.conversation_id == conversation_id,
                conversation_models.Message.order_index > conversation.last_summarized_index,
                conversation_models.Message.sender.notin_(["error", "regen"]))
            .order_by(conversation_models.Message.order_index.asc())
            .all())

        if exclude_last and recent_messages:
            recent_messages = recent_messages[:-1]

        return recent_messages

    async def mark_summarized_up_to(self, conversation_id: str, order_index: int, summary_text: str | None = None) -> None:
        """
        Advances the conversation's last_summarized_index checkpoint and stores an updated summary.

        Parameters:
        - conversation_id (str): the conversation to update — comes from SummarizationService
        - order_index (int): the new checkpoint value — comes from SummarizationService
        - summary_text (str | None): the updated summary text — comes from SummarizationService

        Returns:
        - None: updates the conversation row in the database
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            return
        if order_index <= conversation.last_summarized_index:
            # Drop the response as there are a newer version summary in db already
            return
        conversation.last_summarized_index = order_index
        if summary_text is not None:
            conversation.summary = summary_text
        self.db.commit()

    async def update_conversation_code(self, conversation_id: str, code: str, session_id: str) -> None:
        """
        Updates a conversation's invite code after verifying the session owns it and that it isn't already linked to a different code.

        Parameters:
        - conversation_id (str): the conversation to update — comes from CodeService.match_code
        - code (str): the verified invite code — comes from CodeService.match_code
        - session_id (str): the session to verify ownership against — comes from CodeService.match_code

        Returns:
        - None: updates the conversation row in the database
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            return
        if conversation.owner_session_id != session_id:
            raise ConversationAccessDeniedError()
        if conversation.code and conversation.code != "GUEST" and conversation.code != code:
            # Already upgraded with a different code — re-submitting the
            # SAME code is a harmless no-op below, but switching to a
            # different one would silently swap which invite code this
            # conversation is billed/attributed to.
            raise ConversationCodeAlreadyLinkedError()
        conversation.code = code
        self.db.commit()