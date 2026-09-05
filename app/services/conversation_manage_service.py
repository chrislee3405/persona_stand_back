import uuid
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import Depends
from app.constants import GUEST_CODE, NON_PROMPT_SENDERS, Sender
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
        - code (str | None): invite code to store, defaults to GUEST_CODE — comes from the caller
        - session_id (str): the owning session — comes from the caller

        Returns:
        - Conversation: the newly created row — goes to append_message
        """
        entry = conversation_models.Conversation(
            conversation_id=str(uuid.uuid4()),
            code=code or GUEST_CODE,
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
        Fetches the messages after the conversation's last_summarized_index checkpoint that are genuinely part of the conversation, dropping failed and withheld turns.

        Parameters:
        - conversation_id (str): the conversation to fetch messages for — comes from the caller
        - exclude_last (bool): drop the most recently persisted message, defaults to True — comes from the caller

        Returns:
        - list[Message]: the real conversation after last_summarized_index, oldest first — goes to the caller (e.g. ContextGatherer, SummarizationService). Three kinds of row are removed, so no failed or discarded turn can reappear as if it were a real exchange:
          - sender="error" (a turn that raised, or ResponseGate.check exhausting its retries) and sender="regen" (rejected intermediate attempts) are excluded in SQL via NON_PROMPT_SENDERS;
          - sender="system" (the ResponseGate fallback notice: "Sorry, I couldn't put together a suitable reply...") is dropped in Python, TOGETHER WITH the user message it answered. A fallback means the persona never actually replied, so leaving either behind poisons the next prompt -- the notice gets read back as the persona's own words ('You: Sorry, I couldn't...'), and the unanswered question looks like it was already dealt with. Handled here rather than in the SQL filter precisely because the pairing needs the system row to still be visible.
        """
        conversation = self.get_conversation_locked(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        recent_messages = (
            self.db.query(conversation_models.Message)
            .filter(
                conversation_models.Message.conversation_id == conversation_id,
                conversation_models.Message.order_index > conversation.last_summarized_index,
                conversation_models.Message.sender.notin_(NON_PROMPT_SENDERS))
            .order_by(conversation_models.Message.order_index.asc())
            .all())

        recent_messages = self._drop_withheld_turns(recent_messages)

        if exclude_last and recent_messages:
            recent_messages = recent_messages[:-1]

        return recent_messages

    @staticmethod
    def _drop_withheld_turns(messages: list[conversation_models.Message]) -> list[conversation_models.Message]:
        """
        Removes each system notice and the user message it replaced, leaving only turns the persona actually took part in.

        Parameters:
        - messages (list[Message]): rows in order_index order, already free of error/regen rows — comes from get_recent_messages

        Returns:
        - list[Message]: the same list minus every sender="system" row and the sender="user" row immediately before it. Runs in one pass, so consecutive withheld turns are each paired off correctly.
        """
        kept: list[conversation_models.Message] = []
        for message in messages:
            if message.sender == Sender.SYSTEM:
                # The question this notice stood in for never got an answer,
                # so it is not part of the conversation either.
                if kept and kept[-1].sender == Sender.USER:
                    kept.pop()
                continue
            kept.append(message)
        return kept

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
        self.assert_ownership(conversation, session_id)
        if conversation.code and conversation.code != GUEST_CODE and conversation.code != code:
            # Already upgraded with a different code — re-submitting the
            # SAME code is a harmless no-op below, but switching to a
            # different one would silently swap which invite code this
            # conversation is billed/attributed to.
            raise ConversationCodeAlreadyLinkedError()
        conversation.code = code
        self.db.commit()