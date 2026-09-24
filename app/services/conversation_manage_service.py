import uuid

from fastapi import Depends
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

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

    def __init__(self, db: AsyncSession = Depends(get_db)):
        """
        Stores the injected database session.

        Parameters:
        - db (AsyncSession): SQLAlchemy async session — injected by FastAPI via get_db

        Returns:
        - None: sets self.db
        """
        self.db = db

    async def get_conversation_unlocked(self, conversation_id: str) -> conversation_models.Conversation | None:
        """
        Looks up a conversation row by id without locking it.

        Parameters:
        - conversation_id (str): the conversation to look up — comes from the caller

        Returns:
        - Conversation | None: the matching row, or None if it doesn't exist — goes to the caller
        """
        result = await self.db.execute(
            select(conversation_models.Conversation).where(
                conversation_models.Conversation.conversation_id == conversation_id
            )
        )
        return result.scalar_one_or_none()

    async def get_conversation_locked(self, conversation_id: str) -> conversation_models.Conversation | None:
        """
        Looks up a conversation row by id and locks it for the rest of the current transaction.

        Parameters:
        - conversation_id (str): the conversation to look up — comes from the caller

        Returns:
        - Conversation | None: the matching row, or None if it doesn't exist — goes to the caller

        ONLY for callers that commit promptly. The lock is held until this
        session's transaction ends, so a caller that takes it and then awaits
        something slow pins both a row lock and a pooled connection for that
        whole time. append_message (which needs it, to allocate order_index
        without a race) commits within microseconds; get_recent_messages
        deliberately does NOT use this, because its caller goes on to make
        4-12 Gemini calls before anything commits.
        """
        result = await self.db.execute(
            select(conversation_models.Conversation)
            .where(conversation_models.Conversation.conversation_id == conversation_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

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
        Builds a new conversation row owned by the given session and adds it to the session.

        Parameters:
        - code (str | None): invite code to store, defaults to GUEST_CODE — comes from the caller
        - session_id (str): the owning session — comes from the caller

        Returns:
        - Conversation: the pending row — goes to append_message, which flushes and commits it
        """
        entry = conversation_models.Conversation(
            conversation_id=str(uuid.uuid4()),
            code=code or GUEST_CODE,
            owner_session_id=session_id
        )
        self.db.add(entry)
        return entry

    async def append_message(self, conversation_id: str | None, code: str | None, session_id: str, sender: str, text: str, selected_scenario: str | None = None, selected_document: str | None = None, *, commit: bool = True) -> tuple[conversation_models.Message, str]:
        """
        Inserts a new message row, creating the conversation first if conversation_id is None, or verifying ownership if it isn't.

        Parameters:
        - conversation_id (str | None): conversation to append to, or None to create one — comes from the router
        - code (str | None): invite code for a newly created conversation — comes from the router
        - session_id (str): the caller's session — comes from the router
        - sender (str): one of app.constants.Sender — comes from the caller
        - text (str): the message content — comes from the caller
        - selected_scenario (str | None): scenario topics used to generate this message, backend messages only — comes from the caller
        - selected_document (str | None): document topics used to generate this message, backend messages only — comes from the caller

        Returns:
        - tuple[Message, str]: the created message and its conversation's id — goes to the caller

        Takes the row lock and commits inside one call, so the lock is held
        only for the flush -- long enough to allocate order_index without two
        concurrent appends colliding on the (conversation_id, order_index)
        unique constraint, and no longer.
        """
        if conversation_id:
            conversation = await self.get_conversation_locked(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            self.assert_ownership(conversation, session_id)
        else:
            conversation = self._create_conversation(code, session_id)
            await self.db.flush()

        result = await self.db.execute(
            select(func.coalesce(func.max(conversation_models.Message.order_index), -1)).where(
                conversation_models.Message.conversation_id == conversation.conversation_id
            )
        )
        next_index = result.scalar_one() + 1

        message = conversation_models.Message(
            conversation_id=conversation.conversation_id,
            order_index=next_index,
            sender=sender,
            text=text,
            selected_scenario=selected_scenario,
            selected_document=selected_document
        )
        self.db.add(message)
        await self.db.flush()
        if commit:
            await self.db.commit()
            await self.db.refresh(message)
        return message, conversation.conversation_id

    async def retag_message_sender(self, message: conversation_models.Message, sender: str) -> None:
        """
        Changes an already-persisted message's `sender`, without touching its text.

        Parameters:
        - message (Message): the row to retag — comes from ChatService.handle_chat_turn, which holds the row it just inserted
        - sender (str): the new sender value, one of app.constants.Sender — comes from the caller

        Returns:
        - None: updates the row in the database.

        The one caller (ChatService._handle_failed_turn) retags a stored USER
        message to Sender.UNANSWERED_USER when its turn failed before producing
        a reply -- including when the turn ran past TURN_DEADLINE_SECONDS. The
        row is kept so the owner can see what was asked, and the retag is what
        removes it from the next prompt's history -- get_recent_messages
        filters NON_PROMPT_SENDERS in SQL, and UNANSWERED_USER is in that tuple.
        """
        message.sender = sender
        self.db.add(message)
        await self.db.commit()

    async def get_recent_messages(self, conversation_id: str, exclude_last: bool = True) -> list[conversation_models.Message]:
        """
        Fetches the messages after the conversation's last_summarized_index checkpoint that are genuinely part of the conversation, dropping failed and withheld turns.

        Parameters:
        - conversation_id (str): the conversation to fetch messages for — comes from the caller
        - exclude_last (bool): drop the most recently persisted message, defaults to True — comes from the caller

        Returns:
        - list[Message]: the real conversation after last_summarized_index, oldest first — goes to the caller (e.g. ContextGatherer, SummarizationService). Three kinds of row are removed, so no failed or discarded turn can reappear as if it were a real exchange:
          - sender="error" (a turn that raised, timed out, or exhausted ResponseGate's retries), sender="regen" (rejected intermediate attempts) and sender="not_saved_user" (a visitor message whose turn never produced a reply) are excluded in SQL via NON_PROMPT_SENDERS;
          - sender="system" (the ResponseGate fallback notice: "Sorry, I couldn't put together a suitable reply...") is dropped in Python, TOGETHER WITH the user message it answered. A fallback means the persona never actually replied, so leaving either behind poisons the next prompt -- the notice gets read back as the persona's own words ('You: Sorry, I couldn't...'), and the unanswered question looks like it was already dealt with. Handled here rather than in the SQL filter precisely because the pairing needs the system row to still be visible.

        Reads WITHOUT a row lock, deliberately. This used to take
        `SELECT ... FOR UPDATE` and its caller (ContextGatherer.gather) then
        made up to five sequential Gemini calls before anything committed --
        holding a conversation row lock and a pooled connection for as long as
        the model took. Nothing needs the lock: a session's turns are already
        serialised by RateControlService.turn, and no other session can reach
        this conversation (assert_ownership). The only writer that allocates
        anything order-sensitive is append_message, which takes its own lock.
        """
        conversation = await self.get_conversation_unlocked(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        result = await self.db.execute(
            select(conversation_models.Message)
            .where(
                conversation_models.Message.conversation_id == conversation_id,
                conversation_models.Message.order_index > conversation.last_summarized_index,
                conversation_models.Message.sender.notin_(NON_PROMPT_SENDERS),
            )
            .order_by(conversation_models.Message.order_index.asc())
        )
        recent_messages = list(result.scalars().all())

        recent_messages = self._drop_withheld_turns(recent_messages)

        if exclude_last and recent_messages:
            recent_messages = recent_messages[:-1]

        return recent_messages

    async def get_pending_user_messages(self, conversation_id: str) -> list[conversation_models.Message]:
        """
        Fetches the conversation's user messages that the readiness gate has not dealt with yet.

        Parameters:
        - conversation_id (str): the conversation to read — comes from ContextGatherer.gather

        Returns:
        - list[Message]: Sender.USER rows above the conversation's last_handled_index, oldest first — goes to ContextGatherer.gather, which joins their text into the message the turn actually answers and hands them to ReadinessService

        Sender.USER only. A message retagged Sender.UNANSWERED_USER by a failed
        turn is deliberately NOT pending: nothing answered it, but nothing is
        going to -- the visitor was already told that turn failed. Leaving it
        in would make the next message drag a dead question back into the
        prompt, which is the exact thing the retag exists to prevent.

        Unlocked, for the same reason as get_recent_messages: the caller goes
        on to make several Gemini calls before anything commits, and a
        session's turns are already serialised by the database session-work
        lock, with local pacing provided by RateControlService.turn.
        """
        conversation = await self.get_conversation_unlocked(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        result = await self.db.execute(
            select(conversation_models.Message)
            .where(
                conversation_models.Message.conversation_id == conversation_id,
                conversation_models.Message.order_index > conversation.last_handled_index,
                conversation_models.Message.sender == Sender.USER,
            )
            .order_by(conversation_models.Message.order_index.asc())
        )
        return list(result.scalars().all())

    async def get_latest_user_order_index(self, conversation_id: str) -> int:
        """
        Returns the order_index of the newest user message in a conversation.

        Parameters:
        - conversation_id (str): the conversation to read — comes from ChatService.handle_chat_turn

        Returns:
        - int: the highest Sender.USER order_index, or -1 when the conversation has none — goes to ChatService, which compares it with the index a finished run evaluated to decide whether that run is stale

        Read immediately before publishing a reply, so a message that arrived
        while the models were working is visible here even though the run that
        is about to finish never saw it.
        """
        result = await self.db.execute(
            select(func.coalesce(func.max(conversation_models.Message.order_index), -1)).where(
                conversation_models.Message.conversation_id == conversation_id,
                conversation_models.Message.sender == Sender.USER,
            )
        )
        return result.scalar_one()

    async def mark_handled_up_to(self, conversation_id: str, order_index: int, *, commit: bool = True) -> None:
        """Advance in SQL so an ORM identity-map snapshot can never rewind it."""
        await self.db.execute(
            update(conversation_models.Conversation)
            .where(conversation_models.Conversation.conversation_id == conversation_id)
            .values(last_handled_index=func.greatest(conversation_models.Conversation.last_handled_index, order_index))
            .execution_options(synchronize_session=False)
        )
        if commit:
            await self.db.commit()

    async def publish_turn(self, *, conversation_id: str, session_id: str, code: str | None,
                           evaluated_through: int, decision: str, reply_text: str | None,
                           reply_sender: str, selected_document: str | None = None,
                           selected_scenario: str | None = None) -> str:
        """Publish one evaluated group atomically, rechecking ownership and ordering.

        Model work has finished. End its read transaction, then lock only for the
        short publication transaction. A failed write rolls the whole outcome
        back, leaving the original group recoverable by a later continuation.
        The cursor also makes repeated publication of the same group a no-op.
        """
        await self.db.rollback()
        async with self.db.begin():
            conversation = await self.get_conversation_locked(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            self.assert_ownership(conversation, session_id)
            latest = await self.get_latest_user_order_index(conversation_id)
            if latest > evaluated_through or (
                evaluated_through >= 0 and conversation.last_handled_index >= evaluated_through
            ):
                return "superseded"
            if decision == "wait":
                return decision
            if decision == "respond":
                await self.append_message(
                    conversation_id, code, session_id, reply_sender, reply_text,
                    selected_scenario=selected_scenario, selected_document=selected_document,
                    commit=False,
                )
                if reply_sender == Sender.SYSTEM:
                    await self.db.execute(
                        update(conversation_models.Message).where(
                            conversation_models.Message.conversation_id == conversation_id,
                            conversation_models.Message.sender == Sender.USER,
                            conversation_models.Message.order_index > conversation.last_handled_index,
                            conversation_models.Message.order_index <= evaluated_through,
                        ).values(sender=Sender.UNANSWERED_USER)
                    )
            if evaluated_through >= 0:
                await self.mark_handled_up_to(conversation_id, evaluated_through, commit=False)
        return decision

    async def discard_pending_group(self, conversation_id: str, session_id: str) -> None:
        """Retag and consume a rejected held group in one short transaction."""
        await self.db.rollback()
        async with self.db.begin():
            conversation = await self.get_conversation_locked(conversation_id)
            if conversation is None:
                return
            self.assert_ownership(conversation, session_id)
            pending = await self.get_pending_user_messages(conversation_id)
            if not pending:
                return
            through = pending[-1].order_index
            await self.db.execute(
                update(conversation_models.Message).where(
                    conversation_models.Message.conversation_id == conversation_id,
                    conversation_models.Message.sender == Sender.USER,
                    conversation_models.Message.order_index > conversation.last_handled_index,
                    conversation_models.Message.order_index <= through,
                ).values(sender=Sender.UNANSWERED_USER)
            )
            await self.mark_handled_up_to(conversation_id, through, commit=False)

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
        - None: updates the conversation row in the database, or returns without writing if a newer checkpoint is already stored

        This is the ONLY part of summarization that holds a lock, and it is
        taken here rather than around the whole job: the caller reads its
        messages, ends that transaction, spends several seconds in Gemini, and
        only then calls this. The `order_index <=` guard is what makes that
        safe -- two summarization runs can overlap, and the older one's result
        must not overwrite the newer one's checkpoint or its summary text.
        """
        conversation = await self.get_conversation_locked(conversation_id)
        if conversation is None:
            await self.db.rollback()
            return
        if order_index <= conversation.last_summarized_index:
            # A newer summary is already stored -- drop this one rather than
            # winding the checkpoint backwards. Roll back so the FOR UPDATE
            # lock taken above is released immediately.
            await self.db.rollback()
            return
        conversation.last_summarized_index = order_index
        if summary_text is not None:
            conversation.summary = summary_text
        await self.db.commit()

    async def update_conversation_code(self, conversation_id: str, code: str, session_id: str) -> None:
        """
        Updates a conversation's invite code after verifying the session owns it and that it isn't already linked to a different code.

        Parameters:
        - conversation_id (str): the conversation to update — comes from CodeService.match_code
        - code (str): the verified invite code — comes from CodeService.match_code
        - session_id (str): the session to verify ownership against — comes from CodeService.match_code

        Returns:
        - None: updates the conversation row in the database

        Does NOT commit. The caller (CodeService.match_code) is part of the
        session-rotation transaction in codes_router, which has to move
        conversation ownership and consent records to a new session id and
        link this code as one atomic unit -- so the commit belongs to whoever
        owns that transaction, not here.
        """
        conversation = await self.get_conversation_locked(conversation_id)
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

    async def transfer_ownership(self, old_session_id: str, new_session_id: str) -> int:
        """
        Re-points every conversation owned by one session id at another.

        Parameters:
        - old_session_id (str): the session id being retired — comes from codes_router's session rotation
        - new_session_id (str): the freshly-minted session id — comes from codes_router

        Returns:
        - int: how many conversations moved — goes to codes_router for logging

        Does NOT commit: this is one step of the rotation transaction, which
        must move conversations AND consent records AND link the invite code
        together or not at all. A partial rotation would strand the visitor --
        a new session id owning none of their history, with the old id
        unreachable because the cookie has already been replaced.
        """
        result = await self.db.execute(
            select(conversation_models.Conversation).where(
                conversation_models.Conversation.owner_session_id == old_session_id
            )
        )
        conversations = list(result.scalars().all())
        for conversation in conversations:
            conversation.owner_session_id = new_session_id
        return len(conversations)
