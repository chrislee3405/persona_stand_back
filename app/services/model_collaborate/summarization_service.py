import logging

from fastapi import Depends

from app.constants import SUMMARY_MODEL
from app.database import SessionLocal
from app.services.conversation_manage_service import ConversationService
from app.services.ai.gemini_service import GeminiService
from app.services.model_collaborate.prepare_history import prepare_history

logger = logging.getLogger(__name__)

RECENT_MESSAGES_BEFORE_SUMMARIZE = 10

# The summary is fed straight back into the GENERATION prompt as the persona's
# own conversation history, so how it refers to the two participants matters:
# a summary that calls the persona "the AI" has it reading a description of
# itself as an AI on every later turn, which fights the never-break-character
# rule. It is also a fabrication vector in its own right -- anything invented
# while compressing becomes indistinguishable from something actually said.
_SYSTEM_PROMPT = (
    "You are summarizing a job interview between an interviewer and a candidate. "
    "Write an updated summary from the past summary plus the recent messages in "
    "the user message below.\n"
    "- Refer to the two participants only as \"the interviewer\" and \"the "
    "candidate\". Never call either of them an AI, a model, a bot, or an "
    "assistant, whatever the transcript labels say.\n"
    "- Record only what was actually said. Do not add, infer, or round any "
    "detail that is not in the transcript -- especially dates, numbers, "
    "durations and counts.\n"
    "- Keep straight which side said what, and note where the candidate "
    "declined to answer rather than dropping it.\n"
    "Respond only with the updated summary text, with no preamble or explanation."
)

_USER_PROMPT_TEMPLATE = (
    "Past conversation summary:\n{summary_section}\n\n"
    "Recent conversation messages since that summary:\n{messages_section}"
)


class SummarizationService:
    """
    Folds a conversation's recent messages into its running summary, as a
    background task scheduled after the chat response has already been sent.

    OWNS ITS SESSIONS, and opens them one short transaction at a time. Two
    separate reasons, both of which used to be bugs:

    1. It cannot use the request's session. FastAPI runs a yield-dependency's
       exit code BEFORE the response's background tasks, so by the time this
       runs, `get_db` has already closed the session it was handed. It
       happened to keep working because a closed AsyncSession checks out a
       fresh connection on next use -- which is working by accident.

    2. A summarization run spans a model call that takes seconds. It used to
       hold a `SELECT ... FOR UPDATE` on the conversation row across that whole
       call, because its read went through get_recent_messages while that still
       locked. The visitor's NEXT message then blocked on append_message until
       the summary came back -- a multi-second stall on the turn after every
       tenth message, with nothing to explain it. The read now finishes and
       releases its transaction before Gemini is called, and only the write at
       the end takes a lock, for as long as one UPDATE takes.

    Reading the rows in one session and using them after it closes is safe
    because SessionLocal sets expire_on_commit=False: the instances detach with
    their loaded columns intact, and only `.sender`, `.text` and `.order_index`
    are read afterwards.
    """

    def __init__(self, gemini_service: GeminiService = Depends()):
        """
        Stores the injected Gemini service.

        Parameters:
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI

        Returns:
        - None: sets self.gemini_service. No database session and no ConversationService are injected: this runs after the request's session is gone, so it opens its own — see the class docstring.
        """
        self.gemini_service = gemini_service

    async def summarize_conversation_if_needed(self, conversation_id: str) -> None:
        """
        Fetches the current summary and recent messages for a conversation and checks whether it needs summarizing.

        Parameters:
        - conversation_id (str): the conversation to check — comes from ChatService, scheduled as a background task

        Returns:
        - None: summarizes and stores an updated summary once the conversation has at least
          RECENT_MESSAGES_BEFORE_SUMMARIZE unsummarized messages; otherwise returns without doing anything.

        NEVER RAISES. Starlette awaits background tasks inside the response's
        own __call__, after the body has already been sent, so an exception
        here propagates up the middleware stack with the response already
        committed -- surfacing as "Exception in ASGI application" and, on some
        servers, an abruptly closed connection. The visitor's reply has
        already been delivered and persisted by this point; a failed summary
        must not disturb it. It is logged and dropped, and the next turn past
        the threshold tries again.
        """
        try:
            await self._summarize_conversation_if_needed(conversation_id)
        except Exception:
            logger.exception(
                "summarization failed for conversation_id=%s -- the reply was already sent and persisted, "
                "and the checkpoint is unchanged, so the next turn past the threshold will retry.",
                conversation_id,
            )

    async def _summarize_conversation_if_needed(self, conversation_id: str) -> None:
        """
        The body of summarize_conversation_if_needed, split out so the caller above can be a bare try/except.

        Parameters:
        - conversation_id (str): the conversation to check — comes from summarize_conversation_if_needed

        Returns:
        - None: see summarize_conversation_if_needed
        """
        # --- Read, in its own short-lived session --------------------------
        async with SessionLocal() as read_session:
            conversation_service = ConversationService(db=read_session)
            conversation = await conversation_service.get_conversation_unlocked(conversation_id)
            if conversation is None:
                return
            summary = conversation.summary
            recent_messages = await conversation_service.get_recent_messages(
                conversation_id, exclude_last=False
            )

        # The threshold check lives here rather than in a method of its own --
        # it is one comparison with a single caller.
        if len(recent_messages) < RECENT_MESSAGES_BEFORE_SUMMARIZE:
            return

        await self.summarize_conversation(
            conversation_id=conversation_id,
            summary=summary,
            recent_messages=recent_messages
        )

    async def summarize_conversation(self, conversation_id: str, summary: str | None, recent_messages: list) -> None:
        """
        Generates an updated summary via Gemini and persists it as the conversation's new checkpoint.

        Parameters:
        - conversation_id (str): the conversation to summarize — comes from _summarize_conversation_if_needed
        - summary (str | None): the prior summary, if any — comes from _summarize_conversation_if_needed
        - recent_messages (list): messages to fold into the summary, already detached from the read session — comes from _summarize_conversation_if_needed

        Returns:
        - None: stores the updated summary and checkpoint via ConversationService.mark_summarized_up_to

        NO DATABASE TRANSACTION IS OPEN while the model call runs. The rows
        were read and released by the caller; the write below opens a fresh,
        short-lived one.
        """
        if not recent_messages:
            return

        summary_section = (
            summary if summary
            else "No prior summary exists — this is the first summarization pass for this conversation."
        )

        # Same renderer as every other prompt in the pipeline (see
        # prepare_history); "model" is this prompt's label for the persona's
        # own turns. The summary is passed separately in the template below,
        # so this call renders the recent messages only.
        messages_section = prepare_history(recent_messages, None, assistant_label="model")

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            summary_section=summary_section,
            messages_section=messages_section
        )

        updated_summary = await self.gemini_service.call_model(
            model_name=SUMMARY_MODEL,
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT
        )

        newest_index = recent_messages[-1].order_index

        # --- Write, in its own short-lived session -------------------------
        # mark_summarized_up_to drops the write if a NEWER checkpoint is
        # already stored, which is what makes it safe for two summarization
        # runs to overlap: without that guard the slower one would finish last
        # and wind both the checkpoint and the summary text backwards.
        async with SessionLocal() as write_session:
            conversation_service = ConversationService(db=write_session)
            await conversation_service.mark_summarized_up_to(
                conversation_id=conversation_id,
                order_index=newest_index,
                summary_text=updated_summary
            )
