from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.conversation_manage_service import ConversationService
from app.services.ai.gemini_service import GeminiService

RECENT_MESSAGES_BEFORE_SUMMARIZE = 10

_SYSTEM_PROMPT = (
    "Generate an updated summary of the conversation, based on the past "
    "conversation summary and the recent messages that came after it "
    "provided in the user message below. Respond only with the updated "
    "summary text, with no preamble or explanation."
)

_USER_PROMPT_TEMPLATE = (
    "Past conversation summary:\n{summary_section}\n\n"
    "Recent conversation messages since that summary:\n{messages_section}"
)


class SummarizationService:

    def __init__(self, db: Session = Depends(get_db), conversation_service: ConversationService = Depends(), gemini_service: GeminiService = Depends()):
        """
        Stores the injected database session and service instances.

        Parameters:
        - db (Session): SQLAlchemy session — injected by FastAPI via get_db
        - conversation_service (ConversationService): reads/updates conversation state — injected by FastAPI
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI

        Returns:
        - None: sets self.db, self.conversation_service, self.gemini_service
        """
        self.db = db
        self.conversation_service = conversation_service
        self.gemini_service = gemini_service

    async def summarize_conversation_if_needed(self, conversation_id: str) -> None:
        """
        Fetches the current summary and recent messages for a conversation and checks whether it needs summarizing.

        Parameters:
        - conversation_id (str): the conversation to check — comes from conversations_router, scheduled as a background task

        Returns:
        - None: delegates to summarize_if_needed, which may update the conversation's summary in the database
        """
        conversation = self.conversation_service.get_conversation_unlocked(conversation_id)
        summary = conversation.summary if conversation else None
        recent_messages = await self.conversation_service.get_recent_messages(
            conversation_id, exclude_last=False
        )

        await self.summarize_if_needed(
            conversation_id=conversation_id,
            summary=summary,
            recent_messages=recent_messages
        )

    async def summarize_if_needed(self, conversation_id: str, summary: str | None, recent_messages: list) -> None:
        """
        Triggers summarization once recent messages reach the configured threshold.

        Parameters:
        - conversation_id (str): the conversation being checked — comes from summarize_conversation_if_needed
        - summary (str | None): the current summary — comes from summarize_conversation_if_needed
        - recent_messages (list): messages since the last summary — comes from summarize_conversation_if_needed

        Returns:
        - None: delegates to summarize_conversation when the threshold (RECENT_MESSAGES_BEFORE_SUMMARIZE) is met
        """
        if len(recent_messages) >= RECENT_MESSAGES_BEFORE_SUMMARIZE:
            await self.summarize_conversation(conversation_id, summary, recent_messages)

    async def summarize_conversation(self, conversation_id: str, summary: str | None, recent_messages: list) -> None:
        """
        Generates an updated summary via Gemini and persists it as the conversation's new checkpoint.

        Parameters:
        - conversation_id (str): the conversation to summarize — comes from summarize_if_needed
        - summary (str | None): the prior summary, if any — comes from summarize_if_needed
        - recent_messages (list): messages to fold into the summary — comes from summarize_if_needed

        Returns:
        - None: stores the updated summary and checkpoint via ConversationService.mark_summarized_up_to
        """
        if not recent_messages:
            return

        summary_section = (
            summary if summary
            else "No prior summary exists — this is the first summarization pass for this conversation."
        )

        messages_section = "\n".join(
            f"{'User' if m.sender == 'user' else 'model'}: {m.text}"
            for m in recent_messages
        )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            summary_section=summary_section,
            messages_section=messages_section
        )

        updated_summary = self.gemini_service.call_model(
            model_name="gemini-2.5-flash",
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT
        )

        newest_index = recent_messages[-1].order_index
        await self.conversation_service.mark_summarized_up_to(
            conversation_id=conversation_id,
            order_index=newest_index,
            summary_text=updated_summary
        )