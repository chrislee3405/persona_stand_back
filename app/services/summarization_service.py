import logging

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.conversation_manage_service import ConversationService
from app.services.ai.gemini_service import GeminiService

logger = logging.getLogger(__name__)

# Static instruction only — no user-originated content in the system
# channel. The past summary and recent messages are both derived from
# raw user input, so they go in the user prompt instead (see
# _USER_PROMPT_TEMPLATE), consistent with the same split applied in
# model_collarborate_service.generate_dialogue.
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
        Store the injected database session, ConversationService, and GeminiService instances onto the SummarizationService instance attributes.
        Params: db (Session), conversation_service (ConversationService), gemini_service (GeminiService)
        Returns: None
        """
        self.db = db
        self.conversation_service = conversation_service
        self.gemini_service = gemini_service

    async def summarize_conversation(self, conversation_id: str, summary: str | None, recent_messages: list) -> None:
        """
        Use the conversation's existing summary and its recent messages from the database message table to generate an updated summary via Gemini, then write that summary and reset the last_summarized_index checkpoint on the matching row in the database conversation table.
        Params: conversation_id (str), summary (str | None), recent_messages (list)
        Returns: None
        """
        if not recent_messages:
            return

        summary_section = (
            summary if summary
            else "No prior summary exists — this is the first summarization pass for this conversation."
        )

        messages_section = "\n".join(
            f"{'User' if m.sender == 'user' else 'Assistant'}: {m.text}"
            for m in recent_messages
        )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            summary_section=summary_section,
            messages_section=messages_section
        )

        logger.debug("=== Gemini call: summarize_conversation ===")
        logger.debug("System prompt: %s", _SYSTEM_PROMPT)
        logger.debug("User prompt: %s", user_prompt)

        updated_summary = self.gemini_service.call_model(
            model_name="gemini-2.5-flash",
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT
        )

        logger.debug("Model response: %s", updated_summary)
        logger.debug("=== End Gemini call ===")

        newest_index = recent_messages[-1].order_index
        await self.conversation_service.mark_summarized_up_to(
            conversation_id=conversation_id,
            order_index=newest_index,
            summary_text=updated_summary
        )