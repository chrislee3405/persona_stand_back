from fastapi import Depends

from app.constants import SUMMARY_MODEL
from app.services.conversation_manage_service import ConversationService
from app.services.ai.gemini_service import GeminiService
from app.services.model_collarborate.prepare_history import prepare_history

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

    def __init__(self, conversation_service: ConversationService = Depends(), gemini_service: GeminiService = Depends()):
        """
        Stores the injected service instances.

        Parameters:
        - conversation_service (ConversationService): reads/updates conversation state — injected by FastAPI
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI

        Returns:
        - None: sets self.conversation_service, self.gemini_service. No DB session is taken -- every read
          and write goes through conversation_service, which owns its own.
        """
        self.conversation_service = conversation_service
        self.gemini_service = gemini_service

    async def summarize_conversation_if_needed(self, conversation_id: str) -> None:
        """
        Fetches the current summary and recent messages for a conversation and checks whether it needs summarizing.

        Parameters:
        - conversation_id (str): the conversation to check — comes from conversations_router, scheduled as a background task

        Returns:
        - None: summarizes and stores an updated summary once the conversation has at least
          RECENT_MESSAGES_BEFORE_SUMMARIZE unsummarized messages; otherwise returns without doing anything
        """
        conversation = self.conversation_service.get_conversation_unlocked(conversation_id)
        summary = conversation.summary if conversation else None
        recent_messages = await self.conversation_service.get_recent_messages(
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
        - conversation_id (str): the conversation to summarize — comes from summarize_conversation_if_needed
        - summary (str | None): the prior summary, if any — comes from summarize_conversation_if_needed
        - recent_messages (list): messages to fold into the summary — comes from summarize_conversation_if_needed

        Returns:
        - None: stores the updated summary and checkpoint via ConversationService.mark_summarized_up_to
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

        updated_summary = self.gemini_service.call_model(
            model_name=SUMMARY_MODEL,
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT
        )

        newest_index = recent_messages[-1].order_index
        await self.conversation_service.mark_summarized_up_to(
            conversation_id=conversation_id,
            order_index=newest_index,
            summary_text=updated_summary
        )