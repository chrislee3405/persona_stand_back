import logging

from fastapi import Depends

from app.constants import DEFAULT_MODEL
from app.services.ai.gemini_service import GeminiService

logger = logging.getLogger(__name__)

_SPLIT_RESPONSE_SYSTEM_PROMPT_TEMPLATE = (
    "Role: you split one written response into the separate text messages a "
    "real person would send in a row, instead of one long paragraph.\n\n"
    "Task: break the given response into an ordered list of turns.\n\n"
    "Rules:\n"
    "- Group sentences by semantic relatedness: sentences that develop the "
    "same point stay in the same turn; a new turn starts when the topic or "
    "point genuinely shifts.\n"
    "- Preserve the original wording exactly -- do not paraphrase, "
    "summarize, add, or remove any words. Concatenating all turns in order "
    "must reproduce the original response exactly.\n"
    "- Return at most {max_turns} turns. If natural grouping would need "
    "more, merge the least-related adjacent groups until the count fits.\n"
    "- Never split a single sentence across two turns."
)

_SPLIT_RESPONSE_USER_PROMPT_TEMPLATE = (
    "Original response:\n{response_text}"
)


class ResponseParser:
    def __init__(self, gemini_service: GeminiService = Depends(), min_chars_per_turn: int = 40):
        """
        Stores the injected Gemini service and the turn-length hyperparameter.

        Parameters:
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI, or passed explicitly by ModelCollaborateService
        - min_chars_per_turn (int): minimum characters per display turn, defaults to 40 — comes from ModelCollaborateService's _MIN_CHARS_PER_TURN hyperparameter

        Returns:
        - None: sets self.gemini_service, self.min_chars_per_turn
        """
        self.gemini_service = gemini_service
        self.min_chars_per_turn = min_chars_per_turn

    def parse(self, final_response: str) -> list[str]:
        """
        Splits one response into an ordered list of display turns, grouped by semantic relatedness, to mimic a person sending several texts in a row rather than one long paragraph.

        Parameters:
        - final_response (str): the gate-passed response text to split — comes from ModelCollaborateService.model_orchestration

        Returns:
        - list[str]: the response split into turns (at least one), goes to conversations_router/chat_service for the frontend to reveal sequentially. The full, unsplit final_response is still what gets persisted to the database separately -- this list is display-only.
        """
        max_turns = max(1, len(final_response) // self.min_chars_per_turn)
        if max_turns <= 1:
            return [final_response]

        system_prompt = _SPLIT_RESPONSE_SYSTEM_PROMPT_TEMPLATE.format(max_turns=max_turns)
        user_prompt = _SPLIT_RESPONSE_USER_PROMPT_TEMPLATE.format(response_text=final_response)
        schema = {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        }

        response = self.gemini_service.call_model_structured(
            model_name=DEFAULT_MODEL,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=schema
        )

        if not isinstance(response, list) or not response or not all(isinstance(turn, str) and turn.strip() for turn in response):
            logger.debug("ResponseParser.parse returned %r, expected a non-empty list of strings -- falling back to one turn.", response)
            return [final_response]

        if len(response) > max_turns:
            # Model didn't respect the cap -- merge the excess tail turns
            # together rather than truncating (truncating would silently
            # drop text from what gets displayed).
            response = response[:max_turns - 1] + [" ".join(response[max_turns - 1:])]

        return response
