import logging

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.ai.gemini_service import GeminiService
from app.services.bm25_service import BM25Service
from app.services.conversation_manage_service import ConversationService
from app.services.model_collarborate.context_gatherer import ContextGatherer
from app.services.model_collarborate.prompt_builder import PromptBuilder
from app.services.model_collarborate.response_gate import ResponseGate, FALLBACK_RESPONSES
from app.services.model_collarborate.response_parser import ResponseParser
from app.services.rate_control_service import RateTier

logger = logging.getLogger(__name__)

# Hyperparameters -- the one place to tune these; passed into the sub-services
# below rather than living next to the code that consumes them.
# Max verify-then-regenerate attempts (see ResponseGate.check), per tier.
# Guest gets fewer attempts than invite -- each attempt is a Gemini call, so
# this is a cost lever in the same spirit as RateTier's other per-tier
# limits in rate_control_service.py, not just a quality one. Resolved by
# tier in model_orchestration and passed into ResponseGate.check() per
# call, rather than baked into ResponseGate at construction, since a
# request's tier isn't known until model_orchestration is called.
_REGEN_COUNTER: dict[RateTier, int] = {
    "guest": 2,
    "invite": 4,
}
# Minimum characters per turn when splitting a response into several message
# bubbles. Larger value -> fewer, longer turns allowed; smaller value ->
# more, shorter turns allowed. Tune during testing.
_MIN_CHARS_PER_TURN = 40


class ModelCollaborateService:
    def __init__(self, db: Session = Depends(get_db), gemini_service: GeminiService = Depends(), bm25_service: BM25Service = Depends(), conversation_service: ConversationService = Depends()):
        """
        Stores the injected Gemini service (needed directly for _generate_ai_response) and composes the orchestration sub-services from the same injected dependencies.

        Parameters:
        - db (Session): SQLAlchemy session — injected by FastAPI via get_db
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI
        - bm25_service (BM25Service): retrieves similar past questions — injected by FastAPI
        - conversation_service (ConversationService): reads/writes conversation state — injected by FastAPI

        Returns:
        - None: sets self.gemini_service and the four composed sub-services (self.context_gatherer, self.prompt_builder, self.response_gate, self.response_parser)
        """
        self.gemini_service = gemini_service
        self.context_gatherer = ContextGatherer(db, gemini_service, bm25_service, conversation_service)
        self.prompt_builder = PromptBuilder()
        self.response_gate = ResponseGate(gemini_service, conversation_service)
        self.response_parser = ResponseParser(gemini_service, min_chars_per_turn=_MIN_CHARS_PER_TURN)

    async def model_orchestration(self, user_message: str, conversation_id: str, session_id: str, tier: RateTier) -> tuple[str, list[str], list[str], list[str]]:
        """
        Gathers context, builds prompts, and generates the AI reply to a user message.

        Parameters:
        - user_message (str): the user's current message — comes from conversations_router (guestchat/invitechat)
        - conversation_id (str): the conversation being replied to — comes from conversations_router
        - session_id (str): the caller's session — comes from conversations_router, needed by ResponseGate to persist regen/fallback review rows via ConversationService.append_message
        - tier (RateTier): "guest" or "invite" — comes from ChatService.handle_chat_turn, selects which cap in _REGEN_COUNTER applies

        Returns:
        - tuple[str, list[str], list[str], list[str]]: full reply text (for DB storage), reply split into display turns (for the frontend), selected doc topics, selected scenario topics — goes to conversations_router, which persists the full reply text and logs the selected topics
        """
        # 1. Gather all necessary references and history
        context = await self.context_gatherer.gather(user_message, conversation_id)

        # 2. Format the system and user prompts
        system_prompt, user_prompt = self.prompt_builder.build(user_message, context)

        # 3. Call the model to generate the response
        ai_response = self._generate_ai_response(system_prompt, user_prompt)

        # 4a. response consistency verification
        final_response = await self.response_gate.check(context, user_message, ai_response, system_prompt, user_prompt, conversation_id, session_id, regen_counter=_REGEN_COUNTER[tier])

        # 4b. response parsing into several display turns. Skip it for the
        #     response-gate fallback: that's a single system notice (rendered
        #     as one centred bubble, sender "system"), not a persona sending
        #     several texts -- splitting it would just cost an extra Gemini
        #     call to no benefit.
        if final_response in FALLBACK_RESPONSES:
            response_turns = [final_response]
        else:
            response_turns = self.response_parser.parse(final_response)

        return final_response, response_turns, context["doc_topic_list"], context["scenario_topic_list"]

    def _generate_ai_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends the finalized prompts to Gemini and returns the generated reply.

        Parameters:
        - system_prompt (str): the system instruction — comes from self.prompt_builder.build
        - user_prompt (str): the user prompt — comes from self.prompt_builder.build

        Returns:
        - str: the model's reply text — goes to model_orchestration
        """
        logger.debug("=== Gemini call: model_orchestration ===")
        logger.debug("System prompt: %s", system_prompt)
        logger.debug("User prompt: %s", user_prompt)

        final_response = self.gemini_service.call_model(
            model_name="gemini-3.5-flash-lite",
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )

        logger.debug("Model response: %s", final_response)
        logger.debug("=== End Gemini call ===")

        return final_response
