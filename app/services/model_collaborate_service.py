import logging
from dataclasses import dataclass, field

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat_trace import trace
from app.constants import DEFAULT_MODEL
from app.database import get_db
from app.services.ai.gemini_service import GeminiService
from app.services.bm25_service import BM25Service
from app.services.conversation_manage_service import ConversationService
from app.services.model_collaborate import turn_metrics
from app.services.model_collaborate.context_gatherer import ContextGatherer
from app.services.model_collaborate.grounding_service import GroundingService
from app.services.model_collaborate.prompt_builder import PromptBuilder
from app.services.model_collaborate.readiness_service import RESPOND
from app.services.model_collaborate.response_gate import ResponseGate, is_fallback_response
from app.services.model_collaborate.response_parser import ResponseParser
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


@dataclass
class TurnOutcome:
    """
    What one turn decided and produced.

    Replaces the 4-tuple model_orchestration used to return. A tuple could
    only describe a turn that generated a reply, and two of the three
    decisions now produce no reply at all -- returning None-filled slots for
    those would push the meaning of the turn into the caller's guesswork.

    `pending_messages` are the rows the turn answered -- the whole held
    group, not just the message that triggered it. ChatService needs the rows
    themselves, not only the index, because a withheld reply has to retag
    every one of them.

    `evaluated_through` is the order_index of the newest pending message this
    run actually read. ChatService uses it twice: to refuse to publish a reply
    that a newer message has overtaken, and to advance the handled cursor no
    further than the messages that were genuinely evaluated.
    """

    decision: str
    evaluated_through: int
    reason: str = ""
    pending_messages: list = field(default_factory=list)
    reply_text: str | None = None
    reply_turns: list[str] = field(default_factory=list)
    doc_topics: list[str] = field(default_factory=list)
    scenario_topics: list[str] = field(default_factory=list)


class ModelCollaborateService:
    def __init__(self, db: AsyncSession = Depends(get_db), gemini_service: GeminiService = Depends(), bm25_service: BM25Service = Depends(), conversation_service: ConversationService = Depends()):
        """
        Stores the injected Gemini service (needed directly for _generate_ai_response) and composes the orchestration sub-services from the same injected dependencies.

        Parameters:
        - db (AsyncSession): SQLAlchemy async session — injected by FastAPI via get_db
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI
        - bm25_service (BM25Service): retrieves similar past questions — injected by FastAPI
        - conversation_service (ConversationService): reads/writes conversation state — injected by FastAPI

        Returns:
        - None: sets self.gemini_service and the five composed sub-services (self.context_gatherer, self.grounding_service, self.prompt_builder, self.response_gate, self.response_parser)
        """
        self.gemini_service = gemini_service
        self.context_gatherer = ContextGatherer(db, gemini_service, bm25_service, conversation_service)
        self.grounding_service = GroundingService(gemini_service)
        self.prompt_builder = PromptBuilder()
        self.response_gate = ResponseGate(gemini_service, conversation_service)
        self.response_parser = ResponseParser(gemini_service, min_chars_per_turn=_MIN_CHARS_PER_TURN)

    async def model_orchestration(self, user_message: str, conversation_id: str, session_id: str, tier: RateTier, skip_readiness: bool = False) -> TurnOutcome:
        """
        Gathers context, decides whether to reply at all, and -- if so -- what may truthfully be said and how to say it.

        Parameters:
        - user_message (str): the user's current message — comes from conversations_router (guestchat/invitechat); "" for a continue turn, which answers only what is already held
        - conversation_id (str): the conversation being replied to — comes from conversations_router
        - session_id (str): the caller's session — comes from conversations_router, needed by ResponseGate to persist regen/fallback review rows via ConversationService.append_message
        - tier (RateTier): "guest" or "invite" — comes from ChatService.handle_chat_turn, selects which cap in _REGEN_COUNTER applies
        - skip_readiness (bool): True for ChatService.handle_continue_turn — answer the held group without asking the readiness gate again (see ContextGatherer.gather)

        Returns:
        - TurnOutcome: the decision, how far it evaluated, and (only for "respond") the reply text, its display turns and the selected topics — goes to ChatService.handle_chat_turn, which persists the reply, advances the handled cursor and shapes the response body

        The readiness gate is what decides whether the rest of this pipeline
        runs at all. "wait" and "no_reply" return before grounding, so a turn
        that is not going to answer spends nothing on generation,
        verification or splitting -- which is where the gate's own call pays
        for itself. It is gathered concurrently with topic selection and
        example re-ranking, so it costs no extra round trip either.
        """
        with turn_metrics.turn() as metrics:
            # 1. Gather all necessary references and history, and ask whether
            #    the visitor's pending messages should be answered yet.
            context = await self.context_gatherer.gather(user_message, conversation_id, skip_readiness=skip_readiness)
            readiness = context["readiness"]
            decision = readiness["decision"]
            evaluated_through = context["evaluated_through"]
            turn_metrics.set_decision(decision)

            if decision != RESPOND:
                # Nothing below this line runs: no grounding, no generation,
                # no gate, no split.
                # The reason is the model's own words about the message, so
                # only the decision goes in the ordinary log.
                logger.info(
                    "Readiness gate: %s for conversation_id=%s through order_index=%d | %s",
                    decision, conversation_id, evaluated_through, metrics.format(),
                )
                trace.debug("readiness reason for conversation_id=%s: %s", conversation_id, readiness.get("reason", ""))
                return TurnOutcome(
                    decision=decision,
                    evaluated_through=evaluated_through,
                    reason=readiness.get("reason", ""),
                )

            # The turn answers every pending message, not just the one that
            # triggered it -- see ContextGatherer.gather.
            effective_message = context["effective_user_message"]

            # 2. Stage 1 -- decide what the reference material actually supports.
            #    See prompt_builder for why this is a separate call.
            with turn_metrics.stage("grounding"):
                grounding = await self.grounding_service.ground(effective_message, context)

            # 3. Stage 2 -- write the reply from the approved facts only.
            system_prompt, user_prompt = self.prompt_builder.build_reply(effective_message, context, grounding)
            with turn_metrics.stage("reply"):
                ai_response = await self._generate_reply(system_prompt, user_prompt)

            # 4a. response consistency verification. The gate regenerates against
            #     the STAGE 2 prompts, so a rejected reply is rewritten against the
            #     same approved facts rather than re-grounded from scratch.
            with turn_metrics.stage("gate"):
                final_response = await self.response_gate.check(context, effective_message, ai_response, system_prompt, user_prompt, conversation_id, session_id, regen_counter=_REGEN_COUNTER[tier])

            # 4b. response parsing into several display turns. Skip it for the
            #     response-gate fallback: that's a single system notice (rendered
            #     as one centred bubble, sender "system"), not a persona sending
            #     several texts -- splitting it would just cost an extra Gemini
            #     call to no benefit.
            if is_fallback_response(final_response):
                response_turns = [final_response]
            else:
                with turn_metrics.stage("split"):
                    response_turns = await self.response_parser.parse(final_response)

            logger.info(
                "Readiness gate: respond for conversation_id=%s through order_index=%d | %s",
                conversation_id, evaluated_through, metrics.format(),
            )
            return TurnOutcome(
                decision=RESPOND,
                evaluated_through=evaluated_through,
                reason=readiness.get("reason", ""),
                pending_messages=context["pending_messages"],
                reply_text=final_response,
                reply_turns=response_turns,
                doc_topics=context["doc_topic_list"],
                scenario_topics=context["scenario_topic_list"],
            )

    async def _generate_reply(self, system_prompt: str, user_prompt: str) -> str:
        """
        Stage 2: sends the finalized prompts to Gemini and returns the generated reply.

        Parameters:
        - system_prompt (str): the system instruction — comes from self.prompt_builder.build_reply
        - user_prompt (str): the user prompt — comes from self.prompt_builder.build_reply

        Returns:
        - str: the model's reply text — goes to model_orchestration
        """
        trace.debug("reply (stage 2) system prompt: %s", system_prompt)
        trace.debug("reply (stage 2) user prompt: %s", user_prompt)

        final_response = await self.gemini_service.call_model(
            model_name=DEFAULT_MODEL,
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )

        trace.debug("reply (stage 2) model response: %s", final_response)

        return final_response
