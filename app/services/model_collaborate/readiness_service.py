import logging

from fastapi import Depends

from app.chat_trace import trace
from app.constants import DEFAULT_MODEL
from app.services.ai.gemini_service import GeminiService
from app.services.model_collaborate import turn_metrics
from app.services.model_collaborate.prepare_history import prepare_history

logger = logging.getLogger(__name__)

# The three things a turn can do with the visitor's pending messages.
RESPOND = "respond"
WAIT = "wait"
NO_REPLY = "no_reply"

# What a failed or malformed readiness call falls back to.
#
# FAIL TOWARDS REPLYING, deliberately, and in the opposite direction to
# GroundingService's fallback. There a wrong answer invents a fact, so silence
# is the safe error. Here silence IS the error: "wait" leaves a visitor
# watching a chat that never answers, with no notice and nothing to retry,
# and "no_reply" throws their message away. A needless reply is a worse answer
# to a fragment, not a broken conversation, so an unusable readiness result
# means the turn proceeds exactly as it did before this gate existed.
READINESS_FALLBACK = {"decision": RESPOND, "reason": "readiness check unavailable"}

# The verdicts a continue turn uses in place of asking the model (see
# ChatService.handle_continue_turn). The visitor went quiet after a "wait",
# and that silence answers the question this gate would ask again -- so the
# held group is answered as it stands. With nothing held there is nothing to
# answer, and the turn ends before any model call.
CONTINUE_VERDICT = {"decision": RESPOND, "reason": "visitor went quiet after a wait"}
CONTINUE_NOTHING_HELD = {"decision": NO_REPLY, "reason": "continue found nothing held"}

_READINESS_SYSTEM_PROMPT = (
    "You decide whether a job candidate should reply YET to what an "
    "interviewer has just sent in a live chat. You are not writing the "
    "reply, and you are not judging whether the candidate knows the answer.\n\n"
    "You are given the conversation so far and the interviewer's unanswered "
    "message or messages, oldest first. They arrived without a reply in "
    "between, so treat them as one incoming group.\n\n"
    "Choose one `decision`:\n"
    "- \"respond\"  the group is something to answer now: a question, a "
    "request, a statement inviting a response, or a complete thought of any "
    "kind. This is the DEFAULT -- choose it unless one of the two below "
    "clearly applies.\n"
    "- \"wait\"     the interviewer is visibly mid-thought and the group is "
    "not yet answerable: it breaks off unfinished (\"and the other thing I "
    "wanted to ask\"), or explicitly says more is coming (\"one sec\", "
    "\"hold on, let me check something\", \"two questions, first...\"). "
    "Answering now would cut them off.\n"
    "- \"no_reply\" the group closes the exchange or asks for nothing at all, "
    "so a reply would be noise: a bare acknowledgement (\"ok\", \"got it\", "
    "\"cool thanks\"), or a sign-off already answered (\"bye\" after the "
    "candidate has said goodbye).\n\n"
    # Both non-respond decisions are silent from the visitor's side: nothing
    # comes back. That is cheap when they are right and unsettling when they
    # are wrong, so the bar is deliberately asymmetric.
    "WHEN IN DOUBT, CHOOSE \"respond\". The other two answer with silence, "
    "which a visitor cannot tell apart from a broken chat. A greeting "
    "(\"hi\", \"how are you\") is a turn in the conversation and is always "
    "\"respond\" -- never \"no_reply\". A question is always \"respond\", "
    "even if it looks like more might follow.\n\n"
    "Give a short `reason` -- one clause, for the log, not for the visitor."
)

_READINESS_USER_PROMPT_TEMPLATE = (
    "Conversation so far:\n{history_section}\n\n"
    "Unanswered message(s) from the interviewer, oldest first:\n{pending_section}"
)

_READINESS_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {"type": "STRING", "enum": [RESPOND, WAIT, NO_REPLY]},
        "reason": {"type": "STRING"},
    },
    "required": ["decision", "reason"],
}


class ReadinessService:
    def __init__(self, gemini_service: GeminiService = Depends()):
        """
        Stores the injected Gemini service.

        Parameters:
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI, or passed explicitly by ContextGatherer

        Returns:
        - None: sets self.gemini_service. No database session: this stage reads the pending messages it is handed and persists nothing. The cursor those messages come from is advanced by ChatService, once the turn's outcome is known.
        """
        self.gemini_service = gemini_service

    async def check(self, pending_texts: list[str], recent_messages: list | None) -> dict:
        """
        Asks the model whether the visitor's unanswered messages should be replied to yet.

        Parameters:
        - pending_texts (list[str]): the unanswered user messages, oldest first — comes from ContextGatherer.gather, out of ConversationService.get_pending_user_messages
        - recent_messages (list | None): the conversation before those messages — comes from ContextGatherer.gather

        Returns:
        - dict: {"decision": "respond"|"wait"|"no_reply", "reason": str} — goes to ModelCollaborateService.model_orchestration via the gathered context. Returns READINESS_FALLBACK ("respond") when the call fails, raises, or comes back malformed; see that constant for why this fails towards replying.
        """
        if not pending_texts:
            # Nothing to judge. Reached when the cursor and the message rows
            # disagree, which should not happen -- the caller has just saved a
            # message. Answering is the harmless reading.
            logger.warning("Readiness check called with no pending messages -- treating as respond.")
            return dict(READINESS_FALLBACK)

        history_section = (
            prepare_history(recent_messages, None, assistant_label="Candidate")
            if recent_messages
            else "No conversation yet -- this is the opening message."
        )
        pending_section = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(pending_texts))
        user_prompt = _READINESS_USER_PROMPT_TEMPLATE.format(
            history_section=history_section,
            pending_section=pending_section,
        )

        try:
            with turn_metrics.stage("readiness"):
                response = await self.gemini_service.call_model_structured(
                    model_name=DEFAULT_MODEL,
                    user_prompt=user_prompt,
                    system_prompt=_READINESS_SYSTEM_PROMPT,
                    schema=_READINESS_RESPONSE_SCHEMA,
                )
        except Exception:
            logger.warning(
                "Readiness check failed -- replying anyway, as if the gate were not there.",
                exc_info=True,
            )
            return dict(READINESS_FALLBACK)

        if not isinstance(response, dict) or response.get("decision") not in (RESPOND, WAIT, NO_REPLY):
            logger.warning(
                "Readiness check returned no decision in %s -- replying anyway.",
                (RESPOND, WAIT, NO_REPLY),
            )
            trace.debug("readiness malformed response: %r", response)
            return dict(READINESS_FALLBACK)

        reason = response.get("reason")
        return {
            "decision": response["decision"],
            "reason": reason if isinstance(reason, str) else "",
        }
