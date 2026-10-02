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
    "kind, with no indication the interviewer is still composing that thought.\n"
    "- \"wait\"     the group appears to be part of an unfinished thought, "
    "or the interviewer signals more is coming. An explicit request to pause "
    "is not required: an unfinished clause, setup without the actual question, "
    "promised list or missing comparison can be enough. Judge the whole group "
    "with the history. Being able to guess an answer to one part does not make "
    "the group ready when the rest is evidently still coming.\n"
    "- \"no_reply\" the group closes the exchange or asks for nothing at all, "
    "so a reply would be noise: a bare acknowledgement (\"ok\", \"got it\", "
    "\"cool thanks\"), or a sign-off already answered (\"bye\" after the "
    "candidate has said goodbye).\n\n"
    # The client continues held messages after an idle timeout. Permit a
    # short wait for a likely fragment without withholding complete questions.
    "When the message plausibly continues an unfinished thought, prefer a "
    "brief \"wait\" over interrupting with a guessed answer. Do not require "
    "certainty that another message will arrive. A standalone complete request "
    "should still receive \"respond\"; mere brevity, typos, missing punctuation "
    "or non-native grammar alone do not make it a fragment. A greeting "
    "(\"hi\", \"how are you\") is a turn in the conversation and is always "
    "\"respond\" -- never \"no_reply\". A completed follow-up that supplies "
    "the missing part makes the group ready; do not keep waiting because its "
    "earlier messages were fragments. Uncertainty about "
    "the answer or available facts belongs to later stages, not readiness.\n\n"
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
