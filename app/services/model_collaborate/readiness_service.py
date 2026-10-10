import logging

from fastapi import Depends

from app.chat_trace import trace
from app.services.ai.provider_factory import AIService, get_ai_service
from app.services.model_collaborate import turn_metrics
from app.services.model_collaborate.prepare_history import prepare_history
from app.services.model_collaborate.stage_errors import record_failure

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
    "Decide whether the candidate should reply now to the interviewer's unanswered messages. "
    "Judge the whole group, oldest first, using conversation history. Do not write an answer "
    "or assess whether facts are available.\n"
    "\n"
    "Choose decision:\n"
    "- respond: a complete question, request or thought inviting a reply, with no indication "
    "more is coming. Always respond to greetings such as \"hi\" or \"how are you\".\n"
    "- wait: a plausible unfinished thought, setup, promised list, missing comparison, or "
    "indication more is coming. An explicit pause request is unnecessary; answering one part "
    "is premature if the rest is clearly unfinished.\n"
    "- no_reply: a closing acknowledgement such as \"ok\" or \"cool thanks\", or a goodbye "
    "already answered, where another reply adds nothing.\n"
    "\n"
    "Prefer a brief wait for plausible fragments, but brevity, typos, missing punctuation or "
    "non-native grammar alone do not make a complete request unfinished. Once later messages "
    "complete the group, respond. Uncertainty about the answer belongs to later stages.\n"
    "Give a one-clause reason for the log."
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
    def __init__(self, ai_service: AIService = Depends(get_ai_service)):
        """
        Stores the injected AI service.

        Parameters:
        - ai_service (AIService): calls the selected model — injected by FastAPI, or passed explicitly by ContextGatherer

        Returns:
        - None: sets self.ai_service. No database session: this stage reads the pending messages it is handed and persists nothing. The cursor those messages come from is advanced by ChatService, once the turn's outcome is known.
        """
        self.ai_service = ai_service

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
                response = await self.ai_service.call_model_structured(
                    stage="readiness",
                    user_prompt=user_prompt,
                    system_prompt=_READINESS_SYSTEM_PROMPT,
                    schema=_READINESS_RESPONSE_SCHEMA,
                )
        except Exception:
            record_failure("readiness", "call_failed", "respond")
            return dict(READINESS_FALLBACK)

        if not isinstance(response, dict) or response.get("decision") not in (RESPOND, WAIT, NO_REPLY):
            record_failure("readiness", "invalid_output", "respond")
            trace.debug("readiness malformed response: %r", response)
            return dict(READINESS_FALLBACK)

        reason = response.get("reason")
        return {
            "decision": response["decision"],
            "reason": reason if isinstance(reason, str) else "",
        }
