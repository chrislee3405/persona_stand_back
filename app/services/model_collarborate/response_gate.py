import logging

from fastapi import Depends

from app.services.ai.gemini_service import GeminiService
from app.services.conversation_manage_service import ConversationService
from app.services.model_collarborate.prepare_history import prepare_history

logger = logging.getLogger(__name__)

_VERIFY_NATURAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE = (
    "Role: you are an AI auditor verifying one exchange between a human "
    "interviewer and an AI model role-playing a given interviewee persona.\n\n"
    "Task: check whether the AI response breaks any of the rules below.\n\n"
    "Rules:\n"
    "- The AI response must never break character, reveal that it is an AI model, or refer to itself as an assistant.\n"
    "- The AI response should read as natural, from a normal person's perspective.\n"
    "- The AI response should not ask the user to input any credential or private information.\n"
    "- The AI response should not make any promise to the user.\n"
    "- The AI response should only use the persona's actual punctuation habits: periods, commas, question marks, quotation marks only when essential, and $/% for money or percentages. Reject any other symbols, emoji, or markdown-style formatting (e.g. **, <>, [], ;, !, ...).\n"
    "- The AI response should not contradict facts the persona already stated earlier in this conversation.\n\n"
    "If the response breaks no rule, return \"<pass>\" as result, \"<pass>\" as reason, and \"<pass>\" as quote too.\n\n"
    "If the response breaks a rule, return the reject type as result:\n"
    "- \"<reject-minor>\": another AI could correct the response given only the reject reason and the original response.\n"
    "- \"<reject-major>\": another AI needs the reject reason plus the whole background information again, to regenerate the response with a different approach.\n\n"
    "In both reject cases, give a specific explanation of what broke, as reason -- and as quote, copy the "
    "exact substring of the AI response that demonstrates the violation, character-for-character, not a "
    "paraphrase or summary. If you cannot point to an exact substring that actually demonstrates the "
    "violation, return \"<pass>\" for all three fields instead of guessing."
)

_VERIFY_NATURAL_RESPONSE_USER_PROMPT_TEMPLATE = (
    "{recent_conversation_part}"
    "Current user message: {user_message}\n"
    "Current AI message: {ai_response}\n"
)


class ResponseGate:
    def __init__(self, gemini_service: GeminiService = Depends(), conversation_service: ConversationService = Depends(), regen_counter: int = 3):
        """
        Stores the injected service instances and the regen-attempt hyperparameter.

        Parameters:
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI, or passed explicitly by ModelCollaborateService
        - conversation_service (ConversationService): persists regen/fallback review rows — injected by FastAPI, or passed explicitly
        - regen_counter (int): max verify-then-regenerate attempts, defaults to 3 — comes from ModelCollaborateService's _REGEN_COUNTER hyperparameter

        Returns:
        - None: sets self.gemini_service, self.conversation_service, self.regen_counter
        """
        self.gemini_service = gemini_service
        self.conversation_service = conversation_service
        self.regen_counter = regen_counter

    def _verify_response(self, recent_conversation_part: str, user_message: str, ai_response: str) -> tuple[str, str, str]:
        """
        Asks Gemini to audit one candidate response against the natural-response rules.

        Parameters:
        - recent_conversation_part (str): formatted recent-history block, or "" if none — comes from check
        - user_message (str): the interviewer's current message — comes from check
        - ai_response (str): the candidate response being audited — comes from check (either the original reply or a regenerated one)

        Returns:
        - tuple[str, str, str]: (result_tag, reason, quote), result_tag is one of "<pass>" / "<reject-minor>" / "<reject-major>", quote is the exact substring of ai_response the model claims demonstrates the violation (or "<pass>") — goes to check, which cross-checks quote against ai_response before trusting a reject verdict
        """
        user_prompt = _VERIFY_NATURAL_RESPONSE_USER_PROMPT_TEMPLATE.format(
            recent_conversation_part=recent_conversation_part,
            user_message=user_message,
            ai_response=ai_response
        )
        schema = {
            "type": "OBJECT",
            "properties": {
                "result": {
                    "type": "STRING",
                    "enum": ["<pass>", "<reject-minor>", "<reject-major>"]
                },
                "reason": {
                    "type": "STRING"
                },
                "quote": {
                    "type": "STRING"
                }
            },
            "required": ["result", "reason", "quote"]
        }

        check_result = self.gemini_service.call_model_structured(
            model_name="gemini-3.5-flash-lite",
            user_prompt=user_prompt,
            system_prompt=_VERIFY_NATURAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE,
            schema=schema
        )
        return check_result["result"], check_result["reason"], check_result["quote"]

    async def check(self, context: dict, user_message: str, ai_response: str, system_prompt: str, user_prompt: str, conversation_id: str, session_id: str) -> str:
        """
        Verifies the candidate response against the natural-response rules, regenerating up to self.regen_counter times if rejected.

        Parameters:
        - context (dict): gathered context materials, used for recent_messages — comes from ModelCollaborateService.model_orchestration
        - user_message (str): the interviewer's current message — comes from model_orchestration
        - ai_response (str): the initially generated candidate response — comes from model_orchestration
        - system_prompt (str): the original reply-generation system prompt (role/task/personality) — comes from model_orchestration, reused unmodified as the base for major-reject regeneration
        - user_prompt (str): the original reply-generation user prompt (reference material/history/message) — comes from model_orchestration, reused unmodified for every regeneration attempt
        - conversation_id (str): the conversation being replied to — comes from model_orchestration, used to persist regen/fallback review rows
        - session_id (str): the caller's session — comes from model_orchestration, used to persist regen/fallback review rows (ownership already verified earlier in the same request; conversation_id is always already resolved here, so `code` is irrelevant and passed as None)

        Returns:
        - str: a response that passed verification, or a generic fallback if it's still rejected after self.regen_counter attempts — goes to model_orchestration
        """
        recent_conversation_part = ""
        if context["recent_messages"]:
            recent_conversation_part = "Past few conversation:\n" + prepare_history(context["recent_messages"], None)

        current_response = ai_response
        # Accumulates every attempt's reject reason so later attempts see the
        # full history, not just the immediately preceding one -- otherwise
        # attempt 3's regen prompt would only know about attempt 2's reason
        # and could reintroduce whatever attempt 1 already got rejected for.
        reject_history: list[str] = []
        for attempt in range(self.regen_counter):
            result, reason, quote = self._verify_response(recent_conversation_part, user_message, current_response)

            if result == "<pass>":
                return current_response

            # The model must ground a reject verdict in an exact substring of
            # the response it's judging (see the system prompt's quote
            # instruction). If it can't -- an empty/placeholder quote, or one
            # that doesn't actually appear in current_response -- the verdict
            # itself is untrustworthy (e.g. rejecting for a semicolon or
            # ampersand that was never actually in the text) rather than the
            # response being genuinely at fault, so treat it as a pass
            # instead of burning a regen attempt correcting a problem that
            # doesn't exist.
            is_grounded = bool(quote) and quote != "<pass>" and quote in current_response
            if not is_grounded:
                logger.warning(
                    "ResponseGate.check attempt %d/%d: rejection (%s: %s) quoted %r, which isn't in the response -- treating as ungrounded and passing it through.",
                    attempt + 1, self.regen_counter, result, reason, quote
                )
                return current_response

            logger.warning(
                "ResponseGate.check attempt %d/%d rejected (%s): %s",
                attempt + 1, self.regen_counter, result, reason
            )

            reject_history.append(f"Attempt {attempt + 1} ({result}): {reason}")
            reject_history_text = "\n".join(reject_history)

            # Keep the rejected attempt for later review -- excluded from
            # live prompt history and summarization via the sender="regen"
            # filter in ConversationService.get_recent_messages.
            await self.conversation_service.append_message(
                conversation_id=conversation_id,
                code=None,
                session_id=session_id,
                sender="regen",
                text=f"[Attempt {attempt + 1}/{self.regen_counter}, {result}: {reason}]\n\n{current_response}"
            )

            # No point regenerating on the last attempt -- nothing left in
            # this loop would ever verify it, so it would just be discarded
            # unchecked (and previously was, at the cost of a wasted Gemini
            # call and a misleading "last rejected response" in the fallback
            # log below, since that response was never actually verified).
            if attempt == self.regen_counter - 1:
                break

            if result == "<reject-minor>":
                regen_system_prompt = (
                    f"The response has been rejected across these attempts so far:\n{reject_history_text}\n\n"
                    f"Most recent response:\n{current_response}\n\n"
                    "Regenerate a corrected response that fixes all of these problems at once, "
                    "changing as little else as possible."
                )
            else:  # <reject-major>
                regen_system_prompt = (
                    f"{system_prompt}\n\n"
                    f"Your previous responses were rejected across these attempts so far:\n{reject_history_text}\n\n"
                    f"Most recent response:\n{current_response}\n\n"
                    "Regenerate a new response using a different approach that avoids all of these problems."
                )

            current_response = self.gemini_service.call_model(
                model_name="gemini-3.5-flash-lite",
                user_prompt=user_prompt,
                system_prompt=regen_system_prompt
            )

        logger.warning(
            "ResponseGate.check exhausted %d attempts, still failing verification -- returning fallback response.",
            self.regen_counter
        )
        fallback_response = "Sorry, I'm having trouble forming a suitable response right now. Please try rephrasing your message."
        await self.conversation_service.append_message(
            conversation_id=conversation_id,
            code=None,
            session_id=session_id,
            sender="error",
            text=f"[ResponseGate.check exhausted {self.regen_counter} attempts]\n{chr(10).join(reject_history)}\n\nLast rejected response:\n\n{current_response}"
        )
        return fallback_response
