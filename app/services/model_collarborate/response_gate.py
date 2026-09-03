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
    "Rules (each tagged with its category):\n"
    "- [broke_character] The AI response must never break character, reveal that it is an AI model, or refer to itself as an assistant.\n"
    "- [unnatural] The AI response should read as natural, from a normal person's perspective.\n"
    "- [asked_for_private_info] The AI response should not ask the user to input any credential or private information.\n"
    "- [made_a_promise] The AI response should not make any promise to the user.\n"
    "- [punctuation] The AI response should use ordinary human punctuation matching the persona's habits (periods, commas, question marks, apostrophes, quotation marks, $/%), not tells of AI writing: no emoji, markdown or markup (**, [], <>, backticks, #), bullet or numbered lists, section labels, semicolons in casual chat, or filler ellipses. An exclamation mark is fine in a short exclamation or greeting (\"Hello!\", \"Congratulations!\") but not inside a longer sentence or paragraph.\n"
    "- [contradiction] The AI response should not contradict facts the persona already stated earlier in this conversation.\n\n"
    "If the response breaks no rule, return \"<pass>\" as result, \"<pass>\" as reason, \"<pass>\" as quote, and \"none\" as category.\n\n"
    "If the response breaks a rule, return the reject type as result:\n"
    "- \"<reject-minor>\": another AI could correct the response given only the reject reason and the original response.\n"
    "- \"<reject-major>\": another AI needs the reject reason plus the whole background information again, to regenerate the response with a different approach.\n\n"
    "Also return category: the tag of the broken rule (broke_character, unnatural, asked_for_private_info, made_a_promise, punctuation, or contradiction). If more than one rule was broken, return the tag of the most serious one.\n\n"
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

# Shown to the user when every regenerate-then-verify attempt still fails
# the natural-response check. The generic default; check() swaps in a
# reason-specific variant from _FALLBACK_BY_CATEGORY below when the last
# rejection has a category worth naming.
FALLBACK_RESPONSE = (
    "Sorry, I'm having trouble forming a suitable response right now. "
    "Please try rephrasing your message."
)

# Category (from the auditor's `category` field) -> a fallback message that
# says WHY the reply was withheld. Categories not listed here (unnatural,
# punctuation) fall back to the generic FALLBACK_RESPONSE -- telling a user
# "my reply had bad punctuation" isn't useful.
_FALLBACK_BY_CATEGORY = {
    "asked_for_private_info": (
        "Sorry, I can't send that reply -- it would have asked you for "
        "personal or private information. Please try rephrasing your message."
    ),
    "made_a_promise": (
        "Sorry, I can't send that reply -- it made a commitment it "
        "shouldn't. Please try rephrasing your message."
    ),
    "broke_character": (
        "Sorry, I couldn't put together a natural in-character reply to "
        "that. Please try rephrasing your message."
    ),
    "contradiction": (
        "Sorry, I couldn't form a reply that stays consistent with what "
        "has already been said. Please try rephrasing your message."
    ),
}

# Every string check() can return as a withheld-reply notice. chat_service
# and model_orchestration test membership here to mark the outgoing message
# "system" (a centred bubble) rather than a "backend" persona turn.
FALLBACK_RESPONSES = frozenset({FALLBACK_RESPONSE, *_FALLBACK_BY_CATEGORY.values()})


class ResponseGate:
    def __init__(self, gemini_service: GeminiService = Depends(), conversation_service: ConversationService = Depends()):
        """
        Stores the injected service instances.

        Parameters:
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI, or passed explicitly by ModelCollaborateService
        - conversation_service (ConversationService): persists regen/fallback review rows — injected by FastAPI, or passed explicitly

        Returns:
        - None: sets self.gemini_service, self.conversation_service
        """
        self.gemini_service = gemini_service
        self.conversation_service = conversation_service

    def _verify_response(self, recent_conversation_part: str, user_message: str, ai_response: str) -> tuple[str, str, str, str]:
        """
        Asks Gemini to audit one candidate response against the natural-response rules.

        Parameters:
        - recent_conversation_part (str): formatted recent-history block, or "" if none — comes from check
        - user_message (str): the interviewer's current message — comes from check
        - ai_response (str): the candidate response being audited — comes from check (either the original reply or a regenerated one)

        Returns:
        - tuple[str, str, str, str]: (result_tag, reason, quote, category). result_tag is one of "<pass>" / "<reject-minor>" / "<reject-major>"; quote is the exact substring of ai_response the model claims demonstrates the violation (or "<pass>"); category is the broken-rule tag ("none" on a pass) used by check to pick a reason-specific fallback message — goes to check, which cross-checks quote against ai_response before trusting a reject verdict
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
                },
                "category": {
                    "type": "STRING",
                    "enum": [
                        "none",
                        "broke_character",
                        "unnatural",
                        "asked_for_private_info",
                        "made_a_promise",
                        "punctuation",
                        "contradiction",
                    ]
                }
            },
            "required": ["result", "reason", "quote", "category"]
        }

        check_result = self.gemini_service.call_model_structured(
            model_name="gemini-3.5-flash-lite",
            user_prompt=user_prompt,
            system_prompt=_VERIFY_NATURAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE,
            schema=schema
        )
        return (
            check_result["result"],
            check_result["reason"],
            check_result["quote"],
            check_result.get("category", "none"),
        )

    async def check(self, context: dict, user_message: str, ai_response: str, system_prompt: str, user_prompt: str, conversation_id: str, session_id: str, regen_counter: int) -> str:
        """
        Verifies the candidate response against the natural-response rules, regenerating up to regen_counter times if rejected.

        Parameters:
        - context (dict): gathered context materials, used for recent_messages — comes from ModelCollaborateService.model_orchestration
        - user_message (str): the interviewer's current message — comes from model_orchestration
        - ai_response (str): the initially generated candidate response — comes from model_orchestration
        - system_prompt (str): the original reply-generation system prompt (role/task/personality) — comes from model_orchestration, reused unmodified as the base for major-reject regeneration
        - user_prompt (str): the original reply-generation user prompt (reference material/history/message) — comes from model_orchestration, reused unmodified for every regeneration attempt
        - conversation_id (str): the conversation being replied to — comes from model_orchestration, used to persist regen/fallback review rows
        - session_id (str): the caller's session — comes from model_orchestration, used to persist regen/fallback review rows (ownership already verified earlier in the same request; conversation_id is always already resolved here, so `code` is irrelevant and passed as None)
        - regen_counter (int): max verify-then-regenerate attempts — comes from model_orchestration, resolved from ModelCollaborateService's per-tier _REGEN_COUNTER hyperparameter (a request's tier isn't known until then, so this isn't fixed at ResponseGate construction)

        Returns:
        - str: a response that passed verification, or a fallback notice if it's still rejected after regen_counter attempts — the fallback is reason-specific when the last rejection has a nameable category (see _FALLBACK_BY_CATEGORY), otherwise the generic FALLBACK_RESPONSE — goes to model_orchestration
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
        # Category of the most recent grounded rejection -- picks the
        # fallback wording if every attempt is exhausted.
        last_category = "none"
        for attempt in range(regen_counter):
            result, reason, quote, category = self._verify_response(recent_conversation_part, user_message, current_response)

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
                    attempt + 1, regen_counter, result, reason, quote
                )
                return current_response

            logger.warning(
                "ResponseGate.check attempt %d/%d rejected (%s, %s): %s",
                attempt + 1, regen_counter, result, category, reason
            )

            last_category = category
            reject_history.append(f"Attempt {attempt + 1} ({result}, {category}): {reason}")
            reject_history_text = "\n".join(reject_history)

            # Keep the rejected attempt for later review -- excluded from
            # live prompt history and summarization via the sender="regen"
            # filter in ConversationService.get_recent_messages.
            await self.conversation_service.append_message(
                conversation_id=conversation_id,
                code=None,
                session_id=session_id,
                sender="regen",
                text=f"[Attempt {attempt + 1}/{regen_counter}, {result}: {reason}]\n\n{current_response}"
            )

            # No point regenerating on the last attempt -- nothing left in
            # this loop would ever verify it, so it would just be discarded
            # unchecked (and previously was, at the cost of a wasted Gemini
            # call and a misleading "last rejected response" in the fallback
            # log below, since that response was never actually verified).
            if attempt == regen_counter - 1:
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

        fallback_response = _FALLBACK_BY_CATEGORY.get(last_category, FALLBACK_RESPONSE)
        logger.warning(
            "ResponseGate.check exhausted %d attempts (last category: %s) -- returning fallback response.",
            regen_counter, last_category
        )
        await self.conversation_service.append_message(
            conversation_id=conversation_id,
            code=None,
            session_id=session_id,
            sender="error",
            text=f"[ResponseGate.check exhausted {regen_counter} attempts]\n{chr(10).join(reject_history)}\n\nLast rejected response:\n\n{current_response}"
        )
        return fallback_response
