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
    "- [punctuation] Ordinary human punctuation is fine: periods, commas, question marks, apostrophes, quotation marks, colons (including clock times like \"9:07 AM\" and ratios), parentheses, hyphens in compound words (\"hands-on\"), and $/%. Do NOT reject a mark just because it is not on that list. Reject ONLY these AI-writing tells: emoji; markdown or markup (**, [], <>, backticks, #); bullet or numbered lists; a colon used to head a label or section (\"Skills:\", \"Overview:\"); semicolons in casual chat; filler ellipses; em or en dashes (—, –) or double hyphens (--) used as sentence punctuation for an aside, an appositive, or a dramatic pause (e.g. reject \"teamwork—skills I know are huge\"); an exclamation mark inside a longer sentence or paragraph (a lone \"Hello!\" or \"Congratulations!\" is fine); and a closing full stop on a bare greeting or acknowledgement that is the entire reply (\"Hi.\", \"Hey there.\", \"Yeah.\", \"Sure thing.\", \"Thanks.\", \"No worries.\") -- real people drop the period on messages that short, so reject it. This last one does NOT apply to a brief factual answer that happens to be short (\"It's 9:07.\" or \"About three years.\" keep their period), nor to a reply of a full sentence or longer.\n"
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
# the natural-response check. check() appends the reasons it actually hit
# (see _build_fallback), so e.g. "... It kept breaking character and using
# punctuation or formatting that isn't allowed here. ...".
_FALLBACK_LEAD = "Sorry, I couldn't put together a suitable reply."
_FALLBACK_TAIL = "Please try rephrasing your message."

# One broken-rule category -> a short phrase for the user-facing fallback.
_REJECT_PHRASE = {
    "broke_character": "breaking character",
    "unnatural": "sounding unnatural",
    "asked_for_private_info": "asking you for personal information",
    "made_a_promise": "making a promise it shouldn't",
    "punctuation": "using punctuation or formatting that isn't allowed here",
    "contradiction": "contradicting something said earlier in the conversation",
}


def _build_fallback(categories: list[str]) -> str:
    """
    Composes the withheld-reply notice, naming the distinct rejection categories that were hit across the attempts.

    Parameters:
    - categories (list[str]): the `category` of every grounded rejection, in order -- comes from ResponseGate.check

    Returns:
    - str: the user-facing notice; always starts with _FALLBACK_LEAD so is_fallback_response can recognise it
    """
    seen: list[str] = []
    for c in categories:
        if c in _REJECT_PHRASE and c not in seen:
            seen.append(c)
    phrases = [_REJECT_PHRASE[c] for c in seen]
    if not phrases:
        return f"{_FALLBACK_LEAD} {_FALLBACK_TAIL}"
    joined = phrases[0] if len(phrases) == 1 else f"{', '.join(phrases[:-1])} and {phrases[-1]}"
    return f"{_FALLBACK_LEAD} It kept {joined}. {_FALLBACK_TAIL}"


def is_fallback_response(text: str) -> bool:
    """
    True if `text` is one of check()'s withheld-reply notices (any reason variant). chat_service / model_orchestration use this to mark the outgoing message "system" rather than a "backend" persona turn.
    """
    return text.startswith(_FALLBACK_LEAD)


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
        - str: a response that passed verification, or a fallback notice (built by _build_fallback, naming every rejection category hit across the attempts) if it's still rejected after regen_counter attempts — goes to model_orchestration
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
        # The `category` of every grounded rejection, in order -- the
        # exhausted-attempts fallback names the distinct ones it saw.
        reject_categories: list[str] = []
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

            reject_categories.append(category)
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

        fallback_response = _build_fallback(reject_categories)
        logger.warning(
            "ResponseGate.check exhausted %d attempts (categories: %s) -- returning fallback response.",
            regen_counter, reject_categories
        )
        await self.conversation_service.append_message(
            conversation_id=conversation_id,
            code=None,
            session_id=session_id,
            sender="error",
            text=f"[ResponseGate.check exhausted {regen_counter} attempts]\n{chr(10).join(reject_history)}\n\nLast rejected response:\n\n{current_response}"
        )
        return fallback_response
