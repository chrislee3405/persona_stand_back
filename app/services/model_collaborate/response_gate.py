import logging

from fastapi import Depends

from app.chat_trace import trace
from app.constants import DEFAULT_MODEL, Sender
from app.services.ai.gemini_service import GeminiService
from app.services.conversation_manage_service import ConversationService
from app.services.model_collaborate.prepare_history import prepare_history

logger = logging.getLogger(__name__)

# THE source of truth for the auditor's rejection vocabulary. Everything else
# in this module derives from this one dict: the rule list in the system
# prompt, the tag list the prompt tells the model to choose from, the
# structured-output enum, and the user-facing fallback wording. Add or rename
# a category here and all four follow.
#   rule   -- the rule text shown to the auditor, tagged with its category
#   phrase -- how that rejection is described to the visitor in the fallback
_REJECT_RULES: dict[str, dict[str, str]] = {
    "broke_character": {
        # The carve-out exists because the persona's CORRECT behaviour when it
        # has no information is to say so -- and that reply kept being rejected
        # here, so every unanswerable question ended in the exhausted-attempts
        # fallback. The auditor cannot and need not check whether the
        # information really was missing; that is the generator's job, since it
        # is the one holding the reference material. This rule only has to stop
        # punishing the right answer, which needs no data at all -- just the
        # shape of the reply.
        "rule": (
            "The AI response must never break character, reveal that it is an AI model, or refer "
            "to itself as an assistant. ONE EXCEPTION -- having no answer is a PERMITTED state, "
            "not a character break: do NOT reject a reply that says it does not have that "
            "information, cannot go into it here, or would rather cover it in a live "
            "conversation. You cannot independently check whether reference material was "
            "missing, but still check the reply for an internal contradiction under the "
            "contradiction rule. PASS: \"I don't have the duration here, but I operated a forklift.\" "
            "REJECT: \"As an AI, I don't have access to that.\" / any reply that offers to "
            "assist, apologises for a limitation, or mentions a system, model, database, or "
            "prompt."
        ),
        "phrase": "breaking character",
    },
    "unnatural": {
        "rule": (
            "Natural wording is a writing preference, not a delivery requirement. "
            "Ordinary interview formality, complete sentences, professional vocabulary, "
            "mildly polished phrasing or an opener are not grounds for rejection. "
            "Do not report style-only concerns as violations or relabel them as another category."
        ),
        "phrase": "sounding unnatural",
    },
    "asked_for_private_info": {
        "rule": "The AI response should not ask the user to input any credential or private information.",
        "phrase": "asking you for personal information",
    },
    "made_a_promise": {
        "rule": "The AI response should not make any promise to the user.",
        "phrase": "making a promise it shouldn't",
    },
    "punctuation": {
        # Kept deliberately short. This rule had grown to 1193 chars -- 64% of
        # the whole rule list -- because every false positive added a clause and
        # an example, which crowded out the other five rules. Only the reject
        # list belongs here; the closing "ordinary punctuation" line does the
        # work the old allow-list enumeration did, and the bare-greeting full
        # stop is a generation concern (see prompt_builder), not an audit one.
        "rule": (
            "Reject ONLY these AI-writing tells: emoji; markdown or markup (**, [], <>, backticks, "
            "#); bullet or numbered lists; a colon heading a label or section (\"Skills:\"); "
            "semicolons in casual chat; filler ellipses; an em or en dash (—, –) or double "
            "hyphen used as sentence punctuation for an aside or dramatic pause (reject "
            "\"teamwork—skills I know are huge\"); an exclamation mark inside a longer sentence. "
            "Ordinary punctuation is never by itself a reason to reject."
        ),
        "phrase": "using punctuation or formatting that isn't allowed here",
    },
    "contradiction": {
        "rule": (
            "Reject only two explicit claims that cannot both be true about the same "
            "subject, scope and time. One must be in the current reply; the other must "
            "be elsewhere in that reply or a previous persona reply. Quote both claims "
            "and explain their incompatibility. A user question, implied expectation, "
            "missing detail or failure to answer directly is not a conflicting claim. "
            "Different time horizons or goals are not inherently contradictory. "
            "Saying a specific fact is unavailable while providing that same fact is "
            "a contradiction; answering a supported part while identifying a different "
            "missing part is not."
        ),
        "phrase": "contradicting itself or something said earlier in the conversation",
    },
}

# Derived views. _REJECT_PHRASE keeps its original name and shape so
# _build_fallback below reads unchanged; _CATEGORIES drives both the prompt's
# tag list and the response schema's enum.
_REJECT_PHRASE = {category: spec["phrase"] for category, spec in _REJECT_RULES.items()}
_CATEGORIES = tuple(_REJECT_RULES)
_CATEGORY_ENUM = ["none", *_CATEGORIES]

_RULES_SECTION = "\n".join(
    f"- [{category}] {spec['rule']}" for category, spec in _REJECT_RULES.items()
)

_VERIFY_NATURAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE = (
    "Role: you are an AI auditor verifying one exchange between a human "
    "interviewer and an AI model role-playing a given interviewee persona.\n\n"
    "Task: check whether the AI response breaks any of the rules below.\n\n"
    "Rules (each tagged with its category):\n"
    f"{_RULES_SECTION}\n\n"
    "If the response breaks no rule, return \"<pass>\" as result and an empty "
    "violations list.\n\n"
    "If the response breaks one or more rules, return the reject type as result:\n"
    "- \"<reject-minor>\": another AI could correct the response given only the "
    "reject reasons and the original response.\n"
    "- \"<reject-major>\": another AI needs the reject reasons plus the whole "
    "background information again, to regenerate the response with a different "
    "approach. Use this if ANY single violation needs it.\n\n"
    "Report EVERY rule the response breaks, not just the most serious one -- "
    "return one entry in violations per broken rule, so a single rewrite can "
    "fix all of them at once. Listing only one wastes the regeneration on a "
    "partial fix. Do not report the same rule twice; if one rule is broken in "
    "several places, use one entry and quote the clearest instance.\n\n"
    f"Each violation has: category, the tag of the broken rule ({', '.join(_CATEGORIES)}); "
    "reason, a specific explanation of what broke; and quote, the exact "
    "substring of the AI response that demonstrates it, copied "
    "character-for-character, not a paraphrase or summary. For contradiction, also "
    "supply conflicting_quote (the other exact claim) and conflict_source "
    "('response' or 'history'). History evidence must be from the persona, never "
    "the interviewer. For other categories use an empty conflicting_quote and "
    "conflict_source 'none'. If no explicit opposing claim exists, do not reject.\n\n"
    "A violation you cannot ground in an exact substring of the response is a "
    "guess -- leave it out. If that leaves no violations at all, return "
    "\"<pass>\" with an empty violations list instead."
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


def _has_rejection_evidence(violation: dict, response: str, history: list) -> bool:
    """Style alone never blocks; contradictions need two distinct, sourced claims."""
    category, quote = violation.get("category"), violation.get("quote")
    if category not in _REJECT_RULES or category == "unnatural":
        return False
    if not isinstance(quote, str) or not quote.strip() or quote not in response:
        return False
    if category != "contradiction":
        return True
    other = violation.get("conflicting_quote")
    if not isinstance(other, str) or not other.strip() or other in quote or quote in other:
        return False
    if violation.get("conflict_source") == "response":
        return other in response
    if violation.get("conflict_source") == "history":
        return any(message.sender == Sender.BACKEND and other in message.text for message in history)
    return False


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

    async def _verify_response(self, recent_conversation_part: str, user_message: str, ai_response: str) -> tuple[str, list[dict]]:
        """
        Asks Gemini to audit one candidate response against the natural-response rules, collecting every rule it breaks.

        Parameters:
        - recent_conversation_part (str): formatted recent-history block, or "" if none — comes from check
        - user_message (str): the interviewer's current message — comes from check
        - ai_response (str): the candidate response being audited — comes from check (either the original reply or a regenerated one)

        Returns:
        - tuple[str, list[dict]]: (result_tag, violations). result_tag is one of "<pass>" / "<reject-minor>" / "<reject-major>". violations is one entry per broken rule, each {"category", "reason", "quote"}, empty on a pass — the auditor is asked for ALL of them rather than the single most serious, so one regeneration can correct the whole set instead of spending an attempt per problem. Goes to check, which cross-checks each quote against ai_response before trusting that violation.
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
                "violations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "category": {
                                "type": "STRING",
                                "enum": list(_CATEGORIES)
                            },
                            "reason": {"type": "STRING"},
                            "quote": {"type": "STRING"},
                            "conflicting_quote": {"type": "STRING"},
                            "conflict_source": {"type": "STRING", "enum": ["response", "history", "none"]},
                        },
                        "required": ["category", "reason", "quote", "conflicting_quote", "conflict_source"]
                    }
                }
            },
            "required": ["result", "violations"]
        }

        # The auditor's own failure must not fail the turn. call_model_structured
        # returns whatever json.loads() produced, so a degraded structured-output
        # call can hand back null, a list or a bare string -- and `.get` on any of
        # those raises AttributeError, which propagates all the way to
        # ChatService's catch-all and costs the visitor their reply. Same shape as
        # GroundingService.ground's guard, but the safe direction is the opposite
        # one: with no usable audit there is no EVIDENCE the response is at fault,
        # so it passes through, exactly as an audit returning only ungrounded
        # violations already does below.
        try:
            check_result = await self.gemini_service.call_model_structured(
                model_name=DEFAULT_MODEL,
                user_prompt=user_prompt,
                system_prompt=_VERIFY_NATURAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE,
                schema=schema
            )
        except Exception:
            logger.exception("ResponseGate verification call failed -- passing the response through unaudited.")
            return "<pass>", []

        if not isinstance(check_result, dict):
            logger.warning("ResponseGate verification returned no object -- passing the response through unaudited.")
            trace.debug("response gate malformed verification: %r", check_result)
            return "<pass>", []

        result = check_result.get("result", "<pass>")
        violations = check_result.get("violations") or []
        # Defensive: the schema asks for a list of objects with all three
        # fields, but a malformed item would otherwise blow up the grounding
        # check in the caller.
        violations = [
            v for v in violations
            if isinstance(v, dict) and isinstance(v.get("quote"), str)
        ]
        return result, violations

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
            # Trailing newline: without it the next label in the template was
            # glued onto the last history line ("...Assistant: hiCurrent user message:").
            recent_conversation_part = (
                "Past few conversation:\n"
                + prepare_history(context["recent_messages"], None)
                + "\n"
            )

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
            result, violations = await self._verify_response(recent_conversation_part, user_message, current_response)

            if result == "<pass>":
                return current_response

            # Every violation must be grounded in an exact substring of the
            # response it judges (see the system prompt's quote instruction).
            # One that isn't -- an empty/placeholder quote, or text that never
            # appears -- is the auditor guessing (e.g. rejecting for a
            # semicolon that was never there), so it is discarded rather than
            # spending a regeneration correcting a problem that doesn't exist.
            # Discarding is now per-violation: a single bad quote no longer
            # throws away the genuine problems reported alongside it.
            grounded = [v for v in violations if _has_rejection_evidence(v, current_response, context["recent_messages"])]
            discarded = [v for v in violations if v not in grounded]
            for v in discarded:
                # The category is a fixed label; the reason and quote are the
                # auditor's words about the reply, so they are trace-only.
                logger.warning(
                    "ResponseGate.check attempt %d/%d: dropped %s violation -- advisory or insufficient evidence.",
                    attempt + 1, regen_counter, v.get("category"),
                )
                trace.debug("response gate dropped violation reason=%r quote=%r", v.get("reason"), v.get("quote"))

            # Nothing survived, so there is no evidence the response is
            # actually at fault -- let it through, as before.
            if not grounded:
                logger.warning(
                    "ResponseGate.check attempt %d/%d: %s with no grounded violation -- passing the response through.",
                    attempt + 1, regen_counter, result
                )
                return current_response

            logger.warning(
                "ResponseGate.check attempt %d/%d rejected (%s) for %d violation(s): %s",
                attempt + 1, regen_counter, result,
                len(grounded), ", ".join(v["category"] for v in grounded)
            )

            reject_categories.extend(v["category"] for v in grounded)
            # One history entry per attempt, listing every problem that
            # attempt had, so the regeneration below fixes them together.
            attempt_detail = "\n".join(
                f"  - [{v['category']}] {v['reason']} (quoted: {v['quote']!r})" for v in grounded
            )
            reject_history.append(f"Attempt {attempt + 1} ({result}):\n{attempt_detail}")
            reject_history_text = "\n".join(reject_history)

            # Keep the rejected attempt for later review -- excluded from
            # live prompt history and summarization via the sender=Sender.REGEN
            # filter in ConversationService.get_recent_messages.
            await self.conversation_service.append_message(
                conversation_id=conversation_id,
                code=None,
                session_id=session_id,
                sender=Sender.REGEN,
                text=f"[Attempt {attempt + 1}/{regen_counter}, {result}]\n{attempt_detail}\n\n{current_response}"
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
                    "changing as little else as possible. Preserve the original voice, "
                    "sentence structure and contractions unless a specific correction "
                    "requires changing them. Do not add an acknowledgement, spoken "
                    "filler or a new introductory phrase."
                )
            else:  # <reject-major>
                regen_system_prompt = (
                    f"{system_prompt}\n\n"
                    f"Your previous responses were rejected across these attempts so far:\n{reject_history_text}\n\n"
                    f"Most recent response:\n{current_response}\n\n"
                    "Regenerate a new response using a different approach that avoids all of these problems."
                )

            current_response = await self.gemini_service.call_model(
                model_name=DEFAULT_MODEL,
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
            sender=Sender.ERROR,
            text=f"[ResponseGate.check exhausted {regen_counter} attempts]\n{chr(10).join(reject_history)}\n\nLast rejected response:\n\n{current_response}"
        )
        return fallback_response
