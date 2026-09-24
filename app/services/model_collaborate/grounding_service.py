import logging

from fastapi import Depends

from app.chat_trace import trace
from app.constants import DEFAULT_MODEL
from app.services.ai.gemini_service import GeminiService
from app.services.model_collaborate.prepare_history import prepare_history

logger = logging.getLogger(__name__)

# Stage 1 of reply generation. Decides what the reference material actually
# supports for this message; PromptBuilder + the model then write the reply
# from nothing but the facts this stage approves.
#
# WHY THIS IS A SEPARATE CALL. A single prompt carrying identity, personality,
# voice, punctuation, history rules AND grounding rules reliably invented
# facts: an ablation over three real failures (persona_stand_back/
# probe_ablation.py) scored 8-10 out of 15 fabrications with the 6,535-char
# combined prompt, and 0 out of 15 with a 329-char prompt that did nothing but
# ground. The rule that has to win was ~5% of the old prompt and lost to
# everything around it. Here it is nearly all of the prompt.
#
# Removing personality from the combined prompt was tested too and was the
# WORST condition (15/15) -- the grounding rules live inside core_personality,
# so deleting that block deletes them. Hence ground-then-write, not
# personality-then-facts.
#
# Deliberately excludes core_personality and the scenario guidance: this stage
# judges whether facts exist, and anything about voice or behaviour would only
# compete with that.
#
# TWO INDEPENDENT AXES, deliberately kept as separate fields. An earlier
# version had one enum -- grounded / partial / none / not_factual -- which
# mixed a property of the QUESTION (is it asking for facts?) with a property
# of the MATERIAL (does it cover them?). Asked to collapse both into one
# choice, the model reached for "not_factual" whenever material was absent:
# "what are you doing now" with no reference material scored not_factual 5
# times out of 5. Separating the axes removes the escape hatch -- a question
# is factual or not regardless of whether anything supports it.
_GROUND_SYSTEM_PROMPT = (
    "You decide what a job candidate can truthfully say in reply to an "
    "interviewer's message. You are not writing the reply.\n\n"
    "You are given the reference material available for this message and the "
    "conversation so far. Judge two things separately.\n\n"
    "1. `question_type` -- what the interviewer is asking for:\n"
    "- \"factual\"     anything that is TRUE of the candidate: their job, studies, "
    "employer, school, skills, tools, projects, awards, dates, places, status, "
    "what they are doing now, how many of something, when something happened.\n"
    "- \"behavioural\" how the candidate would ACT, what they value, their "
    "approach or opinion, or a hypothetical -- questions answerable from "
    "character alone.\n"
    # Without this third value, ordinary chat had nowhere to go. "How are
    # you" asks nothing about the candidate's life and nothing about how they
    # would act, so it came back "factual" with coverage "none" -- and the
    # writer dutifully answered a greeting by declining and handing out
    # contact details. Small talk is not a question the reference material
    # could ever cover; it needs no facts at all.
    "- \"conversational\" greetings, small talk and social turns that ask for "
    "no information: \"hi\", \"how are you\", \"nice to meet you\", \"thanks\", "
    "\"how's your day going\", and remarks about the conversation itself.\n"
    "This depends ONLY on the question. Whether any material supports it is "
    "irrelevant here: a factual question with nothing to support it is still "
    "factual. Never use \"behavioural\" to signal that material is missing -- "
    "that is what `coverage` is for, and \"conversational\" is only for a turn "
    "that asks for nothing at all.\n\n"
    "2. `coverage` -- how much of what was asked the material actually "
    "supplies:\n"
    "- \"full\"    everything asked for is there\n"
    "- \"partial\" some of it is there, some is not\n"
    "- \"none\"    none of it is there\n\n"
    "List in `facts` only what the material or the conversation actually "
    "states, staying close to their wording. Do not add, infer, combine or "
    "round anything -- above all not dates, years, durations or counts. If the "
    "material names something without describing it, the name is the fact; its "
    "details are not. For a behavioural question, include any facts that could "
    "serve as a genuine example, or leave the list empty. For a conversational "
    "turn leave `facts` empty, set `coverage` to \"none\" and leave `missing` "
    "empty -- nothing was asked, so nothing is missing.\n\n"
    # Reference rows often open with a CV-style header ("Master of X (AI) |
    # GPA: ...") above the prose that qualifies it. This stage reliably took
    # the header as the fact and dropped the qualifier: given a row saying
    # "I've finished a Master of ...", it emitted the bare noun phrase
    # "Master of Information Technology (AI)" on 3 runs out of 3, and the
    # writer then guessed the status -- reporting a completed degree as still
    # in progress 3 times in 5. Dropping a qualifier is not the safe direction;
    # it silently hands the next stage something to invent.
    "KEEP ANY WORD THAT FIXES STATUS OR TENSE with the fact it qualifies -- "
    "finished or in progress, current or former, holds it or wants it, "
    "completed or expected. Where a bare title or heading and a fuller "
    "sentence describe the same thing, take the status from the sentence. "
    "Never reduce a qualified fact to a bare noun phrase: that does not make "
    "it safer, it just leaves the status for someone else to guess at.\n\n"
    "If the interviewer asked for something the material does not contain, "
    "that fact does not exist. Put what is missing in `missing`; never supply "
    "it yourself. Leave `missing` empty when coverage is \"full\"."
)

_GROUND_USER_PROMPT_TEMPLATE = (
    "Reference material:\n{doc_reference_section}\n\n"
    "Previously answered questions (facts only, not wording to copy):\n{examples_section}\n\n"
    "Conversation so far:\n{history_section}\n\n"
    "Interviewer's current message:\n{user_message}"
)

_GROUND_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "question_type": {"type": "STRING", "enum": ["factual", "behavioural", "conversational"]},
        "coverage": {"type": "STRING", "enum": ["full", "partial", "none"]},
        "facts": {"type": "ARRAY", "items": {"type": "STRING"}},
        "missing": {"type": "STRING"},
    },
    "required": ["question_type", "coverage", "facts", "missing"],
}

# Used when the call fails or comes back malformed: a factual question with
# nothing behind it, so the persona says it does not have the detail. Failing
# toward a decline is the safe direction -- a wrong decline is recoverable, a
# fabrication is not.
GROUND_FALLBACK = {
    "question_type": "factual",
    "coverage": "none",
    "facts": [],
    "missing": "",
}


class GroundingService:
    def __init__(self, gemini_service: GeminiService = Depends()):
        """
        Stores the injected Gemini service.

        Parameters:
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI, or passed explicitly by ModelCollaborateService

        Returns:
        - None: sets self.gemini_service. No DB session and no ConversationService: this stage only reads the context it is handed and persists nothing.
        """
        self.gemini_service = gemini_service

    async def ground(self, user_message: str, context: dict) -> dict:
        """
        Asks the model which facts the reference material and conversation actually support for this message.

        Parameters:
        - user_message (str): the interviewer's current message — comes from ModelCollaborateService.model_orchestration
        - context (dict): gathered context materials — comes from ContextGatherer.gather

        Returns:
        - dict: {"question_type", "coverage", "facts", "missing"} shaped by _GROUND_RESPONSE_SCHEMA — goes to PromptBuilder.build_reply, which turns it into the fact list Stage 2 writes from. Returns GROUND_FALLBACK rather than raising when the call fails or the reply is malformed, so a broken grounding call produces a decline instead of an unconstrained reply.
        """
        system_prompt, user_prompt = self._build_prompts(user_message, context)

        trace.debug("grounding (stage 1) system prompt: %s", system_prompt)
        trace.debug("grounding (stage 1) user prompt: %s", user_prompt)

        try:
            grounding = await self.gemini_service.call_model_structured(
                model_name=DEFAULT_MODEL,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=_GROUND_RESPONSE_SCHEMA
            )
        except Exception:
            logger.exception("Grounding call failed -- falling back to no available facts.")
            grounding = None

        if not isinstance(grounding, dict) or "question_type" not in grounding:
            # The response itself is model output about the conversation, so
            # it goes to the trace logger; this line only says it happened.
            logger.warning("Grounding returned no question_type/coverage -- treating as no facts.")
            trace.debug("grounding malformed response: %r", grounding)
            grounding = dict(GROUND_FALLBACK)

        logger.debug(
            "Grounding question_type=%s coverage=%s facts=%d",
            grounding.get("question_type"), grounding.get("coverage"),
            len(grounding.get("facts") or []),
        )
        trace.debug("grounding facts=%r missing=%r", grounding.get("facts"), grounding.get("missing"))

        return grounding

    def _build_prompts(self, user_message: str, context: dict) -> tuple[str, str]:
        """
        Formats the gathered context into this stage's system and user prompts.

        Parameters:
        - user_message (str): the interviewer's current message — comes from ground
        - context (dict): gathered context materials — comes from ground

        Returns:
        - tuple[str, str]: (system_prompt, user_prompt) — goes to ground. History is labelled "Candidate" rather than "You": this stage is a third party judging an exchange, not the persona continuing one.
        """
        examples_section = (
            "\n\n".join(
                f"A similar question asked before: {ex['question']}\n"
                f"Facts the candidate gave then: {ex['answer']}"
                for ex in context["similar_examples"]
            )
            if context["similar_examples"]
            else "No related past answers found."
        )
        history_section = prepare_history(
            context["recent_messages"], context["summary"], assistant_label="Candidate"
        )
        user_prompt = _GROUND_USER_PROMPT_TEMPLATE.format(
            doc_reference_section=context["doc_reference_section"],
            examples_section=examples_section,
            history_section=history_section,
            user_message=user_message,
        )
        return _GROUND_SYSTEM_PROMPT, user_prompt
