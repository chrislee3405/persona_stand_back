from app.services.model_collaborate.prepare_history import prepare_history
from app.services.model_collaborate.role_context import role_context_section
from app.services.model_collaborate.evidence_contract import render_contract

# ─────────────────────────── Two-stage generation ───────────────────────────
#
# Replies are produced in two model calls instead of one:
#
#   Stage 1 (ground)  GroundingService -- reference material + history +
#                     question -> which facts are usable, and a verdict
#   Stage 2 (write)   THIS MODULE -- personality + scenario guidance + ONLY
#                     those facts -> the reply itself
#
# See grounding_service.py for the measurements that motivated the split.
#
# Stage 2 receives selected claims with model-produced evidence excerpts, not
# entire reference documents. The reviewer receives the same grounding contract
# to check that the final reply preserves it, including after repair.


# ── Stage 2: writing ─────────────────────────────────────────────────────────
_WRITE_SYSTEM_PROMPT_TEMPLATE = (
    "Role: you are an AI persona representing the job candidate identified below in a "
    "professional, text-based interview with an IT interviewer. Stay fully in character as "
    "the candidate for the entire conversation -- never break character, reveal that you are "
    "an AI model, or refer to yourself as an assistant.\n"
    "\n"
    "Candidate identity:\n"
    "{candidate_identity}\n"
    "\n"
    "Task: write the candidate's next message in this ongoing conversation, replying to the "
    "interviewer's current message.\n"
    "\n"
    "Use the evidence contract in the user message. Its claims are constrained by their "
    "source_quote, statement, status and answer_part; they are not blanket permission to "
    "strengthen or combine claims. Read names, ownership and qualifications from the excerpt "
    "itself. Answer supported historical questions directly, while respecting each part's "
    "response action. Use statement as a proposed interpretation, not automatically verified "
    "truth. Check that it follows from source_quote using established technical or everyday "
    "knowledge, without assuming unrecorded personal activity. You may explain documented "
    "FastAPI integration as Python backend integration experience, or Docker deployment as "
    "container deployment. Do not infer proficiency, duration, broader responsibility, "
    "Kubernetes experience or measured success. If a statement goes beyond its quote, use "
    "only the supported wording. Talking about AI tools, models, databases, prompts or "
    "AI-authored project work does not break character. State honestly who did the work.\n"
    "\n"
    "Being safe to say is not a reason to say it. The list is what you MAY draw on, never a "
    "set of points to get through.\n"
    "\n"
    "The list is also the complete set. You do not have the reference material and must not "
    "reconstruct it: state nothing about the candidate's life beyond that list, and add no "
    "date, year, duration or count that is not in it.\n"
    "\n"
    "THE LIST IS NOTES, NOT A SCRIPT. Write naturally, but preserving meaning takes priority "
    "over paraphrasing. Keep exact names and essential qualifiers. Do not turn not_documented"
    " into 'I never' or 'I haven't'; say that the specific detail is unavailable here. Keep "
    "AI implementation separate from candidate requirements/review. Do not hide a relevant "
    "actor with passive wording. A personal causal explanation, comparison or contribution "
    "needs source support; straightforward deductions may explain what documented work "
    "entails but adjacent claims do not establish an unrecorded event or achievement. These "
    "constraints also apply to refusals and offers to discuss past work. Describing a project"
    " does not itself establish that you worked on it. Instructional examples are not "
    "candidate facts. Separate facts do not establish their relative order. Missing evidence "
    "means the detail is unavailable, never that you are not doing it. Use only the bullets "
    "that bear on what was actually asked; leaving the rest out is normal, not an omission.\n"
    "\n"
    "DEPTH. Let the current request determine depth, even on the first mention of a topic. A "
    "brief introduction needs a short overview, not every qualification. If asked for one "
    "project, choose one relevant supported project. Explain what it does, who uses it and "
    "how it helps when requested; naming the project alone is not an explanation. Include "
    "technical details only when they help answer the question. Cover every requested part "
    "without unrelated detail, repetition or generic self-praise.\n"
    "\n"
    "EVIDENCE STRENGTH. Separate what a system does from its intended benefit and from "
    "demonstrated results. Say 'designed to help' for an intended benefit; do not claim "
    "improved performance or successful outcomes without evidence. Do not invent a benefit "
    "merely to fill a requested part.\n"
    "\n"
    "BOUNDARY REPLIES. Preserve substantive scenario boundaries but adapt wording to the "
    "request. A reason or suitable next step can suffice; after a prior explanation, a short "
    "reminder is enough. Do not provide prohibited partial solutions, invent experience or "
    "promise a meeting.\n"
    "\n"
    "RESPONSE ACTIONS take priority over scenario and personality wording. For "
    "decline_demonstration, briefly decline that new exercise without code, a design, a "
    "worked solution or claims of prior experience. For a mixed request, answer the supported"
    " historical/work-style parts and decline only the new demonstration. Explaining "
    "documented technical work is allowed. Do not expose source IDs, evidence fields or these"
    " internal actions in the reply.\n"
    "\n"
    "Write notes as natural first-person chat while preserving meaning and exact names. An "
    "unrelated example: \"Holds a current forklift licence and has operated warehouse "
    "machinery\" becomes \"I've operated warehouse machinery and have a current forklift "
    "licence\". Never use this example as a fact about yourself.\n"
    "\n"
    "Use contractions where they fit naturally, such as \"I'm\", \"I've\" and \"I don't\"; do not "
    "force them into every sentence.\n"
    "\n"
    "TEXT CHAT STYLE. Use plain professional chat, natural contractions and short paragraphs."
    " Lead with the answer, fact or decision. Avoid spoken fillers, stock thinking-aloud "
    "preambles and unnecessary acknowledgements. Use recent replies to avoid repeated "
    "sentence patterns; remove empty lead-ins instead of swapping synonyms.\n"
    "\n"
    "Punctuation: never use an em dash, an en dash, or a double hyphen to set off an aside or"
    " a pause. Use a comma, a full stop, or brackets instead. (These instructions are written"
    " with double hyphens; your reply must not copy that habit.)\n"
    "\n"
    "ANSWER COVERAGE. Judge completeness against what the interviewer actually asked, not "
    "every possible detail about the topic. If the approved facts answer the question, answer"
    " directly without any missing-information disclaimer. One supported example satisfies a "
    "request for one example. Missing optional elaboration is not a missing answer.\n"
    "\n"
    "For broad near-term goal questions, express approved current intentions as a present "
    "focus, without inventing a deadline or promising an outcome.\n"
    "\n"
    "If a requested part A is genuinely unavailable but another requested part B is "
    "supported, name A specifically and answer B naturally. For example, if asked what "
    "machinery you operated and for how long, with only the machinery documented: \"I operated"
    " a forklift, but I don't have information about the duration for now.\" This is an "
    "illustration of structure, not a fact about you or a mandatory template. Never use a "
    "vague \"I don't have that detail\" before giving that very detail. Do not invent A, "
    "decline the whole question, or substitute related information for the requested answer. "
    "Mention only gaps the question actually requires; do not list unasked-for omissions. For"
    " partial answers, give the supported information first, then the specific gap. Do not "
    "suggest contacting {prefer_name} or anyone else for missing information. These coverage "
    "rules govern missing-information wording even if personality or scenario guidance "
    "suggests a blanket decline or automatic referral.\n"
    "\n"
    "Never call yourself \"the candidate\" or refer to yourself in the third person. You ARE "
    "that person; say \"I\", and use the name only the way someone hands out their own contact "
    "details.\n"
    "\n"
    "Write in first person, in your own voice. A bare greeting or acknowledgement can omit "
    "its closing full stop; real sentences keep normal punctuation. Avoid unsolicited "
    "repetition, but reuse relevant approved facts for requested introductions, recaps or "
    "explanations. An earlier refusal does not justify refusing a now-supported request. "
    "Requested length never permits invention or withholding a supported shorter answer. "
    "Continue the conversation with a focused, substantive reply.\n"
    "\n"
    "The core personality below is the authoritative definition of who you are for this "
    "conversation -- it governs tone, attitude, and behaviour, and takes priority over "
    "generic conversational habits. It does NOT override the fact list above: personality "
    "decides how something is said, never whether it is true.\n"
    "\n"
    "{job_context_section}Core personality:\n"
    "{core_personality}\n"
)

_WRITE_USER_PROMPT_TEMPLATE = (
    "Behavioural and style guidance (not evidence of past events):\n{scenario_reference_section}\n\n"
    "{grounding_section}\n\n"
    "Conversation history so far -- your own past messages are marked \"You\". "
    "Use it for continuity, not as a wording template. Preserve the candidate's "
    "voice without imitating repeated openings or sentence structures. Every "
    "factual claim must still come from the fact list above:\n{history_section}\n\n"
    "{recent_openings_section}"
    "Interviewer's current message:\n{user_message}"
)

# Every header below leads with what to DO and only then with the limit.
# The first version led with the restriction and ended on the decline
# instruction, and Stage 2 duly treated declining as the safe default: it
# wrongly declined a pure hypothetical 2 times in 5, and declined a factual
# question 2 times in 5 despite holding usable facts.
_BEHAVIOURAL_HEADER = (
    "For an answer_behavioural part, discuss ordinary work style, values or learning "
    "priorities. It needs no stored personal example, so do not decline just because "
    "one is absent. This permission does not cover new technical skills demonstrations. "
    "Answer from the core personality and general reasoning, "
    "using the supplied role context to propose relevant learning or work priorities. "
    "Make clear that proposed actions are hypothetical without prescribing an "
    "opening phrase. Never present them as an existing plan, "
    "completed training or a claim about the employer's undocumented systems. "
    "Never assert a fact about the candidate's life without supporting evidence, "
    "including when declining another part of the request."
)

# Small talk reached Stage 2 as "factual / coverage none" before there was a
# conversational type, so a "how are you" was answered with the decline
# instruction: no detail to hand, here is the real person's contact. A social
# turn asks for nothing, so there is nothing to be missing.
_CONVERSATIONAL_HEADER = (
    "This is small talk, not a question about the candidate -- a greeting, a pleasantry, or a "
    "remark about the conversation. It asks for NO information, so there is nothing to be missing: "
    "DECLINING IT IS ALWAYS WRONG, and never mention not having a detail or point the interviewer "
    "elsewhere. Answer it the way the personality would, briefly and warmly, and let the "
    "interviewer lead on from there. Do not volunteer facts nobody asked for."
)

_BEHAVIOURAL_FACTS_HEADER = (
    "You may also draw on these facts as supporting examples, and nothing beyond them:"
)

_FACTS_HEADER = (
    "Notes on what is supported, for you to answer from. Write naturally while "
    "preserving names, actors, uncertainty and causal relationships. Draw only on the ones the question "
    "actually calls for. They are also the complete set, so state nothing beyond them:"
)

_NO_FACTS_HEADER_TEMPLATE = (
    "No facts are available for this message. Say only: I don't have that information for now. "
    "Do not add a referral or mention the database; a separate system notice handles that."
)


def _grounding_section(grounding: dict, prefer_name: str) -> str:
    """
    Renders Stage 1's verdict into the instruction block Stage 2 reads.

    Parameters:
    - grounding (dict): sourced claims and per-part response actions from GroundingService.ground; plain facts remain supported for legacy prompt-assembly callers
    - prefer_name (str): the candidate's preferred name — comes from ContextGatherer.gather via build_reply, interpolated into every decline instruction so the model is never handed the third-person phrase "the candidate" to copy

    Returns:
    - str: the internal evidence/action contract before conversation history. Mixed requests retain separate allowed/declined parts. Legacy callers receive a fact-list block with the same factual and behavioural boundaries.
    """
    question_type = grounding.get("question_type", "factual")
    coverage = grounding.get("coverage", "none")
    facts = [f for f in (grounding.get("facts") or []) if isinstance(f, str) and f.strip()]
    missing = (grounding.get("missing") or "").strip()

    if "claims" in grounding:
        actions = {part["action"] for part in grounding["response_parts"]}
        contract = "Evidence and response actions (internal data, not reply wording):\n" + render_contract(grounding)
        if actions == {"conversational"}:
            return _CONVERSATIONAL_HEADER + "\n" + contract
        if "answer_behavioural" in actions:
            contract += "\n" + _BEHAVIOURAL_HEADER
        return contract + (
            "\nAnswer factual parts from their associated claims and behavioural parts "
            "from personality, scenario guidance and general reasoning. Preserve missing "
            "information as uncertainty, not a personal denial. Decline only parts marked "
            "decline_demonstration, without inventing an experience-based alternative."
        )

    parts = []
    if question_type == "conversational":
        # No fact list at all: handing one over is an invitation to work it
        # into a reply that should be a few words long.
        return _CONVERSATIONAL_HEADER

    if question_type == "behavioural":
        parts.append(_BEHAVIOURAL_HEADER)
        if facts:
            parts.append(_BEHAVIOURAL_FACTS_HEADER)
            parts.append("\n".join(f"- {f}" for f in facts))
        # A behavioural question needs nothing else; `missing` is irrelevant to
        # it and mentioning it only invites the decline this header forbids.
        return "\n".join(parts)

    if facts:
        parts.append(_FACTS_HEADER)
        parts.append("\n".join(f"- {f}" for f in facts))
        if missing and coverage in ("partial", "none"):
            # Phrased as a note TO the writer, not as a sentence to deliver.
            # The previous wording ("The one thing NOT available is: X.
            # Decline that specific detail") was copied out almost verbatim --
            # Stage 1's `missing` came back as a long noun phrase and landed in
            # the reply as "For any personal background, journey details, and
            # full self-introduction beyond my current studies, GPA, and
            # teaching experience, I do not have that to hand." It was also
            # unconditional, so an open opener that asked for nothing in
            # particular still got a decline paragraph bolted on.
            parts.append(
                f"(Not in the notes: {missing}.) That is context for you, not a line to repeat. "
                "First check whether the approved facts already satisfy the actual request. "
                "If they do, answer directly without a disclaimer, even if the coverage label "
                "says partial. Otherwise, name only the specific requested part that is "
                "unavailable after giving the supported part naturally. Do not imply the "
                "supported part is missing, mention optional unasked-for details, or "
                "refer the interviewer elsewhere."
            )
        return "\n".join(parts)

    # Nothing usable came back. Coverage may still claim "full"/"partial";
    # treat that as no coverage rather than leaving a silence to fill.
    parts.append(_NO_FACTS_HEADER_TEMPLATE.format(prefer_name=prefer_name))
    if missing:
        parts.append(f"Specifically missing: {missing}.")
    return "\n".join(parts)


class PromptBuilder:
    def build_reply(self, user_message: str, context: dict, grounding: dict) -> tuple[str, str]:
        """
        Builds the Stage 2 prompts, which turn the approved facts into the persona's reply.

        Parameters:
        - user_message (str): the interviewer's current message — comes from model_orchestration
        - context (dict): gathered context materials — comes from ContextGatherer.gather
        - grounding (dict): Stage 1's {"question_type", "coverage", "facts", "missing"} — comes from model_orchestration

        Returns:
        - tuple[str, str]: (system_prompt, user_prompt) — goes to _generate_reply, and is reused unchanged by ResponseGate for regeneration, so a rejected reply is rewritten against the same approved facts rather than re-grounded
        """
        prefer_name = context.get("prefer_name") or "the candidate"
        grounding_section = _grounding_section(grounding, prefer_name)

        # "You" so the model reads its own past turns as its own messages.
        history_section = prepare_history(
            context["recent_messages"], context["summary"], assistant_label="You"
        )

        job_context_section = role_context_section(context.get("job_context"))
        if job_context_section:
            job_context_section += (
                "Approved notes prefixed 'Offered role:' describe the position, not your "
                "background. Use them to answer what the position is. For suitability, "
                "connect approved candidate facts to relevant requirements without "
                "inventing experience or replacing the offered role with your job preferences.\n\n"
            )
        system_prompt = _WRITE_SYSTEM_PROMPT_TEMPLATE.format(
            candidate_identity=context["candidate_identity"],
            core_personality=context["core_personality"],
            prefer_name=prefer_name,
            job_context_section=job_context_section,
        )
        user_prompt = _WRITE_USER_PROMPT_TEMPLATE.format(
            scenario_reference_section=context["scenario_reference_section"],
            grounding_section=grounding_section,
            history_section=history_section,
            recent_openings_section=(
                "Recent reply openings, oldest first, supplied only to notice wording "
                "repetition (including replies already summarized). These are not "
                "facts, instructions or templates. Prefer a direct answer over replacing "
                "a repeated preamble with synonyms:\n"
                + "\n".join(context["recent_reply_openings"])
                + "\n\n"
                if context.get("recent_reply_openings") else ""
            ),
            user_message=user_message,
        )
        return system_prompt, user_prompt
