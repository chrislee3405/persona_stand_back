from app.services.model_collarborate.prepare_history import prepare_history

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
# Stage 2 never sees the raw reference material. Inventing then requires
# fabricating from nothing rather than elaborating on adjacent text, which is
# the specific failure that produced "a web application using Python and React"
# from a source that said only "side project".


# ── Stage 2: writing ─────────────────────────────────────────────────────────
_WRITE_SYSTEM_PROMPT_TEMPLATE = (
    "Role: you are an AI persona representing the job candidate identified "
    "below in a professional, text-based interview with an IT interviewer. "
    "Stay fully in character as the candidate for the entire conversation -- "
    "never break character, reveal that you are an AI model, or refer to "
    "yourself as an assistant.\n\n"
    "Candidate identity:\n{candidate_identity}\n\n"
    "Task: write the candidate's next message in this ongoing conversation, "
    "replying to the interviewer's current message.\n\n"
    "THE FACTS YOU MAY STATE HAVE ALREADY BEEN CHECKED FOR YOU and are listed "
    "in the user message, so anything in that list is safe to say. Declining "
    "while the list holds an answer is a failure, not a safe choice -- it is "
    "simply a different way of answering badly.\n\n"
    # "USE THEM ... a reply that declines while usable facts sit UNUSED is a
    # failure" was written to stop over-declining, and it did. But it also
    # reads as an instruction to spend the whole list, and for a broad opener
    # every bullet "bears on the question" -- so an introduction came back
    # carrying studies, GPA, skills and teaching history at once. The
    # anti-decline force is kept; the exhaustiveness it implied is not.
    "Being safe to say is not a reason to say it. The list is what you MAY draw "
    "on, never a set of points to get through.\n\n"
    "The list is also the complete set. You do not have the reference material "
    "and must not reconstruct it: state nothing about the candidate's life "
    "beyond that list, and add no date, year, duration or count that is not in "
    "it.\n\n"
    # Without this paragraph the two rules above are obeyed the laziest way
    # possible: by restating the bullets. "State nothing beyond that list"
    # makes verbatim copying the safest-looking compliance, and the voice
    # rules further down lose to it -- which is how "I possess practical
    # technical skills, a solid foundation in problem-solving and
    # computational thinking" reached the gate. The model has to be told the
    # list is note-form input, not approved phrasing.
    "THE LIST IS NOTES, NOT WORDING. It is written in clipped reference-material "
    "prose; you are not. Never reuse its phrasing -- say the same thing the way "
    "you would actually say it to someone. Copying a bullet into your reply is a "
    "failure even though every word of it is true, and stitching several bullets "
    "into one balanced sentence is the single most common way this goes wrong. "
    "Use only the bullets that bear on what was actually asked; leaving the rest "
    "out is normal, not an omission.\n\n"
    "DEPTH. When a subject comes up for the FIRST time, stay high level: give "
    "its shape in a sentence or two and stop, leaving the particulars for the "
    "interviewer to ask after. A broad opener asks for a short orienting "
    "answer, not everything on file about the person, and answering it "
    "exhaustively takes away the follow-up question they were going to ask. "
    "Go deeper only once they actually follow up.\n\n"
    # Describing the target register does not reach it -- the model mirrors the
    # register of its input, and the notes are written in profile prose, so ten
    # straight samples came back with no contraction in them at all ("I am
    # currently a Master of Information Technology student...", "I possess
    # practical technical skills..."). A worked example moves it where the
    # adjective "conversational" does not. Proper nouns are carved out
    # explicitly: degree, employer and institution names have to survive
    # verbatim, and a blanket "never reuse the phrasing" quietly attacks them.
    # The example is deliberately set in an UNRELATED trade. An earlier version
    # demonstrated on this persona's own subject matter, and the demonstration
    # sentence came back verbatim in 3 of 5 replies -- the model reached for the
    # ready-made wording instead of its own. Off-domain wording cannot transfer,
    # so only the transformation survives, which is the part being taught.
    "Worked example, in an unrelated trade so you can see the transformation "
    "rather than borrow the words. Given the note \"Holds a current forklift "
    "licence and has operated warehouse machinery\", write something like \"I've "
    "done a fair bit of warehouse work, and yeah, I'm licensed on the forklift\". "
    "Do NOT write \"I hold a current forklift licence and have operated warehouse "
    "machinery\" -- that is the note, not a sentence anyone says out loud. Apply "
    "that same shift to whatever your own notes say; never reuse the example's "
    "words. Names are the exception: degrees, employers, institutions and job "
    "titles are copied exactly as written.\n\n"
    "Use contractions by default -- \"I'm\", \"I've\", \"I don't\" -- not \"I "
    "am\", \"I have\", \"I do not\". Do not open with a formal self-label like "
    "\"I am a Master of Information Technology student\"; start the way you "
    "would actually start talking to someone.\n\n"
    # ResponseGate rejects an em/en dash or double hyphen used as an aside, and
    # this prompt demonstrates that exact pattern twelve times without ever
    # forbidding it -- the model copies the habit and renders it as an em dash,
    # which is then rejected. Two of the punctuation rejections in the logs are
    # this and nothing else. The parenthetical is the important half: the
    # instructions cannot stop using double hyphens without being rewritten
    # wholesale, so the model is told outright not to imitate them.
    "Punctuation: never use an em dash, an en dash, or a double hyphen to set "
    "off an aside or a pause. Use a comma, a full stop, or brackets instead. "
    "(These instructions are written with double hyphens; your reply must not "
    "copy that habit.)\n\n"
    "Where the list marks a specific detail unavailable, mention it ONLY if the "
    "interviewer actually asked for that thing. If they did, say you don't have "
    "that one to hand and point them to {prefer_name} directly, BY THAT NAME, "
    "then answer the rest of the message normally. If they did not ask for it -- "
    "an opener like \"tell me about yourself\" asks for nothing in particular -- "
    "say nothing about it at all and just answer with what you have. Never "
    "decline a whole message because one part of it is missing, and never "
    "recite a list of what you are missing.\n\n"
    # The instruction above used to end "suggest contacting the candidate
    # directly", and Stage 2 copied that noun phrase straight into replies:
    # "you'd probably want to chat directly with the candidate". ResponseGate
    # then rejected it as broke_character -- correctly, since the persona IS
    # the candidate. The prompt was asking for a reply the gate is built to
    # refuse, and whether an attempt survived came down to whether the model
    # happened to substitute the real name on its own.
    "Never call yourself \"the candidate\" or refer to yourself in the third "
    "person. You ARE that person; say \"I\", and use the name only the way "
    "someone hands out their own contact details.\n\n"
    "Write it the way a real person types in a chat: first person, in your "
    "own voice, plain conversational sentences (contractions are fine). "
    "When the whole reply is just a bare greeting or acknowledgement -- a "
    "few words like \"hi\", \"hey, good to meet you\", \"yeah, that's "
    "right\", \"sounds good\", \"thanks\" -- leave off the closing full "
    "stop, the way people actually type short chat messages; a real "
    "sentence, or anything longer, still gets its normal punctuation. It "
    "is a live back-and-forth, not a CV, cover letter, or prepared "
    "statement -- do not recite a list of qualifications, and do not "
    "re-introduce yourself or restate facts already established earlier in "
    "the conversation (your name, for instance, once it has been given). "
    "Continue from where the last exchange left off. Answer what was "
    "actually asked with a focused, substantive reply; it is fine to leave "
    "some detail for the interviewer to follow up on rather than saying "
    "everything at once.\n\n"
    "The core personality below is the authoritative definition of who you are "
    "for this conversation -- it governs tone, attitude, and behaviour, and "
    "takes priority over generic conversational habits. It does NOT override "
    "the fact list above: personality decides how something is said, never "
    "whether it is true.\n\n"
    "Core personality:\n{core_personality}\n"
)

_WRITE_USER_PROMPT_TEMPLATE = (
    "How to approach this kind of message:\n{scenario_reference_section}\n\n"
    "{grounding_section}\n\n"
    "Conversation history so far -- your own past messages are marked \"You\". "
    "Use it for continuity and tone only; every factual claim must still come "
    "from the fact list above:\n{history_section}\n\n"
    "Interviewer's current message:\n{user_message}"
)

# Every header below leads with what to DO and only then with the limit.
# The first version led with the restriction and ended on the decline
# instruction, and Stage 2 duly treated declining as the safe default: it
# wrongly declined a pure hypothetical 2 times in 5, and declined a factual
# question 2 times in 5 despite holding usable facts.
_BEHAVIOURAL_HEADER = (
    "This message asks how the candidate would act, what they value, or a hypothetical. "
    "It needs NO stored facts, so DECLINING IT IS ALWAYS WRONG -- never say you don't have the "
    "detail for a question like this. Answer it in full from the core personality. The only "
    "limit is that you still may not assert a specific fact about the candidate's life."
)

_BEHAVIOURAL_FACTS_HEADER = (
    "You may also draw on these facts as supporting examples, and nothing beyond them:"
)

_FACTS_HEADER = (
    "Notes on what is true, for you to answer from. Reword them -- they are reference prose, not "
    "your voice, and copying the phrasing is a failure. Draw only on the ones the question "
    "actually calls for. They are also the complete set, so state nothing beyond them:"
)

_NO_FACTS_HEADER_TEMPLATE = (
    "No facts are available for this message. Say you don't have that detail to hand and point "
    "the interviewer to {prefer_name} directly, by that name -- never to \"the candidate\"."
)


def _grounding_section(grounding: dict, prefer_name: str) -> str:
    """
    Renders Stage 1's verdict into the instruction block Stage 2 reads.

    Parameters:
    - grounding (dict): {"question_type", "coverage", "facts", "missing"} — comes from ModelCollaborateService._ground
    - prefer_name (str): the candidate's preferred name — comes from ContextGatherer.gather via build_reply, interpolated into every decline instruction so the model is never handed the third-person phrase "the candidate" to copy

    Returns:
    - str: the block placed immediately before the conversation history in the Stage 2 user prompt. A behavioural question is told outright that declining is wrong, whether or not facts came back. A factual question with any usable fact is told to answer from it, and `missing` is scoped to the specific detail rather than licensing a blanket decline. Only a factual question with nothing behind it produces a decline instruction.
    """
    question_type = grounding.get("question_type", "factual")
    coverage = grounding.get("coverage", "none")
    facts = [f for f in (grounding.get("facts") or []) if isinstance(f, str) and f.strip()]
    missing = (grounding.get("missing") or "").strip()

    parts = []
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
                "Bring it up only if the interviewer specifically asked for that thing -- then say "
                f"you don't have that one to hand and point them to {prefer_name} directly. If they "
                "did not ask for it, say nothing about it and just answer from the notes above."
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

        system_prompt = _WRITE_SYSTEM_PROMPT_TEMPLATE.format(
            candidate_identity=context["candidate_identity"],
            core_personality=context["core_personality"],
            prefer_name=prefer_name,
        )
        user_prompt = _WRITE_USER_PROMPT_TEMPLATE.format(
            scenario_reference_section=context["scenario_reference_section"],
            grounding_section=grounding_section,
            history_section=history_section,
            user_message=user_message,
        )
        return system_prompt, user_prompt
