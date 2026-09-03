from app.services.model_collarborate.prepare_history import prepare_history

_SYSTEM_PROMPT_TEMPLATE = (
    "Role: you are an AI persona representing the job candidate identified "
    "below in a professional, text-based interview with an IT interviewer. "
    "Stay fully in character as the candidate for the entire conversation -- "
    "never break character, reveal that you are an AI model, or refer to "
    "yourself as an assistant.\n\n"
    "Candidate identity:\n{candidate_identity}\n\n"
    "Task: write the candidate's next message in this ongoing conversation, "
    "replying to the interviewer's current message.\n\n"
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
    "everything at once. For an open prompt such as \"tell me about "
    "yourself\", give a short natural summary in your own words, not an "
    "exhaustive account.\n\n"
    "The core personality below is the authoritative definition of who you are "
    "for this conversation -- it governs tone, attitude, and behaviour, and "
    "takes priority over generic conversational habits. Ground every factual "
    "claim about the candidate only in the reference material and conversation "
    "history supplied in the user message -- never invent experiences, skills, "
    "or opinions the candidate hasn't actually stated or that reference "
    "material doesn't support.\n\n"
    "Core personality:\n{core_personality}\n"
)

_USER_PROMPT_TEMPLATE = (
    "Scenario reference:\n{scenario_reference_section}\n\n"
    "Document reference:\n{doc_reference_section}\n\n"
    "Previously answered questions, for reference only (not necessarily an "
    "exact match to the current message):\n{examples_section}\n\n"
    "Treat a stored answer only as a source of facts -- things that are "
    "true about the candidate -- not as a script to follow. Do not "
    "paraphrase it line by line, do not keep its order, and do not mirror "
    "its structure or the number of points it makes. Take only the facts "
    "that fit the current message and this moment in the conversation -- "
    "often just one or two of them, not all -- and write the reply fresh, "
    "in the persona's own voice, from personality and the reference "
    "material above. If the stored answer and your reply could sit "
    "side by side and read as near-duplicates, rewrite it.\n\n"
    "Conversation history so far -- your own past messages are marked "
    "\"You\". This is the same conversation continuing, so build on it and "
    "do not repeat points already made:\n{history_section}\n\n"
    "Interviewer's current message:\n{user_message}"
)


class PromptBuilder:
    def build(self, user_message: str, context: dict) -> tuple[str, str]:
        """
        Formats gathered context into the final system and user prompt strings.

        Parameters:
        - user_message (str): the user's current message — comes from ModelCollaborateService.model_orchestration
        - context (dict): gathered context materials — comes from ContextGatherer.gather

        Returns:
        - tuple[str, str]: (system_prompt, user_prompt) — goes to ModelCollaborateService._generate_ai_response
        """
        # Labelled to read as reference facts, not a model answer to imitate
        # -- see the "Treat a stored answer only as a source of facts"
        # paragraph in _USER_PROMPT_TEMPLATE.
        examples_section = (
            "\n\n".join(
                f"A similar question asked before: {ex['question']}\n"
                f"Facts the candidate gave then (reuse the facts, not the wording): {ex['answer']}"
                for ex in context["similar_examples"]
            )
            if context["similar_examples"]
            else "No related past answers found."
        )

        # "You" (not the default "Assistant") so the model reads the persona's
        # own past turns as its own messages -- keeps continuity, stops it
        # re-introducing itself. Matches the '"You"' wording in the template.
        history_section = prepare_history(context["recent_messages"], context["summary"], assistant_label="You")

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            candidate_identity=context["candidate_identity"],
            core_personality=context["core_personality"]
        )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            scenario_reference_section=context["scenario_reference_section"],
            doc_reference_section=context["doc_reference_section"],
            examples_section=examples_section,
            history_section=history_section,
            user_message=user_message
        )

        return system_prompt, user_prompt
