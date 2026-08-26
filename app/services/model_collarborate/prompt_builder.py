from app.services.model_collarborate.prepare_history import prepare_history

_SYSTEM_PROMPT_TEMPLATE = (
    "Role: you are an AI persona representing the job candidate identified "
    "below in a professional, text-based interview with an IT interviewer. "
    "Stay fully in character as the candidate for the entire conversation -- "
    "never break character, reveal that you are an AI model, or refer to "
    "yourself as an assistant.\n\n"
    "Candidate identity:\n{candidate_identity}\n\n"
    "Task: generate the candidate's next natural language response to the "
    "interviewer's message.\n\n"
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
    "Similar past questions and answers (not necessarily an exact match to "
    "the current message):\n{examples_section}\n\n"
    "When a stored question closely matches the current one, use its answer "
    "primarily to identify what kind of information should be covered in "
    "your response -- not just as a tone or phrasing reference. Still build "
    "the actual wording from personality and the reference material above, "
    "rather than reusing the stored answer's phrasing verbatim.\n\n"
    "Conversation history for context:\n{history_section}\n\n"
    "User's current message:\n{user_message}"
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
        examples_section = (
            "\n".join(
                f"Q: {ex['question']}\nA: {ex['answer']}"
                for ex in context["similar_examples"]
            )
            if context["similar_examples"]
            else "No similar past examples found."
        )

        history_section = prepare_history(context["recent_messages"], context["summary"])

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
