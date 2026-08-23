import logging

from fastapi import Depends
from sqlalchemy.orm import Session
from app.services.ai.gemini_service import GeminiService
from app.services.bm25_service import BM25Service
from app.services.conversation_manage_service import ConversationService
from app.database import get_db
from app.models import reference_doc, reference_scenario, reference_personality

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = (
    "Generate a natural language response to the user's message.\n\n"
    "Core personality:\n{core_personality}\n"
)

_USER_PROMPT_TEMPLATE = (
    "Scenario reference:\n{scenario_reference_section}\n\n"
    "Document reference:\n{doc_reference_section}\n\n"
    "Similar past questions and answers to guide tone and content "
    "(not necessarily an exact match):\n{examples_section}\n\n"
    "Conversation history for context:\n{history_section}\n\n"
    "User's current message:\n{user_message}"
)


class ModelCollaborateService:
    def __init__(
        self,
        db: Session = Depends(get_db),
        gemini_service: GeminiService = Depends(),
        bm25_service: BM25Service = Depends(),
        conversation_service: ConversationService = Depends(),
    ):
        """
        Stores the injected database session and service instances.

        Parameters:
        - db (Session): SQLAlchemy session — injected by FastAPI via get_db
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI
        - bm25_service (BM25Service): retrieves similar past questions — injected by FastAPI
        - conversation_service (ConversationService): reads conversation state — injected by FastAPI

        Returns:
        - None: sets self.db, self.gemini_service, self.bm25_service, self.conversation_service
        """
        self.db = db
        self.gemini_service = gemini_service
        self.bm25_service = bm25_service
        self.conversation_service = conversation_service

    async def model_orchestration(self, user_message: str, conversation_id: str) -> tuple[str, list[str], list[str]]:
        """
        Gathers context, builds prompts, and generates the AI reply to a user message.

        Parameters:
        - user_message (str): the user's current message — comes from conversations_router (guestchat/invitechat)
        - conversation_id (str): the conversation being replied to — comes from conversations_router

        Returns:
        - tuple[str, list[str], list[str]]: reply text, selected doc topics, selected scenario topics — goes to conversations_router, which persists the reply and logs the selected topics
        """
        # 1. Gather all necessary references and history
        context = await self._gather_context_materials(user_message, conversation_id)

        # 2. Format the system and user prompts
        system_prompt, user_prompt = self._build_prompts(user_message, context)

        # 3. Call the model to generate the response
        final_response = self._generate_ai_response(system_prompt, user_prompt)

        return final_response, context["doc_topic_list"], context["scenario_topic_list"]

    # --- Refactored Orchestration Helpers ---

    async def _gather_context_materials(self, user_message: str, conversation_id: str) -> dict:
        """
        Collects DB references, similar past questions, and conversation history needed to build a prompt.

        Parameters:
        - user_message (str): the user's current message — comes from model_orchestration
        - conversation_id (str): the conversation being replied to — comes from model_orchestration

        Returns:
        - dict: similar_examples, doc/scenario reference sections, doc/scenario topic lists, core_personality, recent_messages, summary — goes to _build_prompts and model_orchestration
        """
        conversation = self.conversation_service.get_conversation_unlocked(conversation_id)
        summary = conversation.summary if conversation else None

        similar_examples = self.bm25_service.find_similar_questions(user_message, top_k=3)

        # Fetch history first so we can use it to determine topics
        recent_messages = await self.conversation_service.get_recent_messages(conversation_id)
        doc_topic_list, scenario_topic_list = await self.find_topic(user_message, recent_messages)

        doc_reference_section = self._get_doc_references(doc_topic_list)
        scenario_reference_section = self._get_scenario_references(scenario_topic_list)
        core_personality = self._get_core_personality()

        context = {
            "similar_examples": similar_examples,
            "doc_reference_section": doc_reference_section,
            "scenario_reference_section": scenario_reference_section,
            "doc_topic_list": doc_topic_list,
            "scenario_topic_list": scenario_topic_list,
            "core_personality": core_personality,
            "recent_messages": recent_messages,
            "summary": summary
        }

        return context

    def _build_prompts(self, user_message: str, context: dict) -> tuple[str, str]:
        """
        Formats gathered context into the final system and user prompt strings.

        Parameters:
        - user_message (str): the user's current message — comes from model_orchestration
        - context (dict): gathered context materials — comes from _gather_context_materials

        Returns:
        - tuple[str, str]: (system_prompt, user_prompt) — goes to _generate_ai_response
        """
        examples_section = (
            "\n".join(
                f"Q: {ex['question']}\nA: {ex['answer']}"
                for ex in context["similar_examples"]
            )
            if context["similar_examples"]
            else "No similar past examples found."
        )

        history_section = self._prepare_history(context["recent_messages"], context["summary"])

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
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

    def _generate_ai_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends the finalized prompts to Gemini and returns the generated reply.

        Parameters:
        - system_prompt (str): the system instruction — comes from _build_prompts
        - user_prompt (str): the user prompt — comes from _build_prompts

        Returns:
        - str: the model's reply text — goes to model_orchestration
        """
        logger.debug("=== Gemini call: model_orchestration ===")
        logger.debug("System prompt: %s", system_prompt)
        logger.debug("User prompt: %s", user_prompt)

        final_response = self.gemini_service.call_model(
            model_name="gemini-2.5-flash",
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )

        logger.debug("Model response: %s", final_response)
        logger.debug("=== End Gemini call ===")

        return final_response

    # --- DB Helper Method for Core Personality ---

    def _get_core_personality(self) -> str:
        """
        Fetches the core personality text used in every reply's system prompt.

        Parameters:
        - none

        Returns:
        - str: the stored core_personality text from personality_reference (its single/first row — the table has no topic to select between), or a fallback string if the table is empty — goes to _gather_context_materials
        """
        row = self.db.query(reference_personality.PersonalityReference.core_personality).first()
        if row is None:
            logger.warning("personality_reference table is empty — using fallback core personality text.")
            return "No core personality defined."
        return row[0]

    # --- DB Helper Methods for Doc References ---

    def _get_doc_topics(self) -> list[tuple[str, str]]:
        """
        Fetches every document reference topic and its description.

        Parameters:
        - none

        Returns:
        - list[tuple[str, str]]: (document_topic, topic_description) pairs — goes to find_topic
        """
        rows = self.db.query(
            reference_doc.DocReference.document_topic,
            reference_doc.DocReference.topic_description
        ).all()
        return [(row[0], row[1]) for row in rows]

    def _get_doc_from_db(self, topic: str) -> str | None:
        """
        Fetches the stored content for one document reference topic.

        Parameters:
        - topic (str): the document topic to look up — comes from _get_doc_references

        Returns:
        - str | None: the document content, or None if not found — goes to _get_doc_references
        """
        row = (
            self.db.query(reference_doc.DocReference.content)
            .filter(reference_doc.DocReference.document_topic == topic)
            .first()
        )
        return row[0] if row else None

    def _get_doc_references(self, topics: list[str] | None) -> str:
        """
        Builds the formatted document reference section for the prompt from a list of topics.

        Parameters:
        - topics (list[str] | None): document topics to include — comes from _gather_context_materials (via find_topic)

        Returns:
        - str: the concatenated document reference text, or a placeholder if none found — goes to _build_prompts
        """
        if not topics:
            return "No document reference available."

        references = []
        for topic in topics:
            reference = self._get_doc_from_db(topic)
            if reference:
                references.append(f"{topic}:\n{reference}")

        return "\n\n".join(references) if references else "No document reference available."

    # --- DB Helper Methods for Scenario References ---

    def _get_scenario_topics(self) -> list[tuple[str, str]]:
        """
        Fetches every scenario reference topic and its description.

        Parameters:
        - none

        Returns:
        - list[tuple[str, str]]: (scenario_topic, topic_description) pairs — goes to find_topic
        """
        rows = self.db.query(
            reference_scenario.ScenarioReference.scenario_topic,
            reference_scenario.ScenarioReference.topic_description
        ).all()
        return [(row[0], row[1]) for row in rows]

    def _get_scenario_from_db(self, topic: str) -> str | None:
        """
        Fetches the stored content for one scenario reference topic.

        Parameters:
        - topic (str): the scenario topic to look up — comes from _get_scenario_references

        Returns:
        - str | None: the scenario content, or None if not found — goes to _get_scenario_references
        """
        row = (
            self.db.query(reference_scenario.ScenarioReference.content)
            .filter(reference_scenario.ScenarioReference.scenario_topic == topic)
            .first()
        )
        return row[0] if row else None

    def _get_scenario_references(self, topics: list[str] | None) -> str:
        """
        Builds the formatted scenario reference section for the prompt from a list of topics.

        Parameters:
        - topics (list[str] | None): scenario topics to include — comes from _gather_context_materials (via find_topic)

        Returns:
        - str: the concatenated scenario reference text, or a placeholder if none found — goes to _build_prompts
        """
        if not topics:
            return "No scenario reference available."

        references = []
        for topic in topics:
            reference = self._get_scenario_from_db(topic)
            if reference:
                references.append(f"{topic}:\n{reference}")

        return "\n\n".join(references) if references else "No scenario reference available."

    # --- Utility Methods ---

    def _prepare_history(self, recent_messages: list | None, summary: str | None) -> str:
        """
        Formats the conversation summary and recent messages into one history section for the prompt.

        Parameters:
        - recent_messages (list | None): messages since the last summary — comes from _gather_context_materials
        - summary (str | None): the conversation's running summary — comes from _gather_context_materials

        Returns:
        - str: the formatted history text — goes to _build_prompts
        """
        if recent_messages is None:
            logger.warning("model_orchestration received recent_messages=None — treating as no recent messages.")
            recent_messages = []

        summary_part = f"Summary of earlier conversation:\n{summary}" if summary else None
        recent_part = (
            "\n".join(
                f"{'User' if m.sender == 'user' else 'Assistant'}: {m.text}"
                for m in recent_messages
            )
            if recent_messages
            else None
        )

        if summary_part and recent_part:
            return f"{summary_part}\n\nMore recent messages since that summary:\n{recent_part}"

        if summary_part:
            return summary_part

        if recent_part:
            return recent_part

        return "This is a brand new conversation. No recent conversation history available."

    async def find_topic(self, user_message: str, recent_messages: list | None = None) -> tuple[list[str], list[str]]:
        """
        Asks Gemini which document and scenario topics are relevant to the user's message.

        Parameters:
        - user_message (str): the user's current message — comes from _gather_context_materials
        - recent_messages (list | None): recent conversation history for context — comes from _gather_context_materials

        Returns:
        - tuple[list[str], list[str]]: (matched_doc_topics, matched_scenario_topics) — goes to _gather_context_materials
        """
        doc_topics_data = self._get_doc_topics()
        scenario_topics_data = self._get_scenario_topics()

        # Extract up to 4 most recent message pairs (8 messages) for context
        history_context = ""
        if recent_messages:
            context_msgs = recent_messages[-8:]
            history_str = "\n".join(
                f"{'User' if m.sender == 'user' else 'Assistant'}: {m.text}"
                for m in context_msgs
            )
            history_context = f"Recent conversation history for context:\n{history_str}\n\n"

        ### Find Doc Topics ###
        _valid_doc_topics = []
        if doc_topics_data:
            all_doc_topics = [t[0] for t in doc_topics_data]
            doc_descriptions_str = "\n".join(f"- {t[0]}: {t[1]}" for t in doc_topics_data)

            doc_system_prompt = (
                "Your task is to identify which fact document topics are relevant to the user's message. "
                "Return only the exact topic(s) from the provided list that are relevant."
            )

            doc_user_prompt = (
                "The following topics represent different fact documents that may be useful to reply to the user message. "
                "Here are the topics and their descriptions:\n"
                f"{doc_descriptions_str}\n\n"
                f"{history_context}"
                f"Current User message: {user_message}\n\n"
                "Using the conversation history strictly as context to understand references, select the topic(s) "
                "that are most relevant to answering the Current User message."
            )

            doc_schema = {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                    "enum": all_doc_topics
                }
            }

            doc_topic_response = self.gemini_service.call_model_structured(
                model_name="gemini-2.5-flash",
                user_prompt=doc_user_prompt,
                system_prompt=doc_system_prompt,
                schema=doc_schema
            )

            _valid_doc_topics = []
            if isinstance(doc_topic_response, list):
                _valid_doc_topics = [topic for topic in doc_topic_response if topic in all_doc_topics]
            else:
                logger.debug("find_topic doc search returned %r, expected a list", doc_topic_response)

        ### Find Scenario Topics ###
        _valid_scenario_topics = []
        if scenario_topics_data:
            all_scenario_topics = [t[0] for t in scenario_topics_data]
            scenario_descriptions_str = "\n".join(f"- {t[0]}: {t[1]}" for t in scenario_topics_data)

            scenario_system_prompt = (
                "Your task is to identify which scenario approaches are relevant to the user's message. "
                "Return only the exact topic(s) from the provided list that are relevant."
            )

            scenario_user_prompt = (
                "The following topics represent different scenario approaches that may be relevant to reply to the user message. "
                "Here are the topics and their descriptions:\n"
                f"{scenario_descriptions_str}\n\n"
                f"{history_context}"
                f"Current User message: {user_message}\n\n"
                "Using the conversation history strictly as context to understand references, select the topic(s) "
                "that are most relevant to answering the Current User message."
            )

            scenario_schema = {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                    "enum": all_scenario_topics
                }
            }

            scenario_topic_response = self.gemini_service.call_model_structured(
                model_name="gemini-2.5-flash",
                user_prompt=scenario_user_prompt,
                system_prompt=scenario_system_prompt,
                schema=scenario_schema
            )

            _valid_scenario_topics = []
            if isinstance(scenario_topic_response, list):
                _valid_scenario_topics = [topic for topic in scenario_topic_response if topic in all_scenario_topics]
            else:
                logger.debug("find_topic scenario search returned %r, expected a list", scenario_topic_response)

        return _valid_doc_topics, _valid_scenario_topics