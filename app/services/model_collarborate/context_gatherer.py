import logging

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.ai.gemini_service import GeminiService
from app.services.bm25_service import BM25Service
from app.services.conversation_manage_service import ConversationService
from app.models.prompt_reference import DocReference, PersonalityReference, ScenarioReference

logger = logging.getLogger(__name__)

_FIND_TOPIC_SYSTEM_PROMPT_TEMPLATE = (
    "Your task is to identify which {topic_kind} topics are relevant to the "
    "user's current message.\n\n"
    "Be conservative: only select a topic if you are highly confident it is "
    "directly relevant to answering the current message -- a loose, "
    "tangential, or merely thematically-similar connection is not enough. "
    "If no topic clears that bar, return an empty list rather than guessing "
    "or including a weak match.\n\n"
    "Return only the exact topic string(s) from the provided list -- never "
    "invent a topic that isn't listed."
)

_FIND_TOPIC_USER_PROMPT_TEMPLATE = (
    "The following topics represent different {topic_kind} that may be "
    "useful to reply to the user's message. Topics and their descriptions:\n"
    "{topic_descriptions}\n\n"
    "{history_context}"
    "Current user message: {user_message}\n\n"
    "Using the conversation history strictly as context to understand "
    "references (not as a source of topics on its own), select only the "
    "topic(s) you are highly confident are directly relevant to answering "
    "the current message. When in doubt, leave it out."
)


class ContextGatherer:
    def __init__(self, db: Session = Depends(get_db), gemini_service: GeminiService = Depends(), bm25_service: BM25Service = Depends(), conversation_service: ConversationService = Depends()):
        """
        Stores the injected database session and service instances.

        Parameters:
        - db (Session): SQLAlchemy session — injected by FastAPI via get_db, or passed explicitly by ModelCollaborateService
        - gemini_service (GeminiService): calls the Gemini model — injected by FastAPI, or passed explicitly
        - bm25_service (BM25Service): retrieves similar past questions — injected by FastAPI, or passed explicitly
        - conversation_service (ConversationService): reads conversation state — injected by FastAPI, or passed explicitly

        Returns:
        - None: sets self.db, self.gemini_service, self.bm25_service, self.conversation_service
        """
        self.db = db
        self.gemini_service = gemini_service
        self.bm25_service = bm25_service
        self.conversation_service = conversation_service

    async def gather(self, user_message: str, conversation_id: str) -> dict:
        """
        Collects DB references, similar past questions, and conversation history needed to build a prompt.

        Parameters:
        - user_message (str): the user's current message — comes from ModelCollaborateService.model_orchestration
        - conversation_id (str): the conversation being replied to — comes from ModelCollaborateService.model_orchestration

        Returns:
        - dict: similar_examples, doc/scenario reference sections, doc/scenario topic lists, candidate_identity, core_personality, recent_messages, summary — goes to PromptBuilder.build and ModelCollaborateService.model_orchestration
        """
        conversation = self.conversation_service.get_conversation_unlocked(conversation_id)
        summary = conversation.summary if conversation else None

        similar_examples = self.bm25_service.find_similar_questions(user_message, top_k=1)

        # Fetch history first so we can use it to determine topics
        recent_messages = await self.conversation_service.get_recent_messages(conversation_id)
        doc_topic_list, scenario_topic_list = await self._find_topic(user_message, recent_messages)

        doc_reference_section = self._get_doc_references(doc_topic_list)
        scenario_reference_section = self._get_scenario_references(scenario_topic_list)
        candidate_identity, core_personality = self._get_personality_profile()

        context = {
            "similar_examples": similar_examples,
            "doc_reference_section": doc_reference_section,
            "scenario_reference_section": scenario_reference_section,
            "doc_topic_list": doc_topic_list,
            "scenario_topic_list": scenario_topic_list,
            "candidate_identity": candidate_identity,
            "core_personality": core_personality,
            "recent_messages": recent_messages,
            "summary": summary
        }

        return context

    # --- DB Helper Method for Personality Profile ---

    def _get_personality_profile(self) -> tuple[str, str]:
        """
        Fetches the single personality_reference row and builds the candidate identity block plus the core personality text.

        Parameters:
        - none

        Returns:
        - tuple[str, str]: (candidate_identity, core_personality) built from personality_reference's single/first row (the table has no topic to select between), or fallback strings if the table is empty — goes to gather
        """
        row = self.db.query(PersonalityReference).first()
        if row is None:
            logger.warning("personality_reference table is empty — using fallback identity and personality text.")
            return (
                "Name: <not configured -- add a row to personality_reference>",
                "No core personality defined."
            )

        candidate_identity = (
            f"Legal name: {row.legal_name}\n"
            f"Preferred name: {row.prefer_name}\n"
            f"Cultural background: {row.cluture_background}"
        )
        return candidate_identity, row.core_personality

    # --- DB Helper Methods for Doc References ---

    def _get_doc_topics(self) -> list[tuple[str, str]]:
        """
        Fetches every document reference topic and its description.

        Parameters:
        - none

        Returns:
        - list[tuple[str, str]]: (document_topic, topic_description) pairs — goes to _find_topic
        """
        rows = self.db.query(
            DocReference.document_topic,
            DocReference.topic_description
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
            self.db.query(DocReference.content)
            .filter(DocReference.document_topic == topic)
            .first()
        )
        return row[0] if row else None

    def _get_doc_references(self, topics: list[str] | None) -> str:
        """
        Builds the formatted document reference section for the prompt from a list of topics.

        Parameters:
        - topics (list[str] | None): document topics to include — comes from gather (via _find_topic)

        Returns:
        - str: the concatenated document reference text, or a placeholder if none found — goes to gather
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
        - list[tuple[str, str]]: (scenario_topic, topic_description) pairs — goes to _find_topic
        """
        rows = self.db.query(
            ScenarioReference.scenario_topic,
            ScenarioReference.topic_description
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
            self.db.query(ScenarioReference.content)
            .filter(ScenarioReference.scenario_topic == topic)
            .first()
        )
        return row[0] if row else None

    def _get_scenario_references(self, topics: list[str] | None) -> str:
        """
        Builds the formatted scenario reference section for the prompt from a list of topics.

        Parameters:
        - topics (list[str] | None): scenario topics to include — comes from gather (via _find_topic)

        Returns:
        - str: the concatenated scenario reference text, or a placeholder if none found — goes to gather
        """
        if not topics:
            return "No scenario reference available."

        references = []
        for topic in topics:
            reference = self._get_scenario_from_db(topic)
            if reference:
                references.append(f"{topic}:\n{reference}")

        return "\n\n".join(references) if references else "No scenario reference available."

    # --- Topic Selection ---

    def _select_relevant_topics(self, topic_kind: str, topics_data: list[tuple[str, str]], history_context: str, user_message: str) -> list[str]:
        """
        Asks Gemini to conservatively select which topics are relevant to the user's message, requiring high confidence rather than any loose relation.

        Parameters:
        - topic_kind (str): human-readable label for what's being matched (e.g. "fact document", "scenario approach") — comes from _find_topic
        - topics_data (list[tuple[str, str]]): (topic, description) pairs to choose from — comes from _find_topic
        - history_context (str): formatted recent-history block, or "" if none — comes from _find_topic
        - user_message (str): the user's current message — comes from _find_topic

        Returns:
        - list[str]: topics Gemini is highly confident are relevant, filtered to only those actually in topics_data — goes to _find_topic
        """
        if not topics_data:
            return []

        all_topics = [t[0] for t in topics_data]
        # Each topic and its description are unambiguously labeled and
        # blank-line-separated -- topic_description never restates the topic
        # name itself, so a terser "- topic: description" one-liner would
        # rely purely on the model correctly parsing colon placement to keep
        # pairs straight, especially with many topics listed at once.
        descriptions_str = "\n\n".join(f"Topic: {t[0]}\nDescription: {t[1]}" for t in topics_data)

        system_prompt = _FIND_TOPIC_SYSTEM_PROMPT_TEMPLATE.format(topic_kind=topic_kind)
        user_prompt = _FIND_TOPIC_USER_PROMPT_TEMPLATE.format(
            topic_kind=topic_kind,
            topic_descriptions=descriptions_str,
            history_context=history_context,
            user_message=user_message
        )
        schema = {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": all_topics
            }
        }

        response = self.gemini_service.call_model_structured(
            model_name="gemini-3.5-flash-lite",
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=schema
        )

        if not isinstance(response, list):
            logger.debug("_find_topic (%s) returned %r, expected a list", topic_kind, response)
            return []

        return [topic for topic in response if topic in all_topics]

    async def _find_topic(self, user_message: str, recent_messages: list | None = None) -> tuple[list[str], list[str]]:
        """
        Asks Gemini which document and scenario topics are relevant to the user's message.

        Parameters:
        - user_message (str): the user's current message — comes from gather
        - recent_messages (list | None): recent conversation history for context — comes from gather

        Returns:
        - tuple[list[str], list[str]]: (matched_doc_topics, matched_scenario_topics) — goes to gather
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

        doc_topics = self._select_relevant_topics("fact document", doc_topics_data, history_context, user_message)
        scenario_topics = self._select_relevant_topics("scenario approach", scenario_topics_data, history_context, user_message)

        return doc_topics, scenario_topics
