import json
import logging

from fastapi import Depends
from sqlalchemy.orm import Session
from app.services.ai.gemini_service import GeminiService
from app.services.bm25_service import BM25Service
from app.services.conversation_manage_service import ConversationService
from app.services.summarization_service import SummarizationService
from app.database import get_db
from app.models import model_reference as model_reference_models

logger = logging.getLogger(__name__)

RECENT_MESSAGES_BEFORE_SUMMARIZE = 10

# Instructions and developer/admin-curated reference material only —
# nothing user-originated goes here. Topic reference (model_reference)
# and BM25 examples (question_bank) are admin-curated content, not raw
# user input, so they stay in the system channel; conversation history
# is user-originated and goes in the user prompt instead (see
# _USER_PROMPT_TEMPLATE below) so it can't inherit the system channel's
# elevated instruction-following weight — see the prompt-injection
# discussion this split resolves.
_SYSTEM_PROMPT_TEMPLATE = (
    "Generate a natural language response to the user's message.\n\n"
    "Topic reference:\n{reference_section}\n\n"
    "Similar past questions and answers to guide tone and content "
    "(not necessarily an exact match):\n{examples_section}"
)

# Conversation history (summary + recent messages) is user-originated —
# it belongs alongside the actual message being responded to, not mixed
# into the system instruction channel.
_USER_PROMPT_TEMPLATE = (
    "Conversation history for context:\n{history_section}\n\n"
    "User's current message:\n{user_message}"
)


class ModelCollaborateService:
    def __init__(self, db: Session = Depends(get_db), gemini_service: GeminiService = Depends(), bm25_service: BM25Service = Depends(), conversation_service: ConversationService = Depends(), summarization_service: SummarizationService = Depends()):
        """
        Store the injected database session and the GeminiService, BM25Service, ConversationService, and SummarizationService instances onto the ModelCollaborateService instance attributes.
        Params: db (Session), gemini_service (GeminiService), bm25_service (BM25Service), conversation_service (ConversationService), summarization_service (SummarizationService)
        Returns: None
        """
        self.db = db
        self.gemini_service = gemini_service
        self.bm25_service = bm25_service
        self.conversation_service = conversation_service
        self.summarization_service = summarization_service

    async def route_user_message(self, user_message: str, conversation_id: str) -> str:
        """
        Use the user's message and conversation id to fetch topic classification from Gemini, similar Q&A pairs from the database question_bank table, and recent messages from the database message table, send them all to Gemini to generate the reply text, then trigger the summarization service once the recent message count from the database reaches the threshold.
        Params: user_message (str), conversation_id (str)
        Returns: str - the generated reply text
        """

        ### FIRST PART - gather material ###
        topic_response = await self.define_topic(user_message)
        similar_examples = self.bm25_service.find_similar_questions(user_message, top_k=3)
        recent_messages = await self.conversation_service.get_recent_messages(conversation_id)
        conversation = self.conversation_service.get_conversation_unlocked(conversation_id)
        summary = conversation.summary if conversation else None

        # recent_messages always includes THIS turn's user message as its
        # last entry -- the router (conversations_router.py) persists it
        # via append_message before route_user_message ever runs. Exclude
        # it from what generate_dialogue shows as "history", since it's
        # already passed separately as user_message there; including it
        # in both would show the model the same message twice under two
        # different labels. The full recent_messages (current message
        # included) is still used below for the summarization trigger and
        # summarize_conversation, since that needs the complete set to
        # compute the correct checkpoint.
        history_for_prompt = recent_messages[:-1]

        ### SECOND PART - generate response with essential material ###
        final_response = await self.generate_dialogue(
            user_message, topic_response, similar_examples, history_for_prompt, summary
        )

        ### SUMMARIZATION ###
        if len(recent_messages) >= RECENT_MESSAGES_BEFORE_SUMMARIZE:
            await self.summarization_service.summarize_conversation(
                conversation_id, summary, recent_messages
            )

        return final_response

    def _get_topics(self) -> list[str]:
        """
        Read every topic name from the database model_reference table.
        Params: none
        Returns: list[str] - all known topic names
        """
        rows = self.db.query(model_reference_models.ModelReference.topic).all()
        return [row[0] for row in rows]

    def _get_reference(self, topic: str) -> dict | list | None:
        """
        Use the given topic name to look up the matching content field from the database model_reference table.
        Params: topic (str)
        Returns: dict | list | None - the topic's content, or None if no row matches
        """
        row = (
            self.db.query(model_reference_models.ModelReference.content)
            .filter(model_reference_models.ModelReference.topic == topic)
            .first()
        )
        return row[0] if row else None

    async def define_topic(self, user_message: str) -> str | None:
        """
        Send the user's message and the topic names from the database model_reference table to the Gemini model, constrained to return exactly one of those topic names.
        Params: user_message (str)
        Returns: str | None - a topic name from the known list, or None if no topics are configured or the result didn't match
        """
        topics = self._get_topics()

        if not topics:
            system_prompt = (
                "It suppose to have a topic list provided but missing, response directly to user topic list is None"
            )
            logger.debug("=== Gemini call: define_topic (no topics configured, unconstrained) ===")
            logger.debug("System prompt: %s", system_prompt)
            logger.debug("User prompt: %s", user_message)

            response = self.gemini_service.call_model(
                model_name="gemini-2.5-flash",
                user_prompt=user_message,
                system_prompt=system_prompt
            )

            logger.debug("Model response: %s", response)
            logger.debug("=== End Gemini call ===")
            return None

        topic_list_str = ", ".join(topics)
        system_prompt = (
            "Define the topic of the user message based on the "
            f"following provided list: {topic_list_str}"
        )
        schema = {
            "type": "STRING",
            "enum": topics
        }

        logger.debug("=== Gemini call: define_topic (structured) ===")
        logger.debug("System prompt: %s", system_prompt)
        logger.debug("User prompt: %s", user_message)
        logger.debug("Schema: %s", schema)

        topic_response = self.gemini_service.call_model_structured(
            model_name="gemini-2.5-flash",
            user_prompt=user_message,
            system_prompt=system_prompt,
            schema=schema
        )

        logger.debug("Model response: %s", topic_response)
        logger.debug("=== End Gemini call ===")

        if topic_response not in topics:
            logger.debug(
                "define_topic returned %r, not in known topics %r — treating as no match",
                topic_response, topics
            )
            return None

        return topic_response

    async def generate_dialogue(self, user_message: str, topic: str | None, examples: list[dict], recent_messages: list | None, summary: str | None) -> str:
        """
        Use the topic content from the database model_reference table and the BM25-matched examples from the database question_bank table to build the system prompt, and use the recent messages plus summary from the database conversation and message tables together with the user's message to build the user prompt, then send both to the Gemini model.
        Params: user_message (str), topic (str | None), examples (list[dict]), recent_messages (list | None), summary (str | None)
        Returns: str - the model's generated response
        """
        reference = self._get_reference(topic) if topic else None

        reference_section = (
            json.dumps(reference) if reference
            else "No topic reference available."
        )

        examples_section = (
            "\n".join(f"Q: {ex['question']}\nA: {ex['answer']}" for ex in examples)
            if examples
            else "No similar past examples found."
        )

        if recent_messages is None:
            logger.warning(
                "generate_dialogue received recent_messages=None (expected "
                "a list, even if empty) — treating as no recent messages."
            )
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
            history_section = (
                f"{summary_part}\n\nMore recent messages since that summary:\n{recent_part}"
            )
        elif summary_part:
            history_section = summary_part
        elif recent_part:
            history_section = recent_part
        else:
            history_section = "This is a brand new conversation. No recent conversation history available."

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            reference_section=reference_section,
            examples_section=examples_section
        )
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            history_section=history_section,
            user_message=user_message
        )

        logger.debug("=== Gemini call: generate_dialogue ===")
        logger.debug("System prompt: %s", system_prompt)
        logger.debug("User prompt: %s", user_prompt)

        response = self.gemini_service.call_model(
            model_name="gemini-2.5-flash",
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )

        logger.debug("Model response: %s", response)
        logger.debug("=== End Gemini call ===")

        return response