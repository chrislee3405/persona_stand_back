import logging
import traceback

from fastapi import BackgroundTasks, Depends

from app.services.conversation_manage_service import ConversationService, ConversationNotFoundError, ConversationAccessDeniedError
from app.services.model_collarborate_service import ModelCollaborateService
from app.services.summarization_service import SummarizationService

logger = logging.getLogger(__name__)


def _join_topics(topics: list[str] | None) -> str | None:
    """
    Joins multiple matched topics into one comma-separated string, since selected_document/selected_scenario are singular String columns.

    Parameters:
    - topics (list[str] | None): topics matched by ModelCollaborateService.find_topic — comes from ChatService.handle_chat_turn's call to model_orchestration

    Returns:
    - str | None: comma-joined topics, or None if empty/missing — goes into Message.selected_document / Message.selected_scenario via ConversationService.append_message
    """
    if not topics:
        return None
    return ", ".join(topics)


class ChatService:
    def __init__(self, conversation_service: ConversationService = Depends(), model_service: ModelCollaborateService = Depends(), summarization_service: SummarizationService = Depends()):
        """
        Stores the injected service instances used to run a chat turn.

        Parameters:
        - conversation_service (ConversationService): persists messages/conversations — injected by FastAPI
        - model_service (ModelCollaborateService): generates the AI reply — injected by FastAPI
        - summarization_service (SummarizationService): summarizes conversation history — injected by FastAPI

        Returns:
        - None: sets self.conversation_service, self.model_service, self.summarization_service
        """
        self.conversation_service = conversation_service
        self.model_service = model_service
        self.summarization_service = summarization_service

    async def handle_chat_turn(self, session_id: str, code: str | None, conversation_id: str | None, user_text: str, background_tasks: BackgroundTasks) -> dict:
        """
        Persists a user's message, generates the AI reply, persists it, and schedules summarization.

        Parameters:
        - session_id (str): the caller's session — comes from the router (get_or_create_session_id)
        - code (str | None): invite code to attribute the conversation to, or None for guest — comes from the router
        - conversation_id (str | None): conversation to append to, or None to create one — comes from the router's request body
        - user_text (str): the user's message text — comes from the router's request body
        - background_tasks (BackgroundTasks): task queue — comes from the router, used to schedule summarization after the response is sent

        Returns:
        - dict: reply split into display turns, sender, and conversationId — goes back to the router as the response body

        """
        ### rate control gate ###
        # TODO implement a rate control gate to prevent user send too much request
        # remember to implement to the frontend as well (slightly shorter than frontend for better UX)

        ### privacy gate ###
        # TODO implement a local privacy gate to do the first filter to see if the user_text contain sensitive information
        # After first filter pass the user_text to AI to verify if containing privacy or not

        # 1. Persist the user's message, creating the conversation first if
        #    conversation_id is None, or verifying ownership if it isn't. A
        #    stale/tampered conversation_id (not found, or not owned by this
        #    session) isn't treated as an error -- silently fall back to
        #    starting a brand new conversation instead, same as if
        #    conversation_id had been None all along.
        try:
            _, conversation_id = await self.conversation_service.append_message(
                conversation_id=conversation_id,
                code=code,
                session_id=session_id,
                sender="user",
                text=user_text
            )
        except (ConversationNotFoundError, ConversationAccessDeniedError):
            _, conversation_id = await self.conversation_service.append_message(
                conversation_id=None,
                code=code,
                session_id=session_id,
                sender="user",
                text=user_text
            )

        # 2. Generate the reply via the AI flow, plus which doc/scenario
        #    topics were actually selected to produce it. Any failure here
        #    (Gemini error, malformed structured response, etc.) is logged
        #    as a sender="error" message for later review instead of
        #    500ing -- the user's message above stays persisted either way.
        #    get_recent_messages excludes sender="error" rows from both
        #    summarization and future prompt history, so a failed turn
        #    never shows up as a fake prior reply.
        try:
            reply_text, reply_turns, selected_doc_topics, selected_scenario_topics = await self.model_service.model_orchestration(
                user_text, conversation_id, session_id
            )
        except Exception:
            logger.exception("model_orchestration failed for conversation_id=%s", conversation_id)
            await self.conversation_service.append_message(
                conversation_id=conversation_id,
                code=code,
                session_id=session_id,
                sender="error",
                text=traceback.format_exc()
            )
            return {
                "turns": ["Sorry, something went wrong while generating a response. Please try again."],
                "sender": "backend",
                "conversationId": conversation_id
            }

        # 3. Persist the backend's reply, along with which topics were
        #    selected to generate it. Always the full, unsplit reply_text --
        #    reply_turns is display-only, for the frontend to reveal as
        #    several sequential bubbles; the database keeps one row with
        #    the complete original text for the app owner to review.
        await self.conversation_service.append_message(
            conversation_id=conversation_id,
            code=code,
            session_id=session_id,
            sender="backend",
            text=reply_text,
            selected_document=_join_topics(selected_doc_topics),
            selected_scenario=_join_topics(selected_scenario_topics)
        )

        # 4. Now that both the user's message and the backend's reply are
        #    persisted, schedule the summarization threshold check as a
        #    background task -- non-blocking, and summarizes the complete
        #    exchange.
        background_tasks.add_task(
            self.summarization_service.summarize_conversation_if_needed,
            conversation_id=conversation_id
        )

        return {
            "turns": reply_turns,
            "sender": "backend",
            "conversationId": conversation_id  # frontend captures this on first message
        }
