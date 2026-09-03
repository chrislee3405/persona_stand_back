import logging
import traceback

from fastapi import BackgroundTasks, Depends

from app.services.consent_service import ConsentService
from app.services.conversation_manage_service import ConversationService, ConversationNotFoundError, ConversationAccessDeniedError
from app.services.model_collarborate_service import ModelCollaborateService
from app.services.model_collarborate.response_gate import is_fallback_response
from app.services.privacy_gate_service import PrivacyGateService
from app.services.rate_control_service import RateControlService, RateTier, get_rate_control_service, TooManyPendingMessagesFromIpError
from app.services.model_collarborate.summarization_service import SummarizationService

logger = logging.getLogger(__name__)

# Max characters allowed in one user message -- generous enough for a
# normal conversational turn, while bounding worst-case cost (a longer
# message means more tokens sent to Gemini and more text for the privacy
# gate's NER pass to scan) and stopping someone from pasting in a massive
# block of text. Mirrored in Chatroom.tsx (MAX_MESSAGE_LENGTH) as an
# <input maxLength> plus a matching client-side check -- same "frontend is
# UX only, this is the real enforcement" relationship as the other gates.
MAX_MESSAGE_LENGTH = 750


class MessageTooLongError(Exception):
    """Raised when user_text exceeds MAX_MESSAGE_LENGTH."""

    def __init__(self, length: int):
        self.length = length
        super().__init__(f"message length {length} exceeds max of {MAX_MESSAGE_LENGTH}")


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
    def __init__(self, conversation_service: ConversationService = Depends(), model_service: ModelCollaborateService = Depends(), summarization_service: SummarizationService = Depends(), privacy_gate_service: PrivacyGateService = Depends(), rate_control_service: RateControlService = Depends(get_rate_control_service), consent_service: ConsentService = Depends()):
        """
        Stores the injected service instances used to run a chat turn.

        Parameters:
        - conversation_service (ConversationService): persists messages/conversations — injected by FastAPI
        - model_service (ModelCollaborateService): generates the AI reply — injected by FastAPI
        - summarization_service (SummarizationService): summarizes conversation history — injected by FastAPI
        - privacy_gate_service (PrivacyGateService): screens raw user text for PII before it reaches append_message or the LLM — injected by FastAPI
        - rate_control_service (RateControlService): caps/paces how often a session reaches append_message or the LLM — injected by FastAPI as the shared singleton (get_rate_control_service), not a fresh instance per request
        - consent_service (ConsentService): verifies this session has agreed to have its input collected before anything else runs — injected by FastAPI

        Returns:
        - None: sets self.conversation_service, self.model_service, self.summarization_service, self.privacy_gate_service, self.rate_control_service, self.consent_service
        """
        self.conversation_service = conversation_service
        self.model_service = model_service
        self.summarization_service = summarization_service
        self.privacy_gate_service = privacy_gate_service
        self.rate_control_service = rate_control_service
        self.consent_service = consent_service

    async def handle_chat_turn(self, session_id: str, code: str | None, conversation_id: str | None, user_text: str, background_tasks: BackgroundTasks, client_ip: str) -> dict:
        """
        Persists a user's message, generates the AI reply, persists it, and schedules summarization.

        Parameters:
        - session_id (str): the caller's session — comes from the router (get_or_create_session_id)
        - code (str | None): invite code to attribute the conversation to, or None for guest — comes from the router
        - conversation_id (str | None): conversation to append to, or None to create one — comes from the router's request body
        - user_text (str): the user's message text — comes from the router's request body
        - background_tasks (BackgroundTasks): task queue — comes from the router, used to schedule summarization after the response is sent
        - client_ip (str): the caller's client IP — comes from the router (get_client_ip), used only for the rate control gate's per-IP backstop (guest sessions only, see below)

        Returns:
        - dict: reply split into display turns, sender, and conversationId — goes back to the router as the response body

        """
        ### message length gate ###
        # Cheapest check, so it runs before anything that costs real work
        # (consent DB lookup, Presidio's NER pass, a Gemini call) -- no
        # reason to spend any of that on a message that's getting rejected
        # anyway. Raises MessageTooLongError (-> HTTP 413 in
        # conversations_router).
        if len(user_text) > MAX_MESSAGE_LENGTH:
            raise MessageTooLongError(len(user_text))

        ### consent gate ###
        # Must run before anything else touches user_text -- raises
        # ConsentRequiredError (-> HTTP 403 in conversations_router) if
        # this session hasn't agreed to the current consent_policy version
        # yet (see ConsentService.get_current_policy).
        self.consent_service.check(session_id)

        ### privacy gate ###
        # Local Presidio check, Raises PrivacyViolationError when privacy found
        self.privacy_gate_service.check(user_text)

        ### rate control gate ###
        # Invite (verified-code) sessions get a more generous tier than
        # guest sessions -- see RateTier/_INTERVAL_SECONDS/
        # _MAX_PENDING_PER_SESSION in rate_control_service.py. `code` is
        # only ever set here for /api/invitechat (see conversations_router).
        tier: RateTier = "invite" if code else "guest"

        # reserve_slot raises TooManyPendingMessagesError (-> HTTP 429) if
        # this session already has too many messages in flight. turn()
        # then waits for this session's spot -- serialized and paced --
        # before anything below runs, holding the session locked until
        # this whole turn (including persisting the reply) finishes, so a
        # later message's history always sees this one's reply already
        # persisted.
        self.rate_control_service.reserve_slot(session_id, tier)

        # The per-IP backstop only applies to guest traffic. It exists to
        # catch cheap session-cycling -- dropping the session cookie to
        # get a fresh guest session with no memory of prior throttling.
        # That bypass doesn't apply to invite sessions: getting one
        # requires a genuinely valid invite code each time, which is
        # trusted here not to be shared/leaked, so invite traffic isn't
        # IP-limited at all -- only its (higher) per-session cap/interval
        # above applies.
        if tier == "guest":
            try:
                self.rate_control_service.reserve_ip_slot(client_ip)
            except TooManyPendingMessagesFromIpError:
                self.rate_control_service.release_slot(session_id)
                raise

        try:
            async with self.rate_control_service.turn(session_id, tier):
                # 1.    Persist the user's message
                #       creating  conversation if conversation_id is None,
                #       verifying conversation_id ownership.
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

                # 2.    Generate the reply via the AI flow
                #       Any failure here is logged as a sender="error" message
                try:
                    reply_text, reply_turns, selected_doc_topics, selected_scenario_topics = await self.model_service.model_orchestration(
                        user_text, conversation_id, session_id, tier
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
                        "sender": "system",
                        "conversationId": conversation_id
                    }

                #       3. Persist the backend's reply. A reply that is the
                #          ResponseGate fallback (every regen attempt still
                #          failed verification) is a system notice, not a
                #          persona turn -- tag it "system" so the frontend
                #          renders it as a centred notice bubble.
                reply_sender = "system" if is_fallback_response(reply_text) else "backend"
                await self.conversation_service.append_message(
                    conversation_id=conversation_id,
                    code=code,
                    session_id=session_id,
                    sender=reply_sender,
                    text=reply_text,
                    selected_document=_join_topics(selected_doc_topics),
                    selected_scenario=_join_topics(selected_scenario_topics)
                )

                # 4.    Now that both the user's message and the backend's reply are persisted,
                #       schedule the summarization threshold check as a background task
                background_tasks.add_task(
                    self.summarization_service.summarize_conversation_if_needed,
                    conversation_id=conversation_id
                )

                return {
                    "turns": reply_turns,
                    "sender": reply_sender,
                    "conversationId": conversation_id  # frontend captures this on first message
                }
        finally:
            # Only release what was actually reserved above -- releasing
            # unconditionally would wrongly decrement another, unrelated
            # guest request's count for this same IP if this request was
            # invite-tier (and so never reserved an IP slot to begin with).
            if tier == "guest":
                self.rate_control_service.release_ip_slot(client_ip)
            self.rate_control_service.release_slot(session_id)
