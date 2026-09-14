import asyncio
import logging
import uuid

from fastapi import BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.constants import Sender, TURN_DEADLINE_SECONDS
from app.database import get_db
from app.services.consent_service import ConsentService
from app.services.conversation_manage_service import ConversationService, ConversationNotFoundError, ConversationAccessDeniedError
from app.services.model_collaborate_service import ModelCollaborateService
from app.services.model_collaborate.response_gate import is_fallback_response
from app.services.privacy_gate_service import PrivacyGateService
from app.services.rate_control_service import RateControlService, RateTier, get_rate_control_service
from app.services.model_collaborate.summarization_service import SummarizationService

logger = logging.getLogger(__name__)

# Max characters allowed in one user message -- generous enough for a
# normal conversational turn, while bounding worst-case cost (a longer
# message means more tokens sent to Gemini and more text for the privacy
# gate's NER pass to scan) and stopping someone from pasting in a massive
# block of text. Mirrored in useChatDispatch.ts (MAX_MESSAGE_LENGTH) as an
# <input maxLength> plus a matching client-side check -- same "frontend is
# UX only, this is the real enforcement" relationship as the other gates.
MAX_MESSAGE_LENGTH = 750

# How much of a failed turn's exception message to keep on the Sender.ERROR
# review row. The row used to hold `traceback.format_exc()` in full, which is
# the wrong thing to put in this table: a SQLAlchemy StatementError renders
# the failing statement AND its bound parameters, so a failure at or after
# append_message wrote the visitor's message and the assembled prompt into a
# SECOND row -- in a table with no retention policy, in a column nothing
# distinguishes from an ordinary message. The full traceback still goes to the
# log, where its lifetime is a container-runtime concern; what is persisted is
# an identifier, a type and a bounded excerpt.
_ERROR_DETAIL_MAX_CHARS = 300


class MessageTooLongError(Exception):
    """Raised when user_text exceeds MAX_MESSAGE_LENGTH."""

    def __init__(self, length: int):
        self.length = length
        super().__init__(f"message length {length} exceeds max of {MAX_MESSAGE_LENGTH}")


def _join_topics(topics: list[str] | None) -> str | None:
    """
    Joins multiple matched topics into one comma-separated string, since selected_document/selected_scenario are singular String columns.

    Parameters:
    - topics (list[str] | None): topics chosen by ContextGatherer._find_topic — comes from ChatService.handle_chat_turn's call to model_orchestration, which returns them

    Returns:
    - str | None: comma-joined topics, or None if empty/missing — goes into Message.selected_document / Message.selected_scenario via ConversationService.append_message
    """
    if not topics:
        return None
    return ", ".join(topics)


class ChatService:
    def __init__(self, db: AsyncSession = Depends(get_db), conversation_service: ConversationService = Depends(), model_service: ModelCollaborateService = Depends(), summarization_service: SummarizationService = Depends(), privacy_gate_service: PrivacyGateService = Depends(), rate_control_service: RateControlService = Depends(get_rate_control_service), consent_service: ConsentService = Depends()):
        """
        Stores the injected service instances used to run a chat turn.

        Parameters:
        - db (AsyncSession): SQLAlchemy async session — injected by FastAPI via get_db. Held here because the durable rate-limit counters live in Postgres now and RateControlService is a process-wide singleton, so it cannot hold a per-request session of its own.
        - conversation_service (ConversationService): persists messages/conversations — injected by FastAPI
        - model_service (ModelCollaborateService): generates the AI reply — injected by FastAPI
        - summarization_service (SummarizationService): summarizes conversation history — injected by FastAPI
        - privacy_gate_service (PrivacyGateService): screens raw user text for PII before it reaches append_message or the LLM — injected by FastAPI
        - rate_control_service (RateControlService): caps/paces how often a session reaches append_message or the LLM — injected by FastAPI as the shared singleton (get_rate_control_service), not a fresh instance per request
        - consent_service (ConsentService): verifies this session has agreed to have its input collected before anything else runs — injected by FastAPI

        Returns:
        - None: sets self.db and the six service attributes
        """
        self.db = db
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
        - dict: reply split into display turns, sender, conversationId, and userMessageKept — goes back to the router as the response body. userMessageKept is False in the two cases where the message was stored but is not part of the conversation the persona sees: the ResponseGate fallback (_drop_withheld_turns discards the user's message along with the notice), and a failed or timed-out generation (the row is retagged Sender.UNANSWERED_USER). The frontend reads it to mark that bubble as not answered.
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
        await self.consent_service.check(session_id)

        ### privacy gate ###
        # Local Presidio check. Raises PrivacyViolationError when PII is found.
        #
        # OFFLOADED TO A THREAD, not awaited inline. _analyzer.analyze() is a
        # full spaCy NER pass plus ~18 recognizers over up to 750 characters --
        # entirely CPU-bound, entirely synchronous. This handler is `async
        # def`, so FastAPI runs it ON the event loop; calling the analyzer
        # directly meant the single uvicorn worker could serve nothing else
        # for its duration, including GET /api/site-content for a visitor who
        # is merely loading the home page.
        await run_in_threadpool(self.privacy_gate_service.check, user_text)

        ### rate control gate ###
        # Invite (verified-code) sessions get a more generous tier than
        # guest sessions -- see RateTier/_INTERVAL_SECONDS/
        # _MAX_PENDING_PER_SESSION in rate_control_service.py. `code` is
        # only ever set here for /api/invitechat (see conversations_router).
        tier: RateTier = "invite" if code else "guest"

        # reserve_slot raises TooManyPendingMessagesError (-> HTTP 429) if
        # this session already has too many messages in flight, and
        # DailyQuotaExceededError if it has used the day's allowance. The
        # daily half is a Postgres counter now, so it survives restarts and is
        # correct across workers. turn() below then waits for this session's
        # spot -- serialized and paced -- before anything else runs, holding
        # the session locked until this whole turn (including persisting the
        # reply) finishes, so a later message's history always sees this one's
        # reply already persisted.
        await self.rate_control_service.reserve_slot(self.db, session_id, tier)

        # The per-IP backstop only applies to guest traffic. It exists to
        # catch cheap session-cycling -- dropping the session cookie to
        # get a fresh guest session with no memory of prior throttling.
        # That bypass doesn't apply to invite sessions: getting one
        # requires a genuinely valid invite code each time, and each code is
        # issued to one named company, so it is trusted not to be shared.
        if tier == "guest":
            try:
                await self.rate_control_service.reserve_ip_slot(self.db, client_ip)
            except Exception:
                # Any failure here means this request is not proceeding, so the
                # session slot claimed just above has to go back -- nothing else
                # will release it, because the try/finally below has not been
                # entered yet. Deliberately broad: the two rate-limit errors are
                # the expected cases, but reserve_ip_slot now talks to the
                # database, and a connection failure there must not leak a slot
                # for the rest of the process's life. The session's DAILY unit is
                # deliberately not refunded -- see RateControlService.reserve_slot.
                self.rate_control_service.release_slot(session_id)
                raise

        try:
            async with self.rate_control_service.turn(session_id, tier):
                # 1.    Persist the user's message
                #       creating  conversation if conversation_id is None,
                #       verifying conversation_id ownership.
                try:
                    user_message, conversation_id = await self.conversation_service.append_message(
                        conversation_id=conversation_id,
                        code=code,
                        session_id=session_id,
                        sender=Sender.USER,
                        text=user_text
                    )
                except (ConversationNotFoundError, ConversationAccessDeniedError):
                    # Transparently start a new conversation rather than
                    # erroring -- the id the client sent is stale or was never
                    # theirs, and either way there is a message to answer.
                    # Logged because the silent version hid a real failure
                    # mode: a frontend that stops adopting the returned
                    # conversationId opens a NEW conversation per message, the
                    # persona forgets everything every turn, and the only
                    # trace is a growing pile of one-message rows.
                    logger.warning(
                        "conversation_id=%s is unusable for session=%s (missing, or owned by another session) "
                        "-- starting a new conversation for this turn",
                        conversation_id, session_id,
                    )
                    user_message, conversation_id = await self.conversation_service.append_message(
                        conversation_id=None,
                        code=code,
                        session_id=session_id,
                        sender=Sender.USER,
                        text=user_text
                    )

                # 2.    Generate the reply via the AI flow, under a whole-turn
                #       deadline. See TURN_DEADLINE_SECONDS: without it the
                #       per-call ceiling bounds a turn at ~360s, nginx gives up
                #       at 120s, and this process goes on to COMMIT a reply the
                #       visitor was already told had failed. Timing out here
                #       instead means the failure is one we control -- and the
                #       branch below guarantees no Sender.BACKEND row is
                #       written, so nothing from a timed-out turn can reach the
                #       next prompt's history.
                #       Any failure is recorded as a sender="error" review row.
                try:
                    reply_text, reply_turns, selected_doc_topics, selected_scenario_topics = await asyncio.wait_for(
                        self.model_service.model_orchestration(
                            user_text, conversation_id, session_id, tier
                        ),
                        timeout=TURN_DEADLINE_SECONDS,
                    )
                except Exception as exc:
                    return await self._handle_failed_turn(
                        exc=exc,
                        user_message=user_message,
                        conversation_id=conversation_id,
                        code=code,
                        session_id=session_id,
                    )

                #       3. Persist the backend's reply. A reply that is the
                #          ResponseGate fallback (every regen attempt still
                #          failed verification) is a system notice, not a
                #          persona turn -- tag it "system" so the frontend
                #          renders it as a centred notice bubble.
                reply_sender = Sender.SYSTEM if is_fallback_response(reply_text) else Sender.BACKEND
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
                #       schedule the summarization threshold check as a background task.
                #       It opens its own database session -- this request's is
                #       already closed by the time background tasks run.
                background_tasks.add_task(
                    self.summarization_service.summarize_conversation_if_needed,
                    conversation_id=conversation_id
                )

                return {
                    "turns": reply_turns,
                    "sender": reply_sender,
                    "conversationId": conversation_id,  # frontend captures this on first message
                    # A "system" reply is the ResponseGate fallback, and
                    # ConversationService._drop_withheld_turns removes it
                    # TOGETHER WITH the user message it answered -- so that
                    # message is persisted but is no longer part of the
                    # conversation. The frontend cannot infer this from
                    # `sender` alone (the error path above is also "system"
                    # yet keeps its user message), so say it outright.
                    "userMessageKept": reply_sender != Sender.SYSTEM
                }
        finally:
            # Only release what was actually reserved above -- releasing
            # unconditionally would wrongly decrement another, unrelated
            # guest request's count for this same IP if this request was
            # invite-tier (and so never reserved an IP slot to begin with).
            if tier == "guest":
                self.rate_control_service.release_ip_slot(client_ip)
            self.rate_control_service.release_slot(session_id)

    async def _handle_failed_turn(self, exc: Exception, user_message, conversation_id: str, code: str | None, session_id: str) -> dict:
        """
        Records a turn that failed or ran past its deadline, and builds the response the visitor sees.

        Parameters:
        - exc (Exception): what went wrong — comes from handle_chat_turn's except clause. asyncio.TimeoutError means the turn ran past TURN_DEADLINE_SECONDS; anything else came out of the model pipeline.
        - user_message (Message): the visitor's already-persisted message row — comes from handle_chat_turn
        - conversation_id (str): the conversation being replied to — comes from handle_chat_turn
        - code (str | None): the caller's invite code, if any — comes from handle_chat_turn
        - session_id (str): the caller's session — comes from handle_chat_turn

        Returns:
        - dict: the same shape a successful turn returns, with a system notice and userMessageKept=False — goes back to the router

        NO Sender.BACKEND ROW IS WRITTEN on this path, which is the whole
        point of routing the timeout through it: a turn that nginx has already
        given up on must not leave a persona reply in the history that the
        visitor never saw. The visitor's own message stays in the table -- the
        owner needs to see what was being asked when it failed -- but it is
        retagged out of Sender.USER so the next prompt does not read it back as
        a question the persona has already been asked and dealt with. Nothing
        ever answered it.

        Every write here is individually guarded. If the failure was a database
        failure, the session is already unusable and these writes raise too;
        letting that escape would replace a handled turn with a bare 500 and
        lose the response entirely. A visitor who gets the apology is better
        served than one who gets nothing.
        """
        # One id, in the log line and on the stored row, so the owner can find
        # the full traceback for a given review row without keeping tracebacks
        # in the database.
        incident_id = uuid.uuid4().hex[:12]
        timed_out = isinstance(exc, asyncio.TimeoutError)

        # Put the session back in a usable state before writing anything.
        # Two ways it can arrive here dirty: the failure WAS a database error
        # (so there is a failed transaction to clear), or asyncio.wait_for
        # cancelled model_orchestration mid-statement -- ResponseGate commits a
        # review row on every rejected attempt, so a deadline can land inside
        # one. Without this the two writes below fail on a session that is
        # merely unrolled-back, and the turn loses both its review row and its
        # retag for no good reason.
        try:
            await self.db.rollback()
        except Exception:
            logger.exception("[%s] could not roll back the session before recording the failure", incident_id)

        if timed_out:
            logger.error(
                "[%s] chat turn exceeded TURN_DEADLINE_SECONDS=%.0fs for conversation_id=%s -- "
                "abandoning it before nginx does, so no reply is persisted",
                incident_id, TURN_DEADLINE_SECONDS, conversation_id,
            )
            detail = f"turn exceeded the {TURN_DEADLINE_SECONDS:.0f}s deadline"
        else:
            logger.exception(
                "[%s] model_orchestration failed for conversation_id=%s", incident_id, conversation_id
            )
            detail = f"{type(exc).__name__}: {str(exc)[:_ERROR_DETAIL_MAX_CHARS]}"

        try:
            await self.conversation_service.append_message(
                conversation_id=conversation_id,
                code=code,
                session_id=session_id,
                sender=Sender.ERROR,
                text=f"[incident {incident_id}] {detail}",
            )
        except Exception:
            logger.exception("[%s] could not persist the error review row", incident_id)

        try:
            await self.conversation_service.retag_message_sender(
                user_message, Sender.UNANSWERED_USER
            )
        except Exception:
            logger.exception(
                "[%s] could not retag the user message to %s -- it will be read back as part of "
                "the conversation on the next turn",
                incident_id, Sender.UNANSWERED_USER,
            )

        return {
            "turns": [
                "Sorry, that took too long to answer. Please try again."
                if timed_out
                else "Sorry, something went wrong while generating a response. Please try again."
            ],
            "sender": Sender.SYSTEM,
            "conversationId": conversation_id,
            # Retagged above, so it is no longer part of the conversation the
            # persona sees. The frontend renders this as "not answered" rather
            # than "not sent", which is exactly right: the server did receive
            # and store it, and then failed to reply to it.
            "userMessageKept": False
        }
