import asyncio
import logging
import uuid

from fastapi import BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.constants import Sender, TURN_DEADLINE_SECONDS
from app.database import get_db
from app.services.session_work import session_work
from app.services.consent_service import ConsentService
from app.services.conversation_manage_service import ConversationService, ConversationNotFoundError, ConversationAccessDeniedError
from app.services.model_collaborate_service import ModelCollaborateService, TurnOutcome
from app.services.model_collaborate.readiness_service import NO_REPLY, RESPOND
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
        - dict: reply split into display turns, sender, status, conversationId, and userMessageKept — goes back to the router as the response body. `status` is what the readiness gate decided: "respond" (turns carry the reply), "wait" (no reply yet -- the message is held and answered together with the next one), "no_reply" (deliberately unanswered), or "superseded" (a newer message arrived while this turn was generating, so its reply was discarded). Only "respond" ever carries turns; the frontend must not resend the text on any of the other three, because the message is stored and the server is tracking it. userMessageKept is False in the two cases where the message was stored but is not part of the conversation the persona sees: the ResponseGate fallback and a failed or timed-out generation. Both retag every message the turn was answering to Sender.UNANSWERED_USER, so the whole group leaves the prompt history. The frontend reads it to mark those bubbles "Not sent" -- one note for every way a message ends up outside the conversation, since the visitor cannot act on the difference between "never stored" and "stored then dropped".
        """
        # Every gate below rejects the message before it is stored. If this
        # conversation is holding a message the readiness gate said to wait
        # on, that held message was waiting for THIS one -- the continuation
        # the visitor was about to send. A rejection breaks that thought, so
        # the held message is released rather than left pending for whatever
        # the visitor happens to type next. See _discard_pending_group.
        try:
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

        except Exception:
            await self._discard_pending_group(conversation_id, session_id)
            raise

        try:
            await self.db.rollback()  # Do not pin a query connection while queued.
            async with self.rate_control_service.turn(session_id, tier), session_work(session_id):
                # Rotation/withdrawal may have happened while this request waited.
                await self.consent_service.check(session_id)
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
                    outcome = await asyncio.wait_for(
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

                # 2b-4. Persist and shape what the run produced -- shared
                #       with handle_continue_turn, which publishes a run
                #       exactly the same way.
                return await self._publish_outcome(outcome, conversation_id, code, session_id, background_tasks)
        finally:
            # Only release what was actually reserved above -- releasing
            # unconditionally would wrongly decrement another, unrelated
            # guest request's count for this same IP if this request was
            # invite-tier (and so never reserved an IP slot to begin with).
            if tier == "guest":
                self.rate_control_service.release_ip_slot(client_ip)
            self.rate_control_service.release_slot(session_id)

    async def handle_continue_turn(self, session_id: str, code: str | None, conversation_id: str, background_tasks: BackgroundTasks) -> dict:
        """
        Answers a conversation's held messages without a new one: the visitor went quiet after the readiness gate said "wait".

        Parameters:
        - session_id (str): the caller's session — comes from the router (get_or_create_session_id)
        - code (str | None): invite code to attribute the reply to, or None for guest — comes from the router, selects the tier exactly as for a message
        - conversation_id (str): the conversation whose held messages to answer — comes from the router's request body
        - background_tasks (BackgroundTasks): task queue — comes from the router, used to schedule summarization after the response is sent

        Returns:
        - dict: the same shape handle_chat_turn returns. "respond" with the reply when messages were held, or "no_reply" with no turns when nothing was (already answered, or released) -- in which case no model is called at all.

        WHY THIS EXISTS. "wait" is a guess that more is coming, and the only
        thing that proves it wrong is the visitor saying nothing more. The
        frontend asks for this once the input has sat empty and untouched for
        a while after a "wait", so a wrong guess costs the visitor a pause
        rather than a question that is never answered.

        WHAT IT SKIPS, AND WHY THAT IS SAFE.
        - The length and privacy gates: there is no new text. Every held row
          passed both on the way in.
        - The daily quota: each held message already spent its unit when it
          was sent. A continue can only answer messages that were paid for,
          and with nothing held it makes no model call, so repeating it buys
          nothing.
        - The readiness gate: asked again about the same messages it would
          say "wait" again. The visitor's silence is the answer.
        What it keeps: consent (withdrawn consent stops processing of stored
        messages too), ownership, the in-flight cap, the session's turn lock,
        the whole-turn deadline, and the same publishing rules as any turn.

        A REJECTION HERE DOES NOT RELEASE THE HELD GROUP, unlike a rejected
        message. A continue is not the continuation the group was waiting
        for, so failing to run it breaks no thought: the group stays pending
        and is answered together with the visitor's next message. Only a
        failure DURING generation releases it, through _handle_failed_turn,
        for the same reason it does on a normal turn -- the visitor is told
        those messages were not answered.
        """
        ### consent gate ###
        await self.consent_service.check(session_id)

        ### rate control gate ###
        # The in-flight cap only -- see "What it skips" above for the daily
        # quota. No per-IP backstop either: that exists to catch session
        # cycling, and a fresh session holds no messages, so its continue is
        # a no-op before any model call.
        tier: RateTier = "invite" if code else "guest"
        self.rate_control_service.reserve_continue_slot(session_id, tier)

        try:
            await self.db.rollback()
            async with self.rate_control_service.turn(session_id, tier), session_work(session_id):
                # Rotation/withdrawal may have happened while this request waited.
                await self.consent_service.check(session_id)
                # A missing id and a foreign one are both 404 (app/main.py),
                # so a caller cannot probe for other sessions' conversations.
                conversation = await self.conversation_service.get_conversation_unlocked(conversation_id)
                if conversation is None:
                    raise ConversationNotFoundError(conversation_id)
                self.conversation_service.assert_ownership(conversation, session_id)

                if not await self.conversation_service.get_pending_user_messages(conversation_id):
                    logger.info(
                        "Continue for conversation_id=%s found nothing held -- no model call",
                        conversation_id,
                    )
                    return {
                        "turns": [],
                        "sender": Sender.SYSTEM,
                        "status": NO_REPLY,
                        "conversationId": conversation_id,
                        "userMessageKept": True,
                    }

                try:
                    outcome = await asyncio.wait_for(
                        self.model_service.model_orchestration(
                            "", conversation_id, session_id, tier, skip_readiness=True
                        ),
                        timeout=TURN_DEADLINE_SECONDS,
                    )
                except Exception as exc:
                    return await self._handle_failed_turn(
                        exc=exc,
                        user_message=None,
                        conversation_id=conversation_id,
                        code=code,
                        session_id=session_id,
                    )

                return await self._publish_outcome(outcome, conversation_id, code, session_id, background_tasks)
        finally:
            self.rate_control_service.release_slot(session_id)

    async def _publish_outcome(self, outcome: TurnOutcome, conversation_id: str, code: str | None, session_id: str, background_tasks: BackgroundTasks) -> dict:
        """Return a reply only after its complete publication transaction commits."""
        reply_sender = Sender.SYSTEM if is_fallback_response(outcome.reply_text or "") else Sender.BACKEND
        status = await self.conversation_service.publish_turn(
            conversation_id=conversation_id, session_id=session_id, code=code,
            evaluated_through=outcome.evaluated_through, decision=outcome.decision,
            reply_text=outcome.reply_text, reply_sender=reply_sender,
            selected_document=_join_topics(outcome.doc_topics),
            selected_scenario=_join_topics(outcome.scenario_topics),
        )
        if status != RESPOND:
            return {"turns": [], "sender": Sender.SYSTEM, "status": status,
                    "conversationId": conversation_id, "userMessageKept": True}
        background_tasks.add_task(
            self.summarization_service.summarize_conversation_if_needed,
            conversation_id=conversation_id,
        )
        return {"turns": outcome.reply_turns, "sender": reply_sender, "status": RESPOND,
                "conversationId": conversation_id, "userMessageKept": reply_sender != Sender.SYSTEM}

    async def _discard_pending_group(self, conversation_id: str | None, session_id: str) -> None:
        """Rejections wait for active turns before touching their held group."""
        if not conversation_id:
            return
        try:
            # The failed gate may have left a transaction/connection open.
            await self.db.rollback()
            async with session_work(session_id):
                await self.conversation_service.discard_pending_group(conversation_id, session_id)
        except Exception:
            logger.exception("Could not release held messages for conversation_id=%s", conversation_id)

    async def _handle_failed_turn(self, exc: Exception, user_message, conversation_id: str, code: str | None, session_id: str) -> dict:
        """
        Records a turn that failed or ran past its deadline, and builds the response the visitor sees.

        Parameters:
        - exc (Exception): what went wrong — comes from handle_chat_turn's except clause. asyncio.TimeoutError means the turn ran past TURN_DEADLINE_SECONDS; anything else came out of the model pipeline.
        - user_message (Message | None): the visitor's already-persisted message row — comes from handle_chat_turn; None from handle_continue_turn, which appended no row
        - conversation_id (str): the conversation being replied to — comes from handle_chat_turn
        - code (str | None): the caller's invite code, if any — comes from handle_chat_turn
        - session_id (str): the caller's session — comes from handle_chat_turn

        Returns:
        - dict: the same shape a successful turn returns, with a system notice and userMessageKept=False — goes back to the router

        The visitor's whole pending group is released here, not only the
        message this turn appended -- see the call to _discard_pending_group
        below.

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
        # the exception type and sanitized stack locations for a review row
        # without keeping exception content in the database or logs.
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
            detail = type(exc).__name__  # Provider/SQL exception text can contain private input.

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

        # THE WHOLE HELD GROUP GOES, not just the message this turn appended.
        # A failed turn was answering every pending message together, and the
        # visitor is told so: each of those bubbles is marked "Not sent", and
        # the invitation is to send the thought again. Leaving the earlier
        # ones pending would quietly fold them into whatever is typed next
        # and answer them alongside it, minutes later -- while their bubbles
        # said they never arrived. This releases and retags all of them, so
        # what the visitor is looking at is what the server holds.
        await self._discard_pending_group(conversation_id, session_id)

        # A continue turn appended no row of its own; the held group above is
        # everything it was answering.
        if user_message is not None:
            try:
                # Belt and braces for the one row this turn definitely appended.
                # _discard_pending_group above normally covers it, but it is the
                # row whose retag matters most (it is the question that just
                # failed), and it must not depend on a second read succeeding on
                # a session that has already failed once this turn.
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
            # "respond" even though the turn failed: this status answers "is
            # there something in `turns` to show?", and there is -- the
            # apology below. The readiness gate's own three values all mean
            # the server deliberately said nothing, which is a different
            # thing entirely and must not be confused with a failure.
            "status": RESPOND,
            "conversationId": conversation_id,
            # The row was stored and then retagged out of the conversation,
            # along with anything else the turn was answering. The frontend
            # marks them all "Not sent": the persona never saw them and never
            # will, which is the only part the visitor can act on.
            "userMessageKept": False
        }
