import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Literal

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.constants import (
    MAX_DAILY_CONSENT_SUBMISSIONS_PER_IP,
    MAX_DAILY_INVITE_CODE_FAILURES,
    MAX_DAILY_INVITE_CODE_FAILURES_PER_IP,
    RATE_LIMIT_RETENTION_DAYS,
)
from app.models.rate_limit import RateLimitCounter

logger = logging.getLogger(__name__)

# A session is either "guest" (/api/guestchat, code=None) or "invite"
# (/api/invitechat, a verified code) -- see ChatService.handle_chat_turn,
# which derives this from whether `code` is set and passes it into every
# call below. Session-level limits are split by tier so a verified/known
# user isn't held to the same, more defensive limits set for an anonymous
# guest. The per-IP backstop (MAX_PENDING_PER_IP) below is deliberately
# guest-only rather than split by tier -- it exists specifically to catch
# cheap session-cycling (dropping the session cookie for a fresh guest
# session with no memory of prior throttling), which doesn't apply to
# invite sessions: each invite code is issued to one named company, so it
# is trusted not to be shared. The GLOBAL ceiling that bounds total spend
# regardless of tier is the daily quota, which is now durable -- see
# RateLimitCounter and _consume_daily below.
RateTier = Literal["guest", "invite"]

# Hyperparameters -- the one place to tune these.
# Minimum time between the start of two consecutive message-turns for the
# same session, in seconds, per tier, WHILE MORE THAN ONE MESSAGE IS IN
# FLIGHT. Note the qualifier: release_slot drops a session's pacing cursor
# once its last in-flight message finishes (it has to, or the cursor map
# grows for every session the process ever sees), so this paces a burst and
# does not floor the gap between two turns sent one after the other. Bursts
# are what it was for -- a turn takes seconds anyway.
_INTERVAL_SECONDS: dict[RateTier, float] = {
    "guest": 3.0,
    "invite": 1.5,
}

# How many messages a single session may have in flight (sent, reply not
# yet fully returned) at once, per tier. A message beyond this is rejected
# outright (TooManyPendingMessagesError -> HTTP 429 in
# conversations_router) instead of queuing indefinitely -- mirrors a real
# conversation, where stacking more than a few unanswered messages in a
# row stops making sense. Invite sessions get a higher cap than guests for
# the same reason as the interval above. Mirrored client-side in
# useChatDispatch.ts (MAX_PENDING_MESSAGES) for instant UI feedback, but
# this is the actual enforcement point -- the client-side check is
# trivially bypassable.
_MAX_PENDING_PER_SESSION: dict[RateTier, int] = {
    "guest": 2,
    "invite": 5,
}

# How many messages a single IP address may have in flight at once,
# across guest sessions only -- a backstop against a script that defeats
# _MAX_PENDING_PER_SESSION["guest"] by simply dropping its session cookie
# between requests (a fresh session has no memory of prior messages, see
# get_or_create_session_id). Deliberately more generous than the
# per-session cap: multiple genuine guests can share one public IP (NAT,
# office/campus network, mobile carrier), so this only needs to catch
# throughput far beyond what shared-IP legitimate traffic would produce.
# Unlike turn() below, this is cap-only, not paced/serialized -- pacing per
# IP would wrongly queue unrelated users' conversations behind each other
# whenever they happen to share an IP.
_MAX_PENDING_PER_IP = 3

# How many messages a single session may send PER CALENDAR DAY, per tier,
# and how many a single IP may send per day across guest sessions.
#
# The three limits above are concurrency and pacing only -- they cap how
# FAST messages arrive, never how MANY. Each message costs 6-11 Gemini
# calls, so these two are the only thing bounding spend.
#
# They are now stored in Postgres (RateLimitCounter), not in this process's
# memory. That matters for more than restarts: keyed on a session id, an
# in-memory counter was not a limit at all, because a caller mints a fresh
# session id by dropping its cookie. The durable version does not change
# that for `session:` keys -- nothing can -- which is why the `ip:` quota
# below is the one that actually bounds a single machine, and why it now
# survives the restarts that used to reset it.
#
# Sized for what this app is: a portfolio a small number of recruiters and
# hiring managers will each open once or twice. 60 turns is far more
# conversation than any genuine visit needs.
_DAILY_QUOTA_PER_SESSION: dict[RateTier, int] = {
    "guest": 60,
    "invite": 200,
}

# Deliberately looser than the per-session figure for the same reason
# _MAX_PENDING_PER_IP is: several genuine visitors can share one public IP.
# This only has to catch throughput far beyond what shared-IP legitimate
# traffic produces.
_DAILY_QUOTA_PER_IP = 300


# --- Counter key scopes ---------------------------------------------------
# `RateLimitCounter.key` is `<scope>:<value>`; see that model's docstring for
# why it is one prefixed column rather than two.
_KEY_SESSION = "session:{}"
_KEY_IP = "ip:{}"
_KEY_CODE_FAIL_IP = "code_fail_ip:{}"
_KEY_CODE_FAIL_GLOBAL = "code_fail:global"
_KEY_CONSENT_IP = "consent_ip:{}"


class TooManyPendingMessagesError(Exception):
    """Raised when a session already has _MAX_PENDING_PER_SESSION[tier] messages in flight."""

    def __init__(self, session_id: str, tier: RateTier):
        self.session_id = session_id
        self.tier = tier
        super().__init__(f"session {session_id} ({tier}) already has {_MAX_PENDING_PER_SESSION[tier]} messages in flight")


class TooManyPendingMessagesFromIpError(Exception):
    """Raised when a client IP already has _MAX_PENDING_PER_IP messages in flight, across guest sessions (invite traffic is never checked against this -- see RateTier)."""

    def __init__(self, ip: str):
        self.ip = ip
        super().__init__(f"ip {ip} already has {_MAX_PENDING_PER_IP} messages in flight")


class DailyQuotaExceededError(Exception):
    """Raised when a session or an IP has used its whole day's message allowance (_DAILY_QUOTA_PER_SESSION / _DAILY_QUOTA_PER_IP)."""

    def __init__(self, key: str, limit: int):
        self.key = key
        self.limit = limit
        super().__init__(f"{key} has used its daily allowance of {limit} messages")


class InviteVerificationLockedError(Exception):
    """
    Raised when invite-code verification is closed for the rest of the day,
    because failed attempts have hit their per-IP or global daily ceiling
    (MAX_DAILY_INVITE_CODE_FAILURES_PER_IP / MAX_DAILY_INVITE_CODE_FAILURES).

    Fails CLOSED: a genuine invite holder is locked out too until the counter
    rolls over. That is the intended trade -- an invite holder can email
    instead, and a guessed code is unmetered LLM spend.
    """

    def __init__(self, scope: str, limit: int):
        self.scope = scope
        self.limit = limit
        super().__init__(f"invite-code verification locked: {scope} reached {limit} failed attempts today")


class RateControlService:
    """
    Throttles how often messages reach append_message/the LLM, and how often
    an invite code may be guessed.

    Two kinds of state, kept in two different places on purpose:

    DURABLE, in Postgres (RateLimitCounter) -- everything counted PER DAY:
      - the per-session and per-IP message allowances
      - failed invite-code verifications, per IP and globally
      - POST /api/consent submissions, per IP
    These are the limits that bound total volume and total spend, so they
    have to survive a restart and be correct across workers. Each increment
    is one atomic INSERT ... ON CONFLICT DO UPDATE, so there is no
    read-then-write race anywhere below.

    IN-PROCESS, in the dicts below -- everything about CONCURRENCY:
      - how many messages a session (or IP) has in flight right now
      - the per-session asyncio lock that serialises a session's turns
      - that session's pacing cursor
    These describe requests currently being handled by THIS process, so a
    process-local answer is the correct one; there is nothing for another
    worker to know. All three are bounded: release_slot deletes a session's
    entry from every one of them once its last in-flight message finishes,
    so none of them retains anything about a session that is not currently
    talking. No unbounded in-memory counter remains.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_allowed_time: dict[str, float] = {}
        self._pending_counts: dict[str, int] = {}
        self._ip_pending_counts: dict[str, int] = {}
        # The last day this process ran the retention sweep. Not a counter --
        # one date, so that the sweep runs once per process per day instead of
        # on every request.
        self._last_swept_day: date | None = None

    # --- Durable daily counters ------------------------------------------

    async def _consume_daily(self, db: AsyncSession, key: str, limit: int) -> int:
        """
        Atomically counts one unit against a rate-limit key for today, and reports whether that put it over the limit.

        Parameters:
        - db (AsyncSession): the caller's session -- comes from the route's Depends(get_db)
        - key (str): the `<scope>:<value>` counter key -- comes from the _KEY_* templates above
        - limit (int): the day's ceiling for this key -- comes from the caller

        Returns:
        - int: the key's new count for today -- goes to the caller, which compares it against `limit`

        ONE STATEMENT, not read-then-write. `INSERT ... ON CONFLICT DO UPDATE
        SET count = count + 1 RETURNING count` is evaluated by Postgres under a
        row lock, so two concurrent requests for the same key are serialised by
        the database and cannot both read the same old value. A read followed
        by an update would let two turns past a ceiling of one.

        The increment happens BEFORE the limit is known, so a rejected request
        still counts -- and the count keeps climbing past `limit` rather than
        pinning to it. Both are deliberate: a caller hammering a limit is
        exactly what the owner wants visible when reviewing these rows later,
        and pinning would need the read-then-write this avoids.
        """
        await self._sweep_expired_counters(db)

        today = date.today()
        statement = (
            pg_insert(RateLimitCounter)
            .values(key=key, day=today, count=1)
            .on_conflict_do_update(
                constraint="uq_rate_limit_key_day",
                set_={
                    "count": RateLimitCounter.__table__.c.count + 1,
                    "updated_at": func.now(),
                },
            )
            .returning(RateLimitCounter.__table__.c.count)
        )
        result = await db.execute(statement)
        new_count = result.scalar_one()
        # Committed immediately rather than riding along with whatever the
        # request commits later: a quota unit must be spent even if the turn
        # it was spent on goes on to fail, and a rolled-back turn must not
        # refund it.
        await db.commit()
        return new_count

    async def _peek_daily(self, db: AsyncSession, key: str) -> int:
        """
        Reads a rate-limit key's count for today without changing it.

        Parameters:
        - db (AsyncSession): the caller's session -- comes from the route's Depends(get_db)
        - key (str): the `<scope>:<value>` counter key -- comes from the _KEY_* templates above

        Returns:
        - int: today's count for this key, or 0 if there is no row yet -- goes to the caller.
          Used by the invite-code circuit breaker, which must decide whether to
          process an attempt BEFORE knowing if it will fail; only a failure
          increments (see record_invite_code_failure).
        """
        result = await db.execute(
            select(RateLimitCounter.count).where(
                RateLimitCounter.key == key,
                RateLimitCounter.day == date.today(),
            )
        )
        return result.scalar_one_or_none() or 0

    async def _sweep_expired_counters(self, db: AsyncSession) -> None:
        """
        Deletes rate-limit rows older than RATE_LIMIT_RETENTION_DAYS, at most once per process per day.

        Parameters:
        - db (AsyncSession): the caller's session -- comes from the route's Depends(get_db)

        Returns:
        - None: removes expired rows, or does nothing if this process has already swept today

        The rows are kept after their day rolls over so the owner can review
        traffic after the fact -- but `key` contains per-session and per-IP
        values, so the table would otherwise grow forever. Swept from a request
        rather than a cron because this app has no scheduler; it is one DELETE
        over an indexed column, once a day, and `_last_swept_day` is set before
        the statement runs so a burst of concurrent first-requests-of-the-day
        cannot each fire one.
        """
        today = date.today()
        if self._last_swept_day == today:
            return
        self._last_swept_day = today

        cutoff = today - timedelta(days=RATE_LIMIT_RETENTION_DAYS)
        result = await db.execute(delete(RateLimitCounter).where(RateLimitCounter.day < cutoff))
        await db.commit()
        if result.rowcount:
            logger.info(
                "rate control swept %d rate_limit_counter rows older than %s",
                result.rowcount, cutoff
            )

    # --- Per-turn slots ---------------------------------------------------

    def _reserve_pending(self, session_id: str, tier: RateTier) -> None:
        """
        Claims one of this session's in-flight slots, in process memory.

        Parameters:
        - session_id (str): the caller's session -- comes from reserve_slot
        - tier (RateTier): "guest" or "invite" -- comes from reserve_slot, selects which cap in _MAX_PENDING_PER_SESSION applies

        Returns:
        - None: raises TooManyPendingMessagesError if the session is already at its cap; otherwise increments the count
        """
        count = self._pending_counts.get(session_id, 0)
        if count >= _MAX_PENDING_PER_SESSION[tier]:
            logger.info("rate control rejected a message for session=%s (%s): already %d in flight", session_id, tier, count)
            raise TooManyPendingMessagesError(session_id, tier)
        self._pending_counts[session_id] = count + 1

    async def reserve_slot(self, db: AsyncSession, session_id: str, tier: RateTier) -> None:
        """
        Claims one of this session's in-flight slots and one unit of its daily allowance.

        Parameters:
        - db (AsyncSession): the request's database session -- comes from ChatService
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn
        - tier (RateTier): "guest" or "invite" -- comes from ChatService.handle_chat_turn (based on whether `code` is set), selects which cap in _MAX_PENDING_PER_SESSION and _DAILY_QUOTA_PER_SESSION applies

        Returns:
        - None: raises TooManyPendingMessagesError if the session is already at its in-flight cap, DailyQuotaExceededError if it has used the day's allowance; otherwise both are claimed

        The in-flight cap is checked FIRST because it is a dictionary lookup
        and the daily quota is a database round trip -- no reason to spend the
        round trip on a request that is being refused anyway. If the quota then
        rejects, the in-flight slot taken a line earlier is handed back, since
        nothing is going to release it.

        The daily unit is CONSUMED, NOT RESERVED -- release_slot does not give
        it back. A quota measures how much a caller has asked the app to do
        today, and a message rejected further down the pipeline still asked.
        """
        self._reserve_pending(session_id, tier)
        try:
            limit = _DAILY_QUOTA_PER_SESSION[tier]
            count = await self._consume_daily(db, _KEY_SESSION.format(session_id), limit)
            if count > limit:
                logger.info(
                    "rate control rejected a message for session=%s (%s): daily allowance of %d used (attempt %d)",
                    session_id, tier, limit, count
                )
                raise DailyQuotaExceededError(f"session {session_id} ({tier})", limit)
        except Exception:
            self.release_slot(session_id)
            raise

    def release_slot(self, session_id: str) -> None:
        """
        Releases one of this session's in-flight slots once its turn has fully finished (reply persisted, or the turn failed).

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn

        Returns:
        - None: decrements the in-flight count for session_id, and DELETES the key once it reaches zero. The daily count is untouched -- see reserve_slot.

        The key is removed rather than left at 0, and the session's lock and
        pacing cursor go with it. These dicts previously only ever grew: every
        visitor the process ever saw cost one asyncio.Lock, one float and one
        int for the life of the process, and an attacker minting sessions
        filled them for free. Deleting the lock is safe here only because
        this runs after the `async with lock` block in turn() has exited --
        ChatService calls release_slot from its outermost `finally`, and a
        later message from the same session simply creates a fresh lock via
        setdefault. A session with a SECOND message still parked in turn()
        has a pending count above zero, so this returns early and touches
        none of the three.
        """
        count = self._pending_counts.get(session_id, 0)
        remaining = max(0, count - 1)
        if remaining:
            self._pending_counts[session_id] = remaining
            return

        self._pending_counts.pop(session_id, None)
        self._next_allowed_time.pop(session_id, None)
        lock = self._locks.get(session_id)
        # Only drop the lock if nothing is holding or waiting on it. A second
        # message from this session can be parked in turn() right now.
        if lock is not None and not lock.locked():
            self._locks.pop(session_id, None)

    async def reserve_ip_slot(self, db: AsyncSession, ip: str) -> None:
        """
        Claims one of this IP's in-flight slots and one unit of its daily allowance -- independent of, and in addition to, reserve_slot's per-session limits. Tier-agnostic itself; ChatService.handle_chat_turn only calls this for "guest" tier (see RateTier).

        Parameters:
        - db (AsyncSession): the request's database session -- comes from ChatService
        - ip (str): the caller's client IP -- comes from ChatService.handle_chat_turn (get_client_ip)

        Returns:
        - None: raises TooManyPendingMessagesFromIpError if this IP is already at _MAX_PENDING_PER_IP, DailyQuotaExceededError if it has used the day's allowance; otherwise both are claimed. Same check-order and same consumed-not-reserved rule as reserve_slot.
        """
        count = self._ip_pending_counts.get(ip, 0)
        if count >= _MAX_PENDING_PER_IP:
            logger.info("rate control rejected a message for ip=%s: already %d in flight", ip, count)
            raise TooManyPendingMessagesFromIpError(ip)
        self._ip_pending_counts[ip] = count + 1

        try:
            count = await self._consume_daily(db, _KEY_IP.format(ip), _DAILY_QUOTA_PER_IP)
            if count > _DAILY_QUOTA_PER_IP:
                logger.info(
                    "rate control rejected a message for ip=%s: daily allowance of %d used (attempt %d)",
                    ip, _DAILY_QUOTA_PER_IP, count
                )
                raise DailyQuotaExceededError(f"ip {ip}", _DAILY_QUOTA_PER_IP)
        except Exception:
            self.release_ip_slot(ip)
            raise

    def release_ip_slot(self, ip: str) -> None:
        """
        Releases one of this IP's in-flight slots once the triggering turn has fully finished (reply persisted, or the turn failed). Must only be called if reserve_ip_slot was actually called for the same request (i.e. "guest" tier) -- calling it unconditionally would wrongly decrement an unrelated guest request's count for the same IP.

        Parameters:
        - ip (str): the caller's client IP -- comes from ChatService.handle_chat_turn (get_client_ip)

        Returns:
        - None: decrements the in-flight count for ip, and deletes the key once it reaches zero (same reasoning as release_slot). The daily count is untouched.
        """
        count = self._ip_pending_counts.get(ip, 0)
        remaining = max(0, count - 1)
        if remaining:
            self._ip_pending_counts[ip] = remaining
        else:
            self._ip_pending_counts.pop(ip, None)

    # --- Invite-code brute-force protection -------------------------------

    async def assert_invite_verification_allowed(self, db: AsyncSession, ip: str) -> None:
        """
        Checks whether invite-code verification is still open today, before an attempt is processed.

        Parameters:
        - db (AsyncSession): the request's database session -- comes from codes_router
        - ip (str): the caller's client IP -- comes from codes_router (get_client_ip)

        Returns:
        - None: raises InviteVerificationLockedError (-> HTTP 429) if this IP, or the app as a whole, has reached its daily failed-attempt ceiling. Otherwise returns silently and the caller proceeds to check the code.

        Two layers, both required. The per-IP ceiling stops one machine burning
        the global allowance on its own. The GLOBAL ceiling is the one that
        actually bounds brute force: per-IP alone is not a brute-force control
        at all, because an attacker with a rotating proxy pool just spreads the
        guesses across addresses, and every address gets a fresh allowance.

        Read, not incremented. Only a FAILED verification counts (see
        record_invite_code_failure), and whether this attempt fails is not
        known yet.
        """
        ip_failures = await self._peek_daily(db, _KEY_CODE_FAIL_IP.format(ip))
        if ip_failures >= MAX_DAILY_INVITE_CODE_FAILURES_PER_IP:
            logger.warning(
                "invite-code verification locked for ip=%s: %d failed attempts today (cap %d)",
                ip, ip_failures, MAX_DAILY_INVITE_CODE_FAILURES_PER_IP
            )
            raise InviteVerificationLockedError(f"ip {ip}", MAX_DAILY_INVITE_CODE_FAILURES_PER_IP)

        global_failures = await self._peek_daily(db, _KEY_CODE_FAIL_GLOBAL)
        if global_failures >= MAX_DAILY_INVITE_CODE_FAILURES:
            logger.warning(
                "invite-code verification locked GLOBALLY: %d failed attempts today (cap %d)",
                global_failures, MAX_DAILY_INVITE_CODE_FAILURES
            )
            raise InviteVerificationLockedError("global", MAX_DAILY_INVITE_CODE_FAILURES)

    async def record_invite_code_failure(self, db: AsyncSession, ip: str) -> None:
        """
        Counts one failed invite-code verification against both the per-IP and the global daily ceilings.

        Parameters:
        - db (AsyncSession): the request's database session -- comes from codes_router
        - ip (str): the caller's client IP -- comes from codes_router (get_client_ip)

        Returns:
        - None: increments both counters. Never raises on the caller's behalf -- the caller is already returning an error for the failed code; the next attempt is the one that gets refused by assert_invite_verification_allowed.

        Only failures reach here. A successful verification must not count, or
        a company working through its own invite would walk the app toward a
        lockout it did nothing to deserve.
        """
        await self._consume_daily(db, _KEY_CODE_FAIL_IP.format(ip), MAX_DAILY_INVITE_CODE_FAILURES_PER_IP)
        total = await self._consume_daily(db, _KEY_CODE_FAIL_GLOBAL, MAX_DAILY_INVITE_CODE_FAILURES)
        if total >= MAX_DAILY_INVITE_CODE_FAILURES:
            logger.warning(
                "invite-code failures reached the global daily cap (%d/%d) -- verification is now closed until tomorrow",
                total, MAX_DAILY_INVITE_CODE_FAILURES
            )

    # --- Consent submissions ----------------------------------------------

    async def consume_consent_submission(self, db: AsyncSession, ip: str) -> None:
        """
        Counts one POST /api/consent against this IP's daily ceiling.

        Parameters:
        - db (AsyncSession): the request's database session -- comes from consent_router
        - ip (str): the caller's client IP -- comes from consent_router (get_client_ip)

        Returns:
        - None: raises DailyQuotaExceededError (which consent_router maps to 429) once the IP is over MAX_DAILY_CONSENT_SUBMISSIONS_PER_IP

        That endpoint takes no credential and inserts a consent_record row for
        every previously-unseen session id -- and a caller mints a fresh
        session id by dropping its cookie. Without this it is an
        unauthenticated, unbounded row-insertion endpoint pointed at the same
        RDS volume everything else depends on.
        """
        count = await self._consume_daily(
            db, _KEY_CONSENT_IP.format(ip), MAX_DAILY_CONSENT_SUBMISSIONS_PER_IP
        )
        if count > MAX_DAILY_CONSENT_SUBMISSIONS_PER_IP:
            logger.warning(
                "rate control rejected a consent submission for ip=%s: %d today (cap %d)",
                ip, count, MAX_DAILY_CONSENT_SUBMISSIONS_PER_IP
            )
            raise DailyQuotaExceededError(f"ip {ip} consent submissions", MAX_DAILY_CONSENT_SUBMISSIONS_PER_IP)

    # --- Turn serialisation -----------------------------------------------

    @asynccontextmanager
    async def turn(self, session_id: str, tier: RateTier):
        """
        Scopes one message's turn for a session: waits for the session's lock (serializing its turns to one at a time, in arrival order -- so a later message's conversation history always includes an earlier one's already-persisted reply), then sleeps off whatever remains of _INTERVAL_SECONDS[tier] since the previous turn started, before yielding control to the caller. The lock stays held for the caller's entire `async with` block, so the next queued message can't start until this one -- including persisting its reply -- is done.

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn
        - tier (RateTier): "guest" or "invite" -- comes from ChatService.handle_chat_turn (based on whether `code` is set), selects which interval in _INTERVAL_SECONDS applies

        Returns:
        - None (async context manager): yields once this turn is clear to proceed; releases the session's lock on exit
        """
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            scheduled_time = max(now, self._next_allowed_time.get(session_id, now))
            self._next_allowed_time[session_id] = scheduled_time + _INTERVAL_SECONDS[tier]

            wait_seconds = scheduled_time - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            yield


# Module-level singleton -- the in-flight counts, the per-session asyncio
# locks and the pacing cursors must persist across requests to mean anything,
# so this must not be reconstructed per-request the way FastAPI's Depends()
# would if handed the class directly. get_rate_control_service (below) hands
# out this same instance every time instead. The DURABLE counters do not
# depend on this being a singleton -- they live in Postgres and are reached
# through whichever AsyncSession the caller passes in.
_rate_control_service = RateControlService()


def get_rate_control_service() -> RateControlService:
    """
    FastAPI dependency provider returning the shared RateControlService singleton.

    Parameters:
    - none

    Returns:
    - RateControlService: the process-wide instance, so per-session in-flight state persists across requests -- goes to ChatService via Depends(get_rate_control_service)
    """
    return _rate_control_service
