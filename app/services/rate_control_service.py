import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

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
# invite sessions: re-minting one requires a genuinely valid invite code
# each time, and invite codes are trusted here not to be shared/leaked. So
# ChatService.handle_chat_turn only calls reserve_ip_slot/release_ip_slot
# for "guest" tier -- invite traffic relies solely on its (higher)
# per-session cap/interval above.
RateTier = Literal["guest", "invite"]

# Hyperparameters -- the one place to tune these.
# Minimum time between the start of two consecutive message-turns for the
# same session, in seconds, per tier. Throttles how often a session can
# trigger a Gemini call, to bound cost from a flooded/scripted burst of
# messages -- without ever refusing to accept a message outright (that's
# what MAX_PENDING_PER_SESSION below is for). Invite sessions get a
# shorter interval than guests since they're a known/verified user, not
# just anyone who opened the page.
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
# Chatroom.tsx (MAX_PENDING_MESSAGES) for instant UI feedback, but this is
# the actual enforcement point -- the client-side check is trivially
# bypassable.
_MAX_PENDING_PER_SESSION: dict[RateTier, int] = {
    "guest": 2,
    "invite": 5,
}

# How many messages a single IP address may have in flight at once,
# across guest sessions only (see the RateTier comment above for why
# invite traffic isn't checked against this at all) -- a backstop against
# a script that defeats _MAX_PENDING_PER_SESSION["guest"] by simply
# dropping its session cookie between requests (a fresh session has no
# memory of prior messages, see get_or_create_session_id). Deliberately
# more generous than the per-session cap: multiple genuine guests can
# share one public IP (NAT, office/campus network, mobile carrier), so
# this only needs to catch throughput far beyond what shared-IP
# legitimate traffic would produce. Unlike turn() below, this is cap-only,
# not paced/serialized -- pacing per IP would wrongly queue unrelated
# users' conversations behind each other whenever they happen to share an
# IP.
_MAX_PENDING_PER_IP = 3

# How many messages a single session may send PER CALENDAR DAY, per tier,
# and how many a single IP may send per day across guest sessions.
#
# The three limits above are concurrency and pacing only -- they cap how
# FAST messages arrive, never how MANY. Nothing stopped a caller sending one
# message every _INTERVAL_SECONDS forever, and each message costs 6-11
# Gemini calls, so sustained scripted traffic had no ceiling at all except
# the accidental one imposed by the reply pipeline being synchronous (now
# fixed -- see GeminiService, which no longer blocks the event loop, so this
# quota has to do the work that bug was doing by accident).
#
# Sized for what this app is: a portfolio a small number of recruiters and
# hiring managers will each open once or twice. 60 turns is far more
# conversation than any genuine visit needs, and roughly 400 model calls a
# day per session at the current per-turn count.
_DAILY_QUOTA_PER_SESSION: dict[RateTier, int] = {
    "guest": 60,
    "invite": 200,
}

# Deliberately looser than the per-session figure for the same reason
# _MAX_PENDING_PER_IP is: several genuine visitors can share one public IP
# (NAT, office or campus network, mobile carrier). This only has to catch
# throughput far beyond what shared-IP legitimate traffic produces.
_DAILY_QUOTA_PER_IP = 300


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


class RateControlService:
    """
    Throttles how often messages reach append_message/the LLM. Four
    mechanisms:
    - a per-calendar-day allowance per session and per IP, consumed and
      never refunded, which is the only one of the four that caps TOTAL
      volume rather than rate (see _DAILY_QUOTA_PER_SESSION)
    - a hard cap on concurrently in-flight messages per session, rejected
      outright past the cap rather than queued (reserve_slot/release_slot)
    - a minimum interval between the start of consecutive turns for the
      same session, enforced by making each turn wait behind a
      per-session lock so a session's messages are always processed one
      at a time, in arrival order (turn)
    - a hard (looser) cap on concurrently in-flight messages per client
      IP, across guest sessions only -- a backstop for the guest
      per-session cap bypassed by dropping the session cookie
      (reserve_ip_slot/release_ip_slot)

    The first two are split by RateTier ("guest" vs "invite") -- a
    verified/invite-code session gets a shorter interval and a higher
    pending cap than an anonymous guest session. The per-IP backstop is
    deliberately guest-only, not shared across tiers: it exists to catch
    cheap session-cycling, which isn't a meaningful risk for invite
    sessions since re-minting one requires a genuinely valid invite code
    each time -- see the RateTier comment for the full reasoning.

    State is in-process only -- fine for a single Uvicorn worker; would
    need a shared store (e.g. Redis) if this ever runs behind multiple
    workers/replicas, since each process would otherwise track its own
    independent cursors. That applies to the daily quota too, and with one
    extra consequence worth knowing: the counters live in memory, so a
    restart resets everyone's allowance. Acceptable for an app that is
    redeployed rarely and by hand; if the quota ever has to be
    enforceable rather than merely effective, it needs a table or Redis.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_allowed_time: dict[str, float] = {}
        self._pending_counts: dict[str, int] = {}
        self._ip_pending_counts: dict[str, int] = {}
        # Daily quota counters, plus the day they belong to. One shared date
        # rather than a timestamp per key: the whole map is dropped on the
        # first request of a new day, which is also what keeps these two
        # dicts from growing forever.
        self._daily_counts: dict[str, int] = {}
        self._ip_daily_counts: dict[str, int] = {}
        self._quota_day: date = date.today()

    def _roll_quota_day(self) -> None:
        """
        Resets both daily counters when the calendar day has changed.

        Parameters:
        - none

        Returns:
        - None: clears _daily_counts and _ip_daily_counts and advances _quota_day if today is not the stored day; otherwise does nothing.

        Uses the server's local date, deliberately: a portfolio's visitors are
        not synchronised to any timezone, and the exact instant the allowance
        resets does not matter as long as it resets once a day.
        """
        today = date.today()
        if today != self._quota_day:
            self._quota_day = today
            self._daily_counts.clear()
            self._ip_daily_counts.clear()

    def reserve_slot(self, session_id: str, tier: RateTier) -> None:
        """
        Claims one of this session's pending-message slots and one unit of its daily allowance.

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn
        - tier (RateTier): "guest" or "invite" -- comes from ChatService.handle_chat_turn (based on whether `code` is set), selects which cap in _MAX_PENDING_PER_SESSION and _DAILY_QUOTA_PER_SESSION applies

        Returns:
        - None: raises DailyQuotaExceededError if the session has used its whole day's allowance, TooManyPendingMessagesError if it is already at _MAX_PENDING_PER_SESSION[tier]; otherwise increments both counts

        The daily count is CONSUMED, NOT RESERVED -- release_slot does not give
        it back. A quota measures how much a caller has asked the app to do
        today, and a message rejected further down the pipeline still asked.
        Refunding it would also hand a flooding client its allowance back on
        every rejection, which is the opposite of what the limit is for.
        """
        self._roll_quota_day()

        daily = self._daily_counts.get(session_id, 0)
        if daily >= _DAILY_QUOTA_PER_SESSION[tier]:
            logger.info("rate control rejected a message for session=%s (%s): daily allowance of %d used", session_id, tier, _DAILY_QUOTA_PER_SESSION[tier])
            raise DailyQuotaExceededError(f"session {session_id} ({tier})", _DAILY_QUOTA_PER_SESSION[tier])

        count = self._pending_counts.get(session_id, 0)
        if count >= _MAX_PENDING_PER_SESSION[tier]:
            logger.info("rate control rejected a message for session=%s (%s): already %d in flight", session_id, tier, count)
            raise TooManyPendingMessagesError(session_id, tier)

        self._daily_counts[session_id] = daily + 1
        self._pending_counts[session_id] = count + 1

    def release_slot(self, session_id: str) -> None:
        """
        Releases one of this session's pending-message slots once its turn has fully finished (reply persisted, or the turn failed).

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn

        Returns:
        - None: decrements the pending count for session_id, and DELETES the key once it reaches zero. The daily count is untouched -- see reserve_slot.

        The key is removed rather than left at 0, and the session's lock and
        pacing cursor go with it. These dicts previously only ever grew: every
        visitor the process ever saw cost one asyncio.Lock, one float and one
        int for the life of the process, and an attacker minting sessions
        filled them for free. Deleting the lock is safe here only because
        this runs after the `async with lock` block in turn() has exited --
        ChatService calls release_slot from its outermost `finally`, and a
        later message from the same session simply creates a fresh lock via
        setdefault.
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

    def reserve_ip_slot(self, ip: str) -> None:
        """
        Claims one of this IP's pending-message slots -- independent of, and in addition to, reserve_slot's per-session cap. Tier-agnostic itself; ChatService.handle_chat_turn only calls this for "guest" tier (see RateTier).

        Parameters:
        - ip (str): the caller's client IP -- comes from ChatService.handle_chat_turn (get_client_ip)

        Returns:
        - None: raises DailyQuotaExceededError if this IP has used its whole day's allowance, TooManyPendingMessagesFromIpError if it is already at _MAX_PENDING_PER_IP; otherwise increments both counts. The daily count is consumed, not reserved -- see reserve_slot.
        """
        self._roll_quota_day()

        daily = self._ip_daily_counts.get(ip, 0)
        if daily >= _DAILY_QUOTA_PER_IP:
            logger.info("rate control rejected a message for ip=%s: daily allowance of %d used", ip, _DAILY_QUOTA_PER_IP)
            raise DailyQuotaExceededError(f"ip {ip}", _DAILY_QUOTA_PER_IP)

        count = self._ip_pending_counts.get(ip, 0)
        if count >= _MAX_PENDING_PER_IP:
            logger.info("rate control rejected a message for ip=%s: already %d in flight", ip, count)
            raise TooManyPendingMessagesFromIpError(ip)

        self._ip_daily_counts[ip] = daily + 1
        self._ip_pending_counts[ip] = count + 1

    def release_ip_slot(self, ip: str) -> None:
        """
        Releases one of this IP's pending-message slots once the triggering turn has fully finished (reply persisted, or the turn failed). Must only be called if reserve_ip_slot was actually called for the same request (i.e. "guest" tier) -- calling it unconditionally would wrongly decrement an unrelated guest request's count for the same IP.

        Parameters:
        - ip (str): the caller's client IP -- comes from ChatService.handle_chat_turn (get_client_ip)

        Returns:
        - None: decrements the pending count for ip, and deletes the key once it reaches zero (same reasoning as release_slot). The daily count is untouched.
        """
        count = self._ip_pending_counts.get(ip, 0)
        remaining = max(0, count - 1)
        if remaining:
            self._ip_pending_counts[ip] = remaining
        else:
            self._ip_pending_counts.pop(ip, None)

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


# Module-level singleton -- RateControlService's per-session state must
# persist across requests to mean anything, so this must not be
# reconstructed per-request the way FastAPI's Depends() would if handed
# the class directly. get_rate_control_service (below) hands out this same
# instance every time instead.
_rate_control_service = RateControlService()


def get_rate_control_service() -> RateControlService:
    """
    FastAPI dependency provider returning the shared RateControlService singleton.

    Parameters:
    - none

    Returns:
    - RateControlService: the process-wide instance, so per-session state persists across requests -- goes to ChatService via Depends(get_rate_control_service)
    """
    return _rate_control_service
