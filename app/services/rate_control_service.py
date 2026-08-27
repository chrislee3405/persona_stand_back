import asyncio
import logging
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Hyperparameters -- the one place to tune these.
# Minimum time between the start of two consecutive message-turns for the
# same session, in seconds. Throttles how often a session can trigger a
# Gemini call, to bound cost from a flooded/scripted burst of messages --
# without ever refusing to accept a message outright (that's what
# MAX_PENDING_PER_SESSION below is for).
_INTERVAL_SECONDS = 3.0

# How many messages a single session may have in flight (sent, reply not
# yet fully returned) at once. A message beyond this is rejected outright
# (TooManyPendingMessagesError -> HTTP 429 in conversations_router)
# instead of queuing indefinitely -- mirrors a real conversation, where
# stacking more than a few unanswered messages in a row stops making
# sense. Mirrored client-side in Chatroom.tsx (MAX_PENDING_MESSAGES) for
# instant UI feedback, but this is the actual enforcement point -- the
# client-side check is trivially bypassable.
_MAX_PENDING_PER_SESSION = 3

# How many messages a single IP address may have in flight at once, across
# any session -- a backstop against a script that defeats
# _MAX_PENDING_PER_SESSION by simply dropping its session cookie between
# requests (a fresh session has no memory of prior messages, see
# get_or_create_session_id). Deliberately more generous than the
# per-session cap: multiple genuine users can share one public IP (NAT,
# office/campus network, mobile carrier), so this only needs to catch
# throughput far beyond what shared-IP legitimate traffic would produce.
# Unlike turn() below, this is cap-only, not paced/serialized -- pacing
# per IP would wrongly queue unrelated users' conversations behind each
# other whenever they happen to share an IP.
_MAX_PENDING_PER_IP = 5


class TooManyPendingMessagesError(Exception):
    """Raised when a session already has _MAX_PENDING_PER_SESSION messages in flight."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"session {session_id} already has {_MAX_PENDING_PER_SESSION} messages in flight")


class TooManyPendingMessagesFromIpError(Exception):
    """Raised when a client IP already has _MAX_PENDING_PER_IP messages in flight, across any session."""

    def __init__(self, ip: str):
        self.ip = ip
        super().__init__(f"ip {ip} already has {_MAX_PENDING_PER_IP} messages in flight")


class RateControlService:
    """
    Throttles how often messages reach append_message/the LLM. Three
    mechanisms:
    - a hard cap on concurrently in-flight messages per session, rejected
      outright past the cap rather than queued (reserve_slot/release_slot)
    - a minimum interval between the start of consecutive turns for the
      same session, enforced by making each turn wait behind a
      per-session lock so a session's messages are always processed one
      at a time, in arrival order (turn)
    - a hard (looser) cap on concurrently in-flight messages per client
      IP, across sessions -- a backstop for a session cap bypassed by
      dropping the session cookie (reserve_ip_slot/release_ip_slot)

    State is in-process only -- fine for a single Uvicorn worker; would
    need a shared store (e.g. Redis) if this ever runs behind multiple
    workers/replicas, since each process would otherwise track its own
    independent cursors.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_allowed_time: dict[str, float] = {}
        self._pending_counts: dict[str, int] = {}
        self._ip_pending_counts: dict[str, int] = {}

    def reserve_slot(self, session_id: str) -> None:
        """
        Claims one of this session's pending-message slots.

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn

        Returns:
        - None: raises TooManyPendingMessagesError if the session is already at _MAX_PENDING_PER_SESSION; otherwise increments its pending count
        """
        count = self._pending_counts.get(session_id, 0)
        if count >= _MAX_PENDING_PER_SESSION:
            logger.info("rate control rejected a message for session=%s: already %d in flight", session_id, count)
            raise TooManyPendingMessagesError(session_id)
        self._pending_counts[session_id] = count + 1

    def release_slot(self, session_id: str) -> None:
        """
        Releases one of this session's pending-message slots once its turn has fully finished (reply persisted, or the turn failed).

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn

        Returns:
        - None: decrements the pending count for session_id, floored at 0
        """
        count = self._pending_counts.get(session_id, 0)
        self._pending_counts[session_id] = max(0, count - 1)

    def reserve_ip_slot(self, ip: str) -> None:
        """
        Claims one of this IP's pending-message slots -- independent of, and in addition to, reserve_slot's per-session cap.

        Parameters:
        - ip (str): the caller's client IP -- comes from ChatService.handle_chat_turn (get_client_ip)

        Returns:
        - None: raises TooManyPendingMessagesFromIpError if the IP is already at _MAX_PENDING_PER_IP; otherwise increments its pending count
        """
        count = self._ip_pending_counts.get(ip, 0)
        if count >= _MAX_PENDING_PER_IP:
            logger.info("rate control rejected a message for ip=%s: already %d in flight", ip, count)
            raise TooManyPendingMessagesFromIpError(ip)
        self._ip_pending_counts[ip] = count + 1

    def release_ip_slot(self, ip: str) -> None:
        """
        Releases one of this IP's pending-message slots once the triggering turn has fully finished (reply persisted, or the turn failed).

        Parameters:
        - ip (str): the caller's client IP -- comes from ChatService.handle_chat_turn (get_client_ip)

        Returns:
        - None: decrements the pending count for ip, floored at 0
        """
        count = self._ip_pending_counts.get(ip, 0)
        self._ip_pending_counts[ip] = max(0, count - 1)

    @asynccontextmanager
    async def turn(self, session_id: str):
        """
        Scopes one message's turn for a session: waits for the session's lock (serializing its turns to one at a time, in arrival order -- so a later message's conversation history always includes an earlier one's already-persisted reply), then sleeps off whatever remains of _INTERVAL_SECONDS since the previous turn started, before yielding control to the caller. The lock stays held for the caller's entire `async with` block, so the next queued message can't start until this one -- including persisting its reply -- is done.

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn

        Returns:
        - None (async context manager): yields once this turn is clear to proceed; releases the session's lock on exit
        """
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            scheduled_time = max(now, self._next_allowed_time.get(session_id, now))
            self._next_allowed_time[session_id] = scheduled_time + _INTERVAL_SECONDS

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
