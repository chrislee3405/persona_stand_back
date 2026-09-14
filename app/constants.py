"""
Values shared across services that were previously repeated as string
literals at each use site. Nothing here is behaviour -- these are names for
constants that several modules must agree on, so a change lands in one
place and a typo is a NameError instead of a silently wrong query.
"""

from enum import StrEnum


# --- Gemini models -------------------------------------------------------
# The model every prompt in the reply pipeline uses (topic selection,
# example re-ranking, generation, the response gate, turn splitting).
# Changing it here changes it everywhere -- it was previously spelled out
# at six call sites.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Summarization runs once every RECENT_MESSAGES_BEFORE_SUMMARIZE turns, in a
# background task, and folds a whole conversation into one paragraph -- worth
# a stronger (and more expensive) model than the per-turn calls above.
SUMMARY_MODEL = "gemini-2.5-flash"


# --- Conversation attribution -------------------------------------------
# Sentinel `Conversation.code` for a conversation that has never been linked
# to an invite code. Stored rather than NULL so "guest" is an explicit state;
# see ConversationService.update_conversation_code, which treats it as
# upgradeable while any other code is not.
GUEST_CODE = "GUEST"


# --- Message senders -----------------------------------------------------
class Sender(StrEnum):
    """
    The `Message.sender` vocabulary. A StrEnum so members compare equal to
    the plain strings already in the database -- no migration, and existing
    rows keep working.

    USER / BACKEND / SYSTEM are shown to the visitor (the frontend renders
    them as right-hand, left-hand and centred-notice bubbles respectively).
    ERROR and REGEN are review-only rows kept for debugging and never shown
    or fed back into a prompt -- see NON_PROMPT_SENDERS.
    """

    USER = "user"        # the visitor's own message
    BACKEND = "backend"  # a persona reply turn
    SYSTEM = "system"    # status / error notice, including the gate fallback
    ERROR = "error"      # a failed turn's traceback, kept for review
    REGEN = "regen"      # a response-gate attempt that was rejected

    # A visitor message whose turn failed before a reply existed (see
    # ChatService._handle_failed_turn). The row is kept verbatim so the app
    # owner can see what was being asked; the retag only removes it from the
    # next prompt's history, where "user" would read as a question already
    # answered.
    #
    # The stored VALUE keeps its original spelling: existing message rows
    # carry it, and renaming it would silently return those rows to prompt
    # history.
    UNANSWERED_USER = "not_saved_user"


# Senders excluded from live prompt history AND from summarization, so a
# failed or discarded attempt never reappears as if it were a real reply.
# Consumed by ConversationService.get_recent_messages.
NON_PROMPT_SENDERS = (Sender.ERROR, Sender.REGEN, Sender.UNANSWERED_USER)


# --- Turn deadline -------------------------------------------------------
# Whole-turn wall clock ceiling, seconds. A turn makes 4-12 sequential Gemini
# calls (guest at most 8), each with its own 30s ceiling (_REQUEST_TIMEOUT_MS in
# app/services/ai/gemini_service.py), so the per-call timeout alone bounds a
# turn at ~360s -- far past nginx's proxy_read_timeout (120s in
# persona_stand_front/nginx.conf).
#
# When nginx gives up first the visitor gets a 504 while this process happily
# finishes the turn and COMMITS the reply. The next turn's history then holds
# a persona message the visitor never saw, and the frontend has already told
# them their own message was "Not sent". Both halves of the transcript are
# wrong and neither side knows.
#
# This deadline makes the backend give up FIRST, so the failure is one we
# control: ChatService treats it exactly like any other generation failure --
# the visitor's message is retagged Sender.UNANSWERED_USER and no Sender.BACKEND
# row is ever written, so nothing from a timed-out turn reaches the next
# prompt's history.
#
# MUST stay comfortably under nginx's proxy_read_timeout. Keep the two in step.
TURN_DEADLINE_SECONDS = 100.0


# --- Invite-code brute-force protection ----------------------------------
# Global daily ceiling on FAILED invite-code verifications, summed across every
# IP. Successful verifications never count. Once the day's total reaches this,
# /api/code stops checking codes at all and returns 429 until the counter rolls
# over at midnight (server local date, same rule as every other daily counter).
#
# Global rather than per-IP because per-IP alone is not a brute-force control:
# an attacker with a botnet or a rotating proxy pool simply spreads the guesses.
# A global circuit breaker bounds total guesses per day no matter how they are
# distributed. It fails closed -- legitimate invite holders are locked out for
# the rest of the day too -- which is the correct trade for this app: an invite
# holder can email instead, and a guessed code is unmetered LLM spend.
MAX_DAILY_INVITE_CODE_FAILURES = 300

# Per-IP daily ceiling on failed invite-code verifications. The first layer:
# it stops one machine burning the global allowance on its own, so the global
# breaker above is only reached by genuinely distributed traffic.
MAX_DAILY_INVITE_CODE_FAILURES_PER_IP = 20

# Per-IP daily ceiling on POST /api/consent. That endpoint takes no credential
# and INSERTS a consent_record row per previously-unseen session id, and a
# caller mints a fresh session id simply by dropping its cookie -- so without
# this it is an unauthenticated, unbounded row-insertion endpoint. Generous
# enough that a shared NAT/office IP never trips it: a genuine visitor consents
# once per policy version.
MAX_DAILY_CONSENT_SUBMISSIONS_PER_IP = 50


# --- Rate-limit counter retention ----------------------------------------
# How many days of daily rate-limit rows to keep. They are retained rather than
# deleted on rollover so the owner can review traffic and spot abuse after the
# fact -- but they are one row per (key, day), and `key` includes per-session
# and per-IP values, so without a ceiling the table grows forever.
# Swept opportunistically on the first request of each new day; see
# RateControlService._sweep_expired_counters.
RATE_LIMIT_RETENTION_DAYS = 90
