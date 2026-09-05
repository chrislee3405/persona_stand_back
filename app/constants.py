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


# Senders excluded from live prompt history AND from summarization, so a
# failed or discarded attempt never reappears as if it were a real reply.
# Consumed by ConversationService.get_recent_messages.
NON_PROMPT_SENDERS = (Sender.ERROR, Sender.REGEN)
