"""
The one logger allowed to carry conversation content.

Anything that would put a visitor's words -- or a model's output about them --
into the logs goes through `trace`, never an ordinary module logger:

  - full system / user prompts and model responses
  - model-written fields about the conversation (the readiness `reason`,
    grounding's `missing`, response-gate quotes)
  - a malformed model response, printed for debugging

It is silent unless CHAT_TRACE is on (app/runtime_settings.py), whatever
LOG_LEVEL says. With CHAT_TRACE on it logs at DEBUG and passes its records
to the normal handlers even when LOG_LEVEL is higher -- turning it on is the
explicit decision, so it is not also gated on the level.

Ordinary loggers keep the non-content half of every such line (a decision,
a count, a category), so production logs still say what happened, just not
what anyone said.
"""

import logging

from app.runtime_settings import chat_trace_enabled

LOGGER_NAME = "app.chat_trace"

# Above CRITICAL, so no record passes: the logger is off, not merely quiet.
_OFF = logging.CRITICAL + 1

trace = logging.getLogger(LOGGER_NAME)


def configure() -> None:
    """
    Switches the trace logger on or off from CHAT_TRACE.

    Parameters:
    - none

    Returns:
    - None: sets the trace logger's level -- DEBUG when CHAT_TRACE is on, above CRITICAL otherwise.

    Runs at import, so every entrypoint that reaches the pipeline (the app,
    the probe scripts, tests) gets the same default without having to call it.
    Tests call it again after changing CHAT_TRACE.
    """
    trace.setLevel(logging.DEBUG if chat_trace_enabled() else _OFF)


configure()
