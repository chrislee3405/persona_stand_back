"""
Deployment settings read from the environment at startup.

LOG_LEVEL and SESSION_COOKIE_SECURE used to be one variable. `ENV=production`
raised the log level to INFO AND marked the session cookie Secure, and those
two cannot move together: a Secure cookie is never sent over plain http://, so
a deployment without TLS had to stay at `ENV=development` -- which kept DEBUG
logging on, and with it every visitor's full prompt, history and reply in the
container logs. They are separate now, so logging can be production-grade
before TLS exists and the cookie can follow TLS on its own schedule.

`ENV` still works as the fallback for both, so an existing .env that only sets
ENV behaves exactly as before.
"""

import logging
import os

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _is_production() -> bool:
    """
    Reports whether the legacy ENV variable says production.

    Parameters:
    - none

    Returns:
    - bool: True when ENV=production -- the default for both settings below when their own variable is unset
    """
    return os.environ.get("ENV", "development") == "production"


def log_level() -> int:
    """
    Resolves the root log level.

    Parameters:
    - none

    Returns:
    - int: the logging level from LOG_LEVEL (DEBUG, INFO, WARNING or ERROR, any case), else INFO under ENV=production and DEBUG otherwise -- goes to logging.basicConfig in app/main.py.
      Raises RuntimeError for any other LOG_LEVEL value: a typo must not silently leave DEBUG, and the prompt dumps it carries, switched on.

    DEBUG includes the reply pipeline's full prompts and model responses, so a
    deployment serving real visitors should run at INFO or above.
    """
    raw = os.environ.get("LOG_LEVEL", "").strip()
    if not raw:
        return logging.INFO if _is_production() else logging.DEBUG
    level = _LOG_LEVELS.get(raw.upper())
    if level is None:
        raise RuntimeError(f"LOG_LEVEL={raw!r} is not one of: {', '.join(_LOG_LEVELS)}")
    return level


def _flag(name: str, default: bool) -> bool:
    """
    Reads an on/off environment variable strictly.

    Parameters:
    - name (str): the variable to read -- comes from the setting functions below
    - default (bool): the value when the variable is unset or empty -- comes from the same caller

    Returns:
    - bool: true/false, 1/0, yes/no, on/off in any case -- goes to the caller.
      Raises RuntimeError for any other value, rather than guessing which way a typo was meant.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise RuntimeError(f"{name}={raw!r} must be true or false")


def session_cookie_secure() -> bool:
    """
    Resolves whether the session cookie carries the Secure flag.

    Parameters:
    - none

    Returns:
    - bool: SESSION_COOKIE_SECURE parsed strictly (see _flag), else True under ENV=production and False otherwise -- goes to SessionMiddleware's https_only in app/middleware.py

    Turn it on only once the site is served over HTTPS end to end. A browser
    never sends a Secure cookie over plain http://, so enabling it early
    empties every visitor's session on every request: consent never sticks,
    every chat turn returns 403, and every message opens a new conversation.
    """
    return _flag("SESSION_COOKIE_SECURE", default=_is_production())


def chat_trace_enabled() -> bool:
    """
    Resolves whether conversation content may be written to the logs.

    Parameters:
    - none

    Returns:
    - bool: CHAT_TRACE parsed strictly (see _flag), False when unset -- goes to app/chat_trace.py, which switches the trace logger on or off

    OFF UNLESS SET, IN EVERY ENVIRONMENT, and independent of LOG_LEVEL. The
    trace logger carries full prompts, model responses and anything a model
    wrote about the conversation. Tying it to DEBUG meant that one log-level
    change -- or one missing ENV line -- put every visitor's messages in the
    production logs. Now it takes this explicit, separately named switch, and
    the deploy script refuses a release that turns it on
    (persona_stand_ec2yml/scripts/deploy_release.py).
    """
    return _flag("CHAT_TRACE", default=False)
