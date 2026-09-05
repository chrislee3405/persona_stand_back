import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.middleware import setup_middleware
from app.services.chat_service import MessageTooLongError, MAX_MESSAGE_LENGTH
from app.services.consent_service import ConsentRequiredError
from app.services.conversation_manage_service import (
    ConversationAccessDeniedError,
    ConversationNotFoundError,
)
from app.services.privacy_gate_service import PrivacyViolationError
from app.services.rate_control_service import (
    TooManyPendingMessagesError,
    TooManyPendingMessagesFromIpError,
)
from app.routers import codes_router, conversations_router, consent_router, site_content_router
from app.database import engine, Base, SessionLocal
from app.models.consent import ConsentPolicy
from app.models.site_content import SiteContent  # noqa: F401  -- registers table for create_all
from app.models.site_image import SiteImage  # noqa: F401  -- registers table for create_all
from app.models.site_journey import SiteJourney  # noqa: F401  -- registers table for create_all
from app.models.site_project import SiteProject  # noqa: F401  -- registers table for create_all


logging.basicConfig(
    level=logging.DEBUG if os.environ.get("ENV", "development") != "production" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


Base.metadata.create_all(bind=engine) # create tables in models folder if they don't exist in db

# Seed an initial consent policy if none exists yet -- otherwise
# consent_policy starts empty and ConsentService.check would block every
# session with no way to actually agree to anything. Placeholder wording;
# add a new row (don't edit this one) with real text and a new version
# once it's ready -- see ConsentPolicy/ConsentService.get_current_policy.
with SessionLocal() as _seed_db:
    if _seed_db.query(ConsentPolicy).first() is None:
        _seed_db.add(ConsentPolicy(
            version="v1-placeholder",
            condition_text=(
                "This chatroom collects and stores the messages you send so the "
                "conversation can function and be reviewed. By clicking \"I Agree\", "
                "you consent to your messages being collected for this purpose."
            )
        ))
        _seed_db.commit()

app = FastAPI(title="My Backend API")



setup_middleware(app)

# --- Domain error -> HTTP status ----------------------------------------
# Registered once here rather than as a try/except in each chat route: both
# /api/guestchat and /api/invitechat previously carried an identical
# six-clause block, so a new gate in ChatService had to be mapped twice or
# one endpoint would return an unhandled 500 for a case the other handled.
# A route that wants different behaviour can still catch an error itself --
# codes_router does exactly that, mapping ConversationAccessDeniedError to
# 403 with its own message, and its explicit except wins over the handler
# below because the exception never propagates this far.
_ERROR_STATUS_DETAIL: list[tuple[type[Exception] | tuple[type[Exception], ...], int, str]] = [
    (
        (ConversationNotFoundError, ConversationAccessDeniedError),
        404,
        # Same response for both -- a caller must not be able to tell an
        # id that does not exist from one owned by somebody else.
        "conversationId not found",
    ),
    (
        PrivacyViolationError,
        400,
        "Your message appears to contain personal or sensitive information "
        "(e.g. a email, phone number, or ID). Please remove it and try again.",
    ),
    (
        (TooManyPendingMessagesError, TooManyPendingMessagesFromIpError),
        429,
        "You're sending messages faster than they can be answered. "
        "Please wait for a reply before sending another.",
    ),
    (
        ConsentRequiredError,
        403,
        "You must agree to the data collection notice before sending messages.",
    ),
    (
        MessageTooLongError,
        413,
        f"Message is too long (max {MAX_MESSAGE_LENGTH} characters).",
    ),
]


def _register_error_handlers(application: FastAPI) -> None:
    """
    Maps each domain exception to its HTTP status and user-facing detail.

    Parameters:
    - application (FastAPI): the app instance -- comes from module scope at startup

    Returns:
    - None: registers one exception handler per entry in _ERROR_STATUS_DETAIL
    """
    for exc_types, status_code, detail in _ERROR_STATUS_DETAIL:
        for exc_type in (exc_types if isinstance(exc_types, tuple) else (exc_types,)):
            # Bind the loop values per handler -- a closure over the loop
            # variables would leave every handler using the last entry.
            def handler(request: Request, exc: Exception, _status=status_code, _detail=detail):
                return JSONResponse(status_code=_status, content={"detail": _detail})

            application.add_exception_handler(exc_type, handler)


_register_error_handlers(app)

app.include_router(conversations_router.router, tags=["conversations"])
app.include_router(codes_router.router, tags=["codes"])
app.include_router(consent_router.router, tags=["consent"])
app.include_router(site_content_router.router, tags=["site-content"])
