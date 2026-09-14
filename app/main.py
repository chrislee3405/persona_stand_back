import logging
import os
from contextlib import asynccontextmanager

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
    DailyQuotaExceededError,
    TooManyPendingMessagesError,
    TooManyPendingMessagesFromIpError,
)
from app.routers import chatroom_router, codes_router, conversations_router, consent_router, site_content_router
from app.database import engine, Base
from app.models.consent import ConsentPolicy  # noqa: F401  -- registers table for create_all
from app.models.rate_limit import RateLimitCounter  # noqa: F401  -- registers table for create_all
from app.models.site_content import SiteContent  # noqa: F401  -- registers table for create_all
from app.models.site_image import SiteImage  # noqa: F401  -- registers table for create_all
from app.models.site_journey import SiteJourney  # noqa: F401  -- registers table for create_all
from app.models.site_project import SiteProject  # noqa: F401  -- registers table for create_all


logging.basicConfig(
    level=logging.DEBUG if os.environ.get("ENV", "development") != "production" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Runs the app's startup work, and nothing on shutdown but disposing the connection pool.

    Parameters:
    - application (FastAPI): the app instance -- passed by FastAPI

    Returns:
    - None (async context manager): yields once startup has finished

    WHY THIS IS NOT AT IMPORT TIME ANY MORE. `Base.metadata.create_all` and a
    consent-policy seed used to run as module-level statements, executed while
    `app.main` was being imported -- before uvicorn had an event loop and
    before the ASGI app object existed. A database that was briefly
    unreachable (an RDS failover, a maintenance window, an EC2 restart that
    beat RDS to readiness) therefore raised during IMPORT: uvicorn exited
    non-zero, `restart: unless-stopped` restarted it, and the container
    crash-looped for as long as the database was away. In a lifespan handler
    the same failure is a legible startup error with a log line, and the
    process still owns its own exit.

    The seed is gone entirely. Seeding is data, not application code: the
    consent policy now lives with the rest of the seed data in
    app/models/seed/ and is loaded by `python -m app.models.seed.load`. See
    that package's README for why a placeholder policy created by the app
    itself was the wrong default -- it silently became the live legal notice
    every visitor agreed to.
    """
    # DEVELOPMENT CONVENIENCE, and deliberately unconditional while the schema
    # is still moving: it creates any table in app/models that does not exist
    # yet, so adding a model needs no migration step. It does NOT alter
    # existing tables -- a changed column is silently ignored here and
    # surfaces later as an UndefinedColumn at query time -- so it is not a
    # migration tool and must not be treated as one.
    #
    # REMOVE THIS BEFORE PRODUCTION, along with the DDL grant it requires on
    # every boot. Replace it with Alembic (or a one-off migration step in
    # persona_stand_ec2yml/Part_C.md) once the schema settles.
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("startup: schema check complete (create_all)")

    yield

    await engine.dispose()


app = FastAPI(title="My Backend API", lifespan=lifespan)



setup_middleware(app)

# --- Domain error -> HTTP status ----------------------------------------
# Registered once here rather than as a try/except in each chat route: both
# /api/guestchat and /api/invitechat previously carried an identical
# six-clause block, so a new gate in ChatService had to be mapped twice or
# one endpoint would return an unhandled 500 for a case the other handled.
# A route that wants different behaviour can still catch an error itself --
# codes_router does exactly that, mapping ConversationAccessDeniedError to
# 403 with its own message, and consent_router does it for
# DailyQuotaExceededError (whose message below is written for the chat
# endpoints and would be nonsense on a consent submission). An explicit
# except in a route wins over the handler below, because the exception never
# propagates this far.
_ERROR_STATUS_DETAIL: list[tuple[type[Exception] | tuple[type[Exception], ...], int, str]] = [
    (
        (ConversationNotFoundError, ConversationAccessDeniedError),
        404,
        # A BACKSTOP, not a live path -- and knowingly so. No current route
        # lets these escape: ChatService catches both and starts a fresh
        # conversation (logging a warning so a frontend that stops adopting the
        # returned conversationId is visible), and codes_router maps
        # ConversationAccessDeniedError to its own 403. The entry stays for
        # whatever route is written next and forgets to catch them, because the
        # property it enforces is a security one: the same response for both,
        # so a caller cannot tell an id that does not exist from one owned by
        # somebody else.
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
        # Also 429, but a different fact about the world: the first one means
        # "slow down", this one means "come back tomorrow". Sharing the pacing
        # message here would tell someone who has used their allowance to wait
        # for a reply that is never going to be allowed.
        DailyQuotaExceededError,
        429,
        "You've reached the daily message limit for this chat. "
        "Please try again tomorrow, or get in touch by email instead.",
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

app.include_router(chatroom_router.router, tags=["chatroom"])
app.include_router(conversations_router.router, tags=["conversations"])
app.include_router(codes_router.router, tags=["codes"])
app.include_router(consent_router.router, tags=["consent"])
app.include_router(site_content_router.router, tags=["site-content"])
