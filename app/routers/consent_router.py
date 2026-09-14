import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.session import get_client_ip, get_or_create_session_id
from app.services.consent_service import (
    ConsentService,
    ConsentTextMismatchError,
    NoConsentPolicyConfiguredError,
)
from app.services.rate_control_service import (
    DailyQuotaExceededError,
    RateControlService,
    get_rate_control_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Shown to the visitor whenever the terms cannot be produced, whatever the
# reason. Deliberately ONE message for three different causes -- no policy row
# configured, the current (highest-id) row's terms unusable, and the database unreachable --
# because the difference is an operational detail the visitor can do nothing
# with.
#
# The frontend renders this state as a card with both choices inert and only a
# close button (see `termsUnavailable` in Chatroom.tsx): there is nothing to
# agree OR object to when nobody can read the terms, so offering either would
# be collecting a consent that means nothing.
_UNAVAILABLE_DETAIL = "Consent terms are currently unavailable. Please try again later."


class ConsentSubmission(BaseModel):
    """
    Body for POST /api/consent.

    `conditionText` is the policy's `condition` string, echoed back exactly --
    not a bare confirmation (see ConsentService.record_consent). The client
    gets it from GET /api/chatroom_initialize's `consent.conditionTerms.condition`.
    The header is not echoed: it is a label for the box, not the thing being
    agreed to.

    `extra="forbid"` and a length bound, like every other body model in this
    API. This one had neither, so an arbitrarily large JSON string was read
    and parsed before record_consent compared it and rejected it -- bounded
    only by nginx's default client_max_body_size. 8 KB is far more than any
    real policy text and far less than a payload worth parsing by accident.
    """
    model_config = ConfigDict(extra="forbid")

    conditionText: str = Field(..., min_length=1, max_length=8192)


@router.post("/api/consent")
async def submit_consent(
    payload: ConsentSubmission,
    request: Request,
    db: AsyncSession = Depends(get_db),
    service: ConsentService = Depends(),
    rate_control: RateControlService = Depends(get_rate_control_service),
):
    """
    Handles POST /api/consent: records that this session has agreed to the current policy version -- the caller must submit the exact current condition text (see ConsentSubmission), which is validated and stored on the record.

    Parameters:
    - payload (ConsentSubmission): conditionText -- comes from the validated request body
    - request (Request): the incoming request — comes from FastAPI, used to read/write the session cookie
    - db (AsyncSession): the request's database session — injected by FastAPI, used by the rate-limit counter
    - service (ConsentService): validates and persists the consent record — injected by FastAPI
    - rate_control (RateControlService): enforces the per-IP daily submission ceiling — injected by FastAPI as the shared singleton

    Returns:
    - dict: consented (always True on success) and policyVersion — sent back to the client as the JSON response

    REFUSES rather than records whenever there is nothing usable to agree to:
    503 with the same message GET /api/chatroom_initialize reports, so the popup falls back to its
    inert state instead of offering a button the server will keep rejecting.
    """
    # RATE LIMITED, per IP per day. This endpoint takes no credential and
    # INSERTS a consent_record row for every previously-unseen session id --
    # and a caller mints a fresh session id simply by dropping its cookie. It
    # was therefore an unauthenticated, unbounded row-insertion endpoint
    # pointed at the same RDS volume everything else depends on, and a full
    # volume makes RDS read-only, at which point every write in the app fails.
    # The ceiling is generous enough that a shared office/NAT address never
    # trips it: a genuine visitor consents once per policy version.
    client_ip = get_client_ip(request)
    try:
        await rate_control.consume_consent_submission(db, client_ip)
    except DailyQuotaExceededError:
        # Caught here rather than left to the handler in app/main.py: that
        # one's message is written for the chat endpoints ("You've reached the
        # daily message limit for this chat") and would be nonsense on a
        # consent submission.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests from this network. Please try again later.",
        )
    except SQLAlchemyError:
        # The counter itself lives in Postgres, so this is the first place a
        # database outage shows up on this route. Refuse, rather than skip the
        # limiter and carry on into a write that is going to fail anyway.
        logger.exception("consent rate-limit check failed -- refusing the submission")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE_DETAIL
        )

    session_id = get_or_create_session_id(request)
    try:
        version = await service.record_consent(session_id, payload.conditionText)
    except NoConsentPolicyConfiguredError:
        # 503, not 500. There being no usable policy is not a bug in this
        # server -- it is a configuration state the app is designed to have,
        # and the right answer is to refuse the submission and say the terms
        # are unavailable. Recording an agreement against a policy that does
        # not exist would produce a consent_record whose policy_version points
        # at nothing, which is worse than refusing: it would look like proof.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE_DETAIL
        )
    except ConsentTextMismatchError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Submitted consent text doesn't match the current policy. Fetch GET /api/chatroom_initialize for the current terms first.",
        )
    except SQLAlchemyError:
        logger.exception("recording consent failed -- reporting terms as unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE_DETAIL
        )

    return {
        "consented": True,
        "policyVersion": version,
    }


@router.post("/api/consent/withdraw")
async def withdraw_consent(request: Request, service: ConsentService = Depends()):
    """
    Handles POST /api/consent/withdraw: withdraws this session's consent, so nothing further it sends is processed or stored until it agrees again.

    Parameters:
    - request (Request): the incoming request — comes from FastAPI, used to read the session cookie
    - service (ConsentService): marks the session's active consent records withdrawn — injected by FastAPI

    Returns:
    - dict: {"consented": false} — sent back to the client as the JSON response. The
      frontend returns the visitor to the consent card, which they must explicitly
      agree to before the conversation can continue.

    Idempotent, and deliberately so: withdrawing when nothing is in force
    returns the same answer, because "not consented" is already the state the
    caller asked for.

    Not rate limited. Unlike POST /api/consent it inserts nothing -- it only
    stamps `withdrawn_at` on rows this session already owns -- so it cannot be
    used to grow the database.

    What this does NOT do is delete the conversation. Withdrawal stops future
    collection; removing what was already collected is a separate request made
    to the site owner, quoting the conversation reference shown in the
    chatroom (see the README's "Deleting a conversation on request").
    """
    session_id = get_or_create_session_id(request)
    try:
        await service.withdraw_consent(session_id)
    except SQLAlchemyError:
        # Say so rather than pretending: a withdrawal that silently failed would
        # leave the visitor believing collection had stopped when it had not.
        logger.exception("consent withdrawal failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Your consent could not be withdrawn right now. Please try again in a moment.",
        )
    return {"consented": False}
