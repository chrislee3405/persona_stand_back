import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.session import get_client_ip, get_or_create_session_id, set_verified_invite_code_id
from app.services.code_service import CodeService, InvalidCodeError
from app.services.consent_service import ConsentService
from app.services.conversation_manage_service import (
    ConversationAccessDeniedError,
    ConversationCodeAlreadyLinkedError,
    ConversationService,
)
from app.services.rate_control_service import (
    InviteVerificationLockedError,
    RateControlService,
    get_rate_control_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class CodeSubmission(BaseModel):
    """
    Body for POST /api/code.

    camelCase, like every other endpoint in this API. This one was
    snake_case on both the request and the response (`input_code`,
    `conversation_id`, `returned_result`) while /api/guestchat,
    /api/invitechat, /api/consent and /api/site-content were all camelCase
    -- one convention per API, and this was the outlier.

    `inputCode` is length-bounded: it is looked up against the database, and
    nothing else stopped an arbitrarily large string getting that far. Real
    codes are short.
    """
    model_config = ConfigDict(extra="forbid")

    inputCode: str = Field(..., min_length=1, max_length=128)
    conversationId: str | None = Field(default=None, max_length=64)


@router.post("/api/code")
async def verify_code(
    payload: CodeSubmission,
    request: Request,
    db: AsyncSession = Depends(get_db),
    service: CodeService = Depends(),
    conversation_service: ConversationService = Depends(),
    consent_service: ConsentService = Depends(),
    rate_control: RateControlService = Depends(get_rate_control_service),
):
    """
    Handles POST /api/code: verifies a submitted invite code, rotates the session id, and marks the new session as verified.

    Parameters:
    - payload (CodeSubmission): inputCode and optional conversationId — comes from the request body
    - request (Request): the incoming request — comes from FastAPI, used to read/write the session cookie
    - db (AsyncSession): the request's database session — injected by FastAPI; shared with every service below, so the rotation is one transaction
    - service (CodeService): looks up the code and links it to a conversation — injected by FastAPI
    - conversation_service (ConversationService): moves conversation ownership to the rotated session id — injected by FastAPI
    - consent_service (ConsentService): moves consent records to the rotated session id — injected by FastAPI
    - rate_control (RateControlService): enforces the brute-force ceilings — injected by FastAPI as the shared singleton

    Returns:
    - dict: {"status": "success"} — sent back to the client as the JSON response. The code itself is deliberately not echoed.
    """
    session_id = get_or_create_session_id(request)
    client_ip = get_client_ip(request)

    # --- Brute-force ceilings, BEFORE the code is even looked up ----------
    # Two layers. The per-IP one stops a single machine burning the global
    # allowance; the GLOBAL one is what actually bounds brute force, because
    # per-IP alone is defeated by a rotating proxy pool -- every address just
    # gets a fresh allowance. Only FAILED attempts count toward either, so a
    # company working through its own invite can never walk the app toward a
    # lockout. See MAX_DAILY_INVITE_CODE_FAILURES in app/constants.py.
    try:
        await rate_control.assert_invite_verification_allowed(db, client_ip)
    except InviteVerificationLockedError:
        # Deliberately generic and deliberately identical for both layers: an
        # attacker must not be able to tell whether they personally are
        # locked out or the app as a whole is, because the difference tells
        # them whether spreading the attack would help.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Invite code verification is temporarily unavailable. Please try again later.",
        )

    try:
        matched = await service.match_code(
            input_code=payload.inputCode,
            conversation_id=payload.conversationId,
            session_id=session_id
        )
    except InvalidCodeError:
        # Count it, then refuse. This is the only thing that increments either
        # ceiling.
        await rate_control.record_invite_code_failure(db, client_ip)
        # 401, not 400: the request itself is well-formed, the credential in it
        # is simply not valid. (It said 400 with the internal phrase "Process
        # result not found", which the frontend surfaces verbatim to the
        # visitor -- errorDetail() in lib/api.ts prefers the server's message.)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="That invite code wasn't recognised. Check it and try again."
        )
    except ConversationAccessDeniedError:
        # The code itself was valid, but this session didn't create the
        # conversation it tried to upgrade — don't leak which is true.
        # Not a failed CODE attempt, so it does not count toward the ceilings.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="cannot link code to a conversation this session doesn't own"
        )
    except ConversationCodeAlreadyLinkedError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="conversation is already linked to a different invite code"
        )

    # --- Session rotation --------------------------------------------------
    # This is the moment the session gains privilege, so it stops using the id
    # it had before it. Without rotation, an id an attacker already knows --
    # and over plain HTTP one can be planted in the victim's browser, because
    # a response that is not authenticated can be rewritten in flight -- is
    # the same id that ends up holding the invite tier, the consent record and
    # ownership of the conversation.
    #
    # There is no server-side session store to expire (the session IS the
    # signed cookie), so "invalidating" the old id means moving everything
    # that made it worth having: after the transaction below, a replayed old
    # cookie names a session that owns no conversation and holds no consent.
    #
    # ONE TRANSACTION. The code link (done inside match_code), the ownership
    # move and the consent move commit together or not at all -- a partial
    # rotation would strand the visitor with a new cookie whose id owns none
    # of their history and an old id their browser no longer has.
    new_session_id = str(uuid.uuid4())
    moved_conversations = await conversation_service.transfer_ownership(session_id, new_session_id)
    moved_consents = await consent_service.transfer_records(session_id, new_session_id)
    await db.commit()

    request.session.clear()
    request.session["session_id"] = new_session_id
    # This is the actual authorization event. Every /api/invitechat call
    # from here on derives its verified status from this session cookie —
    # the client never needs to (and no longer does) send the code again.
    # The session records the code's database id, never the code: the cookie
    # is signed but readable, so the plaintext code used to be recoverable
    # from it with a base64 decode (see set_verified_invite_code_id).
    set_verified_invite_code_id(request, matched.id)

    logger.info(
        "invite code verified: rotated session id, moved %d conversation(s) and %d consent record(s)",
        moved_conversations, moved_consents,
    )

    # No echo of the code in any form. `received`, `returned_result` and then
    # `verifiedCode` each handed the credential back to whoever sent it; the
    # client already knows what it typed, and a second tab learns only that
    # the session IS verified (GET /api/chatroom_initialize), never with what.
    return {
        "status": "success",
    }
