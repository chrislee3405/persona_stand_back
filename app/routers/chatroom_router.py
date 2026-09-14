import logging

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.dependencies.session import get_or_create_session_id, get_verified_invite_code_id
from app.services.consent_service import ConsentService

logger = logging.getLogger(__name__)

router = APIRouter()

# The `consent` block reported when the terms cannot be produced -- no policy
# row, an unusable row, or the database unreachable. Same SHAPE as the success
# case, so the frontend branches on the status code and on `conditionTerms`
# being null, never on which keys exist.
#
# Fails CLOSED: "we could not find out" reads as "not consented", never as
# consented. The frontend renders it as the inert "terms unavailable" card.
_CONSENT_UNAVAILABLE = {
    "consented": False,
    "policyVersion": None,
    "conditionTerms": None,
}


@router.get("/api/chatroom_initialize")
async def chatroom_initialize(request: Request, consent_service: ConsentService = Depends()):
    """
    Handles GET /api/chatroom_initialize: everything the chatroom needs to know about this session when it first loads.

    Parameters:
    - request (Request): the incoming request — comes from FastAPI, used to read/write the session cookie
    - consent_service (ConsentService): looks up the current policy and this session's agreement — injected by FastAPI

    Returns:
    - dict: sent back to the client as the JSON response --
        {
          "consent": {
            "consented":      bool,
            "policyVersion":  str | null,
            "conditionTerms": {"header": str, "condition": str} | null
          },
          "verified": bool
        }
      Status 200 normally, including when there is simply no usable policy
      (nulls, nothing is broken). Status 503 when the database could not be
      reached -- `consent` is then the unavailable block, but `verified` is
      still accurate, because it comes from the signed session cookie and needs
      no database at all.

    WHY THIS REPLACED GET /api/consent. A chatroom opened in a new tab needs two
    facts before it can render correctly: whether this session has agreed to
    the terms, and whether it has verified an invite code. The second used to
    live only in the tab that did the verifying (sessionStorage), so a second
    tab treated an invite session as a guest -- guest pacing, guest daily
    quota, and half the regeneration budget. Adding verification to an endpoint
    called /api/consent would have hidden invite logic somewhere nobody would
    look for it, so the endpoint is named for what it actually is.

    `verified` IS REPORTED, NOT RE-CHECKED. It says whether the signed session
    carries an invite-code id; it does not look the code up. Checking a code
    is the verification flow's job (POST /api/code), and the one place a stale
    verification matters -- a code deleted since -- is caught on the next
    /api/invitechat call, which resolves the id and falls back to guest. The
    invite code itself never appears in this response, in any form.
    """
    session_id = get_or_create_session_id(request)
    verified = get_verified_invite_code_id(request) is not None

    try:
        current = await consent_service.get_current_terms()
        consented = await consent_service.is_consented(session_id)
    except SQLAlchemyError:
        # The database is unreachable or erroring. 503 so the frontend can say
        # "disconnected" rather than landing on the right card by accident --
        # but still with a full body, so verification state reaches the tab.
        logger.exception("chatroom initialisation: consent lookup failed -- reporting terms as unavailable")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"consent": dict(_CONSENT_UNAVAILABLE), "verified": verified},
        )

    if current is None:
        return {"consent": dict(_CONSENT_UNAVAILABLE), "verified": verified}

    version, terms = current
    return {
        "consent": {
            "consented": consented,
            "policyVersion": version,
            "conditionTerms": {"header": terms["header"], "condition": terms["condition"]},
        },
        "verified": verified,
    }
