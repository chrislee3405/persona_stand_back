from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.services.consent_service import ConsentService, ConsentTextMismatchError, NoConsentPolicyConfiguredError
from app.dependencies.session import get_or_create_session_id

router = APIRouter()


class ConsentSubmission(BaseModel):
    """Body for POST /api/consent -- the client must echo back the exact current policy text, not just a bare confirmation (see ConsentService.record_consent)."""
    conditionText: str


@router.get("/api/consent")
async def get_consent_status(request: Request, service: ConsentService = Depends()):
    """
    Handles GET /api/consent: reports whether this session has already consented to the current policy version, plus the policy's text for the popup to render.

    Parameters:
    - request (Request): the incoming request — comes from FastAPI, used to read/write the session cookie
    - service (ConsentService): checks the consent record and looks up the current policy — injected by FastAPI

    Returns:
    - dict: consented (bool), policyVersion, and conditionText (both None if no policy is configured yet) — sent back to the client as the JSON response
    """
    session_id = get_or_create_session_id(request)
    policy = service.get_current_policy()
    return {
        "consented": service.is_consented(session_id),
        "policyVersion": policy.version if policy else None,
        "conditionText": policy.condition_text if policy else None
    }


@router.post("/api/consent")
async def submit_consent(payload: ConsentSubmission, request: Request, service: ConsentService = Depends()):
    """
    Handles POST /api/consent: records that this session has agreed to the current policy version -- the caller must submit the exact current policy text (see ConsentSubmission), which is validated and stored on the record.

    Parameters:
    - payload (ConsentSubmission): conditionText -- comes from the validated request body
    - request (Request): the incoming request — comes from FastAPI, used to read/write the session cookie
    - service (ConsentService): validates and persists the consent record — injected by FastAPI

    Returns:
    - dict: consented (always True on success) and policyVersion — sent back to the client as the JSON response
    """
    session_id = get_or_create_session_id(request)
    try:
        service.record_consent(session_id, payload.conditionText)
    except NoConsentPolicyConfiguredError:
        raise HTTPException(status_code=500, detail="No consent policy is configured yet.")
    except ConsentTextMismatchError:
        raise HTTPException(status_code=400, detail="Submitted consent text doesn't match the current policy. Fetch GET /api/consent for the current text first.")

    policy = service.get_current_policy()
    return {
        "consented": True,
        "policyVersion": policy.version if policy else None
    }
