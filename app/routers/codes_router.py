from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, ConfigDict, Field

from app.services.code_service import CodeService, InvalidCodeError
from app.services.conversation_manage_service import ConversationAccessDeniedError, ConversationCodeAlreadyLinkedError
from app.dependencies.session import get_or_create_session_id


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
async def verify_code(payload: CodeSubmission, request: Request, service: CodeService = Depends()):
    """
    Handles POST /api/code: verifies a submitted invite code and marks the session as verified.

    Parameters:
    - payload (CodeSubmission): inputCode and optional conversationId — comes from the request body
    - request (Request): the incoming request — comes from FastAPI, used to read/write the session cookie
    - service (CodeService): looks up the code and links it to a conversation — injected by FastAPI

    Returns:
    - dict: status and verifiedCode (the matched code) — sent back to the client as the JSON response
    """
    session_id = get_or_create_session_id(request)

    try:
        processed_result = await service.match_code(
            input_code=payload.inputCode,
            conversation_id=payload.conversationId,
            session_id=session_id
        )
    except InvalidCodeError:
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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="cannot link code to a conversation this session doesn't own"
        )
    except ConversationCodeAlreadyLinkedError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="conversation is already linked to a different invite code"
        )

    # This is the actual authorization event. Every /api/invitechat call
    # from here on derives its verified status from this session cookie —
    # the client never needs to (and no longer does) send the code again.
    request.session["verified_code"] = processed_result

    # `received` (an echo of the submitted code) and `returned_result` (the
    # internal variable name in CodeService) are both gone. Neither was read
    # by the client, and an authentication endpoint should not be echoing the
    # credential it was handed back at whoever sent it.
    return {
        "status": "success",
        "verifiedCode": processed_result
    }