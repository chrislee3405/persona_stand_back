from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel

from app.services.code_service import CodeService, InvalidCodeError
from app.services.conversation_manage_service import ConversationAccessDeniedError
from app.dependencies.session import get_or_create_session_id


router = APIRouter()

class CodeSubmission(BaseModel):
    input_code: str
    conversation_id: str | None = None

@router.post("/api/code")
async def verify_code(
    payload: CodeSubmission,
    request: Request,
    service: CodeService = Depends()
):
    session_id = get_or_create_session_id(request)

    try:
        processed_result = await service.match_code(
            input_code=payload.input_code,
            conversation_id=payload.conversation_id,
            session_id=session_id
        )
    except InvalidCodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Process result not found"
        )
    except ConversationAccessDeniedError:
        # The code itself was valid, but this session didn't create the
        # conversation it tried to upgrade — don't leak which is true.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="cannot link code to a conversation this session doesn't own"
        )

    # This is the actual authorization event. Every /api/invitechat call
    # from here on derives its verified status from this session cookie —
    # the client never needs to (and no longer does) send the code again.
    request.session["verified_code"] = processed_result

    return {
        "status": "success",
        "received": payload.input_code,
        "returned_result": processed_result
    }