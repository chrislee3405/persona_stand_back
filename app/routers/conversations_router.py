from fastapi import APIRouter, BackgroundTasks, Depends, Request, HTTPException

from app.services.chat_service import ChatService
from app.dependencies.session import get_or_create_session_id, get_verified_code, get_client_ip
from app.validators.chat_validator import ChatMessageCreate

# Domain errors raised below this point (privacy, consent, rate control,
# message length, conversation ownership) are mapped to HTTP status codes by
# the exception handlers registered in app/main.py, so both handlers here
# stay free of the identical six-clause try/except they used to carry.
router = APIRouter()


@router.post("/api/guestchat")
async def guestchat(payload: ChatMessageCreate, request: Request, background_tasks: BackgroundTasks, chat_service: ChatService = Depends()):
    """
    Handles POST /api/guestchat: routes a guest user's message to ChatService and maps its errors to HTTP responses.

    Parameters:
    - payload (ChatMessageCreate): conversationId and text — comes from the validated request body
    - request (Request): the incoming request — comes from FastAPI, used to read/write the session cookie
    - background_tasks (BackgroundTasks): task queue — comes from FastAPI, passed through to ChatService
    - chat_service (ChatService): persists the message, generates the AI reply, and schedules summarization — injected by FastAPI

    Returns:
    - dict: reply text, sender, and conversationId — sent back to the client as the JSON response.
      Domain errors propagate to the handlers in app/main.py, which turn them into 400/403/404/413/429.
    """
    session_id = get_or_create_session_id(request)
    client_ip = get_client_ip(request)

    return await chat_service.handle_chat_turn(
        session_id=session_id,
        code=None,  # guest chat has no code
        conversation_id=payload.conversationId,
        user_text=payload.text,
        background_tasks=background_tasks,
        client_ip=client_ip
    )


@router.post("/api/invitechat")
async def invitechat(payload: ChatMessageCreate, request: Request, background_tasks: BackgroundTasks, chat_service: ChatService = Depends()):
    """
    Handles POST /api/invitechat: routes a verified user's message to ChatService and maps its errors to HTTP responses.

    Parameters:
    - payload (ChatMessageCreate): conversationId and text — comes from the validated request body
    - request (Request): the incoming request — comes from FastAPI, used to read the session cookie
    - background_tasks (BackgroundTasks): task queue — comes from FastAPI, passed through to ChatService
    - chat_service (ChatService): persists the message, generates the AI reply, and schedules summarization — injected by FastAPI

    Returns:
    - dict: reply text, sender, and conversationId — sent back to the client as the JSON response
    """
    session_id = get_or_create_session_id(request)
    verified_code = get_verified_code(request)
    client_ip = get_client_ip(request)

    if not verified_code:
        # Session hasn't verified a code — client-supplied "code" is no
        # longer accepted here, so this is the only way in.
        raise HTTPException(status_code=401, detail="invite code not verified for this session")

    return await chat_service.handle_chat_turn(
        session_id=session_id,
        code=verified_code,
        conversation_id=payload.conversationId,
        user_text=payload.text,
        background_tasks=background_tasks,
        client_ip=client_ip
    )
