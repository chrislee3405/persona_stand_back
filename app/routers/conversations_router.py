from fastapi import APIRouter, BackgroundTasks, Depends, Request, HTTPException
from app.services.conversation_manage_service import ConversationNotFoundError, ConversationAccessDeniedError
from app.services.chat_service import ChatService
from app.dependencies.session import get_or_create_session_id, get_verified_code
from app.validators.chat_validator import ChatMessageCreate

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
    - dict: reply text, sender, and conversationId — sent back to the client as the JSON response
    """
    session_id = get_or_create_session_id(request)

    try:
        return await chat_service.handle_chat_turn(
            session_id=session_id,
            code=None,  # guest chat has no code
            conversation_id=payload.conversationId,
            user_text=payload.text,
            background_tasks=background_tasks
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversationId not found")
    except ConversationAccessDeniedError:
        # Same response — don't give info that a conversation_id actually exists.
        raise HTTPException(status_code=404, detail="conversationId not found")


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

    if not verified_code:
        # Session hasn't verified a code — client-supplied "code" is no
        # longer accepted here, so this is the only way in.
        raise HTTPException(status_code=401, detail="invite code not verified for this session")

    try:
        return await chat_service.handle_chat_turn(
            session_id=session_id,
            code=verified_code,
            conversation_id=payload.conversationId,
            user_text=payload.text,
            background_tasks=background_tasks
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversationId not found")
    except ConversationAccessDeniedError:
        raise HTTPException(status_code=404, detail="conversationId not found")
