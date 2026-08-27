from fastapi import APIRouter, BackgroundTasks, Depends, Request, HTTPException
from app.services.conversation_manage_service import ConversationNotFoundError, ConversationAccessDeniedError
from app.services.chat_service import ChatService, MessageTooLongError, MAX_MESSAGE_LENGTH
from app.services.privacy_gate_service import PrivacyViolationError
from app.services.rate_control_service import TooManyPendingMessagesError, TooManyPendingMessagesFromIpError
from app.services.consent_service import ConsentRequiredError
from app.dependencies.session import get_or_create_session_id, get_verified_code, get_client_ip
from app.validators.chat_validator import ChatMessageCreate

router = APIRouter()

_PRIVACY_VIOLATION_DETAIL = "Your message appears to contain personal or sensitive information (e.g. a email, phone number, or ID). Please remove it and try again."
_TOO_MANY_PENDING_DETAIL = "You're sending messages faster than they can be answered. Please wait for a reply before sending another."
_CONSENT_REQUIRED_DETAIL = "You must agree to the data collection notice before sending messages."
_MESSAGE_TOO_LONG_DETAIL = f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."


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
    client_ip = get_client_ip(request)

    try:
        return await chat_service.handle_chat_turn(
            session_id=session_id,
            code=None,  # guest chat has no code
            conversation_id=payload.conversationId,
            user_text=payload.text,
            background_tasks=background_tasks,
            client_ip=client_ip
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversationId not found")
    except ConversationAccessDeniedError:
        # Same response — don't give info that a conversation_id actually exists.
        raise HTTPException(status_code=404, detail="conversationId not found")
    except PrivacyViolationError:
        raise HTTPException(status_code=400, detail=_PRIVACY_VIOLATION_DETAIL)
    except (TooManyPendingMessagesError, TooManyPendingMessagesFromIpError):
        raise HTTPException(status_code=429, detail=_TOO_MANY_PENDING_DETAIL)
    except ConsentRequiredError:
        raise HTTPException(status_code=403, detail=_CONSENT_REQUIRED_DETAIL)
    except MessageTooLongError:
        raise HTTPException(status_code=413, detail=_MESSAGE_TOO_LONG_DETAIL)


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

    try:
        return await chat_service.handle_chat_turn(
            session_id=session_id,
            code=verified_code,
            conversation_id=payload.conversationId,
            user_text=payload.text,
            background_tasks=background_tasks,
            client_ip=client_ip
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversationId not found")
    except ConversationAccessDeniedError:
        raise HTTPException(status_code=404, detail="conversationId not found")
    except PrivacyViolationError:
        raise HTTPException(status_code=400, detail=_PRIVACY_VIOLATION_DETAIL)
    except (TooManyPendingMessagesError, TooManyPendingMessagesFromIpError):
        raise HTTPException(status_code=429, detail=_TOO_MANY_PENDING_DETAIL)
    except ConsentRequiredError:
        raise HTTPException(status_code=403, detail=_CONSENT_REQUIRED_DETAIL)
    except MessageTooLongError:
        raise HTTPException(status_code=413, detail=_MESSAGE_TOO_LONG_DETAIL)
