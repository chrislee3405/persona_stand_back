from fastapi import APIRouter, Depends, Request, HTTPException
from app.services.conversation_manage_service import (
    ConversationService, ConversationNotFoundError, ConversationAccessDeniedError
)
from app.services.model_collarborate_service import ModelCollaborateService
from app.dependencies.session import get_or_create_session_id, get_verified_code

router = APIRouter()


@router.post("/api/guestchat")
async def guestchat(
    request: Request,
    service: ConversationService = Depends(),
    ai_service: ModelCollaborateService = Depends()
):
    session_id = get_or_create_session_id(request)

    body = await request.json()
    conversation_id = body.get("conversationId")  # may be None on first message
    user_text = body.get("text")

    if not user_text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        # 1. Persist the user's message immediately.
        #    If conversation_id is None, a new conversation is created here,
        #    owned by this session. If it's provided, append_message checks
        #    that this session actually owns it before touching anything.
        _, conversation_id = await service.append_message(
            conversation_id=conversation_id,
            code=None,  # guest chat has no code
            session_id=session_id,
            sender="user",
            text=user_text
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversationId not found")
    except ConversationAccessDeniedError:
        # Same response as "not found" — don't confirm to an attacker that
        # a conversation_id they don't own actually exists.
        raise HTTPException(status_code=404, detail="conversationId not found")

    # 2. Generate the reply via the AI flow.
    reply_text = await ai_service.route_user_message(user_text)

    # 3. Persist the backend's reply — conversation_id is now guaranteed to exist
    #    and to be owned by this session.
    await service.append_message(
        conversation_id=conversation_id,
        code=None,
        session_id=session_id,
        sender="backend",
        text=reply_text
    )

    return {
        "text": reply_text,
        "sender": "backend",
        "conversationId": conversation_id  # frontend captures this on first message
    }


@router.post("/api/invitechat")
async def invitechat(
    request: Request,
    service: ConversationService = Depends(),
    ai_service: ModelCollaborateService = Depends()
):
    session_id = get_or_create_session_id(request)
    verified_code = get_verified_code(request)

    if not verified_code:
        # Session hasn't verified a code — client-supplied "code" is no
        # longer accepted here, so this is the only way in.
        raise HTTPException(status_code=401, detail="invite code not verified for this session")

    body = await request.json()
    conversation_id = body.get("conversationId")  # may be None on first message
    user_text = body.get("text")

    if not user_text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        _, conversation_id = await service.append_message(
            conversation_id=conversation_id,
            code=verified_code,
            session_id=session_id,
            sender="user",
            text=user_text
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversationId not found")
    except ConversationAccessDeniedError:
        raise HTTPException(status_code=404, detail="conversationId not found")

    reply_text = await ai_service.route_user_message(user_text)

    await service.append_message(
        conversation_id=conversation_id,
        code=verified_code,
        session_id=session_id,
        sender="backend",
        text=reply_text
    )

    return {
        "text": reply_text,
        "sender": "backend",
        "conversationId": conversation_id
    }