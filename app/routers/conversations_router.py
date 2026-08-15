from fastapi import APIRouter, Depends, Request, HTTPException
from app.services.conversation_service import ConversationService, ConversationNotFoundError

router = APIRouter()


@router.post("/api/guestchat")
async def guestchat(request: Request, service: ConversationService = Depends(ConversationService)):
    print(">>> BUTTON CLICK EVENT RECEIVED BY BACKEND <<<")
    body = await request.json()
    conversation_id = body.get("conversationId")  # may be None on first message
    user_text = body.get("text")

    if not user_text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        # 1. Persist the user's message immediately.
        #    If conversation_id is None, a new conversation is created here.
        _, conversation_id = await service.append_message(
            conversation_id=conversation_id,
            code=None,  # guest chat has no code
            sender="user",
            text=user_text
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversationId not found")

    # 2. Generate the reply (placeholder — replace with your actual logic).
    reply_text = f"reach guestchat: {user_text}"

    # 3. Persist the backend's reply — conversation_id is now guaranteed to exist.
    await service.append_message(
        conversation_id=conversation_id,
        code=None,
        sender="backend",
        text=reply_text
    )

    return {
        "text": reply_text,
        "sender": "backend",
        "conversationId": conversation_id  # frontend captures this on first message
    }


@router.post("/api/invitechat")
async def invitechat(request: Request, service: ConversationService = Depends(ConversationService)):
    body = await request.json()
    conversation_id = body.get("conversationId")  # may be None on first message
    user_text = body.get("text")
    code = body.get("code")

    if not user_text:
        raise HTTPException(status_code=400, detail="text is required")

    if not code:
        raise HTTPException(status_code=400, detail="code is required for invitechat")

    try:
        _, conversation_id = await service.append_message(
            conversation_id=conversation_id,
            code=code,
            sender="user",
            text=user_text
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="conversationId not found")

    reply_text = f"reach invitechat: {user_text}"

    await service.append_message(
        conversation_id=conversation_id,
        code=code,
        sender="backend",
        text=reply_text
    )

    return {
        "text": reply_text,
        "sender": "backend",
        "conversationId": conversation_id
    }