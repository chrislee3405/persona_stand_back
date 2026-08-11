from fastapi import APIRouter, Depends, Request
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/api", tags=["conversation"])


@router.get("/guestchat")
async def get_chat_dialogues():
    return {"reach guestchat"}


@router.post("/save-conversation")
async def save_conversation(request: Request, service: ConversationService = Depends()):
    body = await request.json()

    await service.save_conversation(
        content=body.get("content", []),
        code=body.get("code")
    )

    return {"status": "saved"}