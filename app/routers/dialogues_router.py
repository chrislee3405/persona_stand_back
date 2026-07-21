from fastapi import APIRouter

router = APIRouter()



@router.get("/guestchat")
async def get_chat_dialogues():
    return {"reach guestchat"}


