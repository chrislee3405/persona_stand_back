from pydantic import BaseModel, ConfigDict, Field


class ChatMessageCreate(BaseModel):
    """Validates the body of a POST /api/guestchat or /api/invitechat request."""
    model_config = ConfigDict(extra="forbid")

    conversationId: str | None = None
    text: str = Field(..., min_length=1)
