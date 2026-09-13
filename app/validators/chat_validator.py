from pydantic import BaseModel, ConfigDict, Field


class ChatMessageCreate(BaseModel):
    """Validates the body of a POST /api/guestchat or /api/invitechat request."""
    model_config = ConfigDict(extra="forbid")

    # Length-bounded, matching CodeSubmission.conversationId. It goes
    # straight into a WHERE clause -- parameterised, so not an injection, but
    # there is no reason to carry an arbitrarily long string that far. Backend
    # ids are UUIDs.
    conversationId: str | None = Field(default=None, max_length=64)
    text: str = Field(..., min_length=1)
