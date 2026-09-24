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


class ChatContinueRequest(BaseModel):
    """Validates the body of a POST /api/guestchat/continue or /api/invitechat/continue request."""
    model_config = ConfigDict(extra="forbid")

    # Required: a continue answers what an existing conversation is holding,
    # so there is nothing to do without one. No text field at all -- the held
    # messages are read from the database, never resent by the client.
    conversationId: str = Field(..., min_length=1, max_length=64)
