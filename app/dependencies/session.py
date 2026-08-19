import uuid
from fastapi import Request


def get_or_create_session_id(request: Request) -> str:
    """
    Ensures this request's session has a stable session_id, creating one on
    first contact — guest or not. Starlette's SessionMiddleware serializes
    request.session back into the signed, httpOnly cookie automatically at
    the end of the request, so nothing extra needs to happen here.
    """
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]


def get_verified_code(request: Request) -> str | None:
    """None if this session hasn't verified an invite code yet."""
    return request.session.get("verified_code")