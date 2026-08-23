import uuid
from fastapi import Request


def get_or_create_session_id(request: Request) -> str:
    """
    Ensures the current request's session has a stable session_id, creating one on first contact.

    Parameters:
    - request (Request): the incoming request — comes from the router handler that calls this

    Returns:
    - str: the session_id — read from or written into request.session, persisted into the signed cookie by SessionMiddleware
    """
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]


def get_verified_code(request: Request) -> str | None:
    """
    Checks whether the current session has already verified an invite code.

    Parameters:
    - request (Request): the incoming request — comes from the router handler that calls this

    Returns:
    - str | None: the verified code stored in request.session, or None if this session hasn't verified one yet
    """
    return request.session.get("verified_code")