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


def get_client_ip(request: Request) -> str:
    """
    Best-effort client IP for rate-limiting purposes (see RateControlService.reserve_ip_slot).

    Parameters:
    - request (Request): the incoming request — comes from the router handler that calls this

    Returns:
    - str: the first hop in X-Forwarded-For if present (set by persona_stand_front's nginx.conf when it proxies /api/ to this backend), otherwise the direct TCP peer address. X-Forwarded-For is attacker-controlled if this backend is ever reachable directly rather than only through that reverse proxy -- docker-compose.ec2.yml/Part_A.md already recommend not exposing the backend's port 8000 publicly in production for this reason.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"