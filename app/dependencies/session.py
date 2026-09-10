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
    - str: the client IP as seen by the reverse proxy (X-Real-IP), falling back to the direct TCP peer address.

    WHY NOT X-Forwarded-For. This used to read the first hop of
    X-Forwarded-For, which is attacker-controlled EVEN BEHIND THE PROXY.
    persona_stand_front/nginx.conf sets
    `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for`, and
    $proxy_add_x_forwarded_for is defined as "the client's own
    X-Forwarded-For, comma-appended with $remote_addr" -- so it PRESERVES
    whatever the client sent, as element zero. A request carrying
    `X-Forwarded-For: 203.0.113.99` arrived here as
    `203.0.113.99, <real ip>`, and .split(",")[0] returned the attacker's
    value. That defeated RateControlService's per-IP backstop, which is
    the only limit not keyed on something the caller already controls, and
    therefore the only thing bounding LLM spend from one machine.

    X-Real-IP is set by that same nginx block from $remote_addr -- the TCP
    peer address, which a client cannot forge -- and nginx OVERWRITES it
    rather than appending, so a client-supplied X-Real-IP is discarded.

    If a second proxy is ever put in front (an ALB, CloudFront), it must
    either set X-Real-IP itself or nginx must be given
    `set_real_ip_from <trusted cidr>` + `real_ip_header X-Forwarded-For`
    so $remote_addr becomes the true client address. Do not go back to
    reading X-Forwarded-For here.
    """
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"