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


# The session key holding a verified session's invite code, BY DATABASE ID.
#
# An id, not the code. Starlette's SessionMiddleware SIGNS the session but does
# not ENCRYPT it -- the cookie value is base64(json) plus an HMAC -- so anything
# stored here can be read back by anyone holding the cookie with a one-line
# decode. This key used to hold the invite code itself, which quietly undid
# codes_router's decision not to echo the credential back: it was handed back
# anyway, in the Set-Cookie on the same response. An integer primary key is
# useless to anyone who reads it, and the signature still stops it being
# forged.
#
# It also gives revocation for free: delete a row from `code` and every
# session verified with it stops being verified on its next request, because
# the id no longer resolves (see conversations_router.invitechat).
_INVITE_CODE_ID_KEY = "invite_code_id"

# The key sessions verified before that change still carry, holding the
# plaintext code. Never read -- only removed -- so a legacy cookie is simply
# treated as unverified and its holder re-enters their code once.
_LEGACY_VERIFIED_CODE_KEY = "verified_code"


def get_verified_invite_code_id(request: Request) -> int | None:
    """
    Reports which invite code, if any, the current session has verified -- as a database id, never the code itself.

    Parameters:
    - request (Request): the incoming request — comes from the router handler that calls this

    Returns:
    - int | None: the `code.id` recorded when this session verified, or None if it
      has not verified one. Reads the signed session only; it does NOT look the id
      up, so a returned id is a claim the session made at verification time, not
      proof the code still exists. Callers that act on it (invitechat) must
      resolve it; callers that only report state (chatroom_initialize) need not.

    A legacy plaintext `verified_code` key is dropped on sight rather than
    honoured or migrated: honouring it would keep the credential sitting in a
    readable cookie, and migrating it would mean re-checking the code here,
    which is the verification flow's job, not a session helper's.
    """
    request.session.pop(_LEGACY_VERIFIED_CODE_KEY, None)
    value = request.session.get(_INVITE_CODE_ID_KEY)
    # A signed session cannot be forged, but it can outlive a schema change;
    # anything that is not a plain int is treated as "not verified".
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def set_verified_invite_code_id(request: Request, invite_code_id: int) -> None:
    """
    Records that the current session has verified an invite code, by its database id.

    Parameters:
    - request (Request): the incoming request — comes from codes_router after a successful verification
    - invite_code_id (int): the matched `code.id` — comes from CodeService.match_code

    Returns:
    - None: writes the id into the signed session cookie
    """
    request.session.pop(_LEGACY_VERIFIED_CODE_KEY, None)
    request.session[_INVITE_CODE_ID_KEY] = invite_code_id


def clear_verified_invite_code(request: Request) -> None:
    """
    Removes the current session's invite-code verification.

    Parameters:
    - request (Request): the incoming request — comes from the router that found the recorded id no longer resolves

    Returns:
    - None: deletes both the id key and any legacy plaintext key from the session
    """
    request.session.pop(_INVITE_CODE_ID_KEY, None)
    request.session.pop(_LEGACY_VERIFIED_CODE_KEY, None)


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