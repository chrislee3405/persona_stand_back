import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

def setup_middleware(app: FastAPI) -> None:
    """
    Attaches CORS and signed-session-cookie middleware to the FastAPI app.

    Parameters:
    - app (FastAPI): the application instance — comes from app/main.py at startup

    Returns:
    - None: configures `app` in place by registering CORSMiddleware and SessionMiddleware
    """
    # NOT the path development or production actually uses. Both run the
    # frontend behind nginx (persona_stand_front/nginx.conf), which proxies
    # /api/ to this backend, so every browser request is SAME-ORIGIN and
    # never triggers a CORS preflight at all. `npm run dev` (Vite on 5173
    # talking cross-origin to :8000) is not a supported setup -- the Vite
    # config has no /api proxy. This list is kept only so that a direct
    # cross-origin frontend would work if one is ever introduced; if you are
    # wondering why editing it changes nothing, that is why.
    origins = [
        "http://localhost:80",
        "http://localhost:5173",  # unused: see the note above
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,      # Allows cookies/session tracking if needed later
        allow_methods=["*"],         # Allows GET, POST, OPTIONS, etc.
        allow_headers=["*"],         # Allows headers like Content-Type
    )

    # Signed, httpOnly session cookie. This is the actual authentication
    # artifact for every request from here on — the client cannot read or
    # forge it, unlike a plain value the JS controls. Fail loudly if the
    # secret isn't configured rather than silently falling back to a
    # guessable default.
    session_secret = os.environ["SESSION_SECRET_KEY"]
    is_prod = os.environ.get("ENV", "development") == "production"

    app.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        session_cookie="session",
        same_site="lax",       # switch to "none" (+ https_only=True) if frontend/backend are on different domains
        https_only=is_prod,    # True in prod; False only so local http dev still works
        max_age=60 * 60 * 24 * 30,  # 30 days
    )