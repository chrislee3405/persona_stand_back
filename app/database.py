import os
from typing import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

load_dotenv()

_RAW_DATABASE_URL = os.getenv("DATABASE_URL")
if not _RAW_DATABASE_URL:
    # Explicit, rather than letting create_async_engine(None) surface as
    # `ArgumentError: Expected string or URL object, got None` from somewhere
    # inside SQLAlchemy. This is the single most common first-run failure --
    # see persona_stand_ec2yml/Part_C.md -- so it should name itself.
    raise RuntimeError(
        "DATABASE_URL is not set. Copy .env.example to .env and fill it in "
        "(local development), or add it to the instance's .env (deployment) -- "
        "see persona_stand_ec2yml/Part_C.md."
    )

# Query parameters SQLAlchemy's asyncpg dialect will not accept on the URL --
# it forwards unknown ones straight to asyncpg.connect(), which raises
# `TypeError: connect() got an unexpected keyword argument 'sslmode'`. They are
# lifted off the URL here and translated into connect_args below, so ONE
# DATABASE_URL keeps working for both this app and any psql/psycopg/pgAdmin
# tooling pointed at the same string.
_PSYCOPG_ONLY_PARAMS = {"sslmode", "sslrootcert", "sslcert", "sslkey", "target_session_attrs"}

# The sslmode values asyncpg understands. It uses libpq's exact vocabulary, so
# a mode lifted off the URL is passed through verbatim rather than reduced to
# a bool -- see the note in _to_async_url.
_ASYNCPG_SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}


def _to_async_url(url: str) -> tuple[str, dict]:
    """
    Rewrites a psycopg2-style DATABASE_URL into the asyncpg dialect, lifting driver-incompatible query parameters into connect_args.

    Parameters:
    - url (str): the raw DATABASE_URL -- comes from the environment

    Returns:
    - tuple[str, dict]: (async URL, connect_args) -- goes to create_async_engine.
      `postgres://` and `postgresql://` both become `postgresql+asyncpg://`; an
      explicit `+driver` already in the URL is left alone. An `sslmode` is
      moved into connect_args unchanged.

    SSLMODE IS PASSED THROUGH AS A STRING, not reduced to a bool. asyncpg
    understands libpq's exact vocabulary (disable/allow/prefer/require/
    verify-ca/verify-full), so forwarding the mode verbatim preserves whatever
    the URL asked for. Collapsing it to `ssl=True` would NOT: asyncpg reads a
    bare True as "build a default SSL context", i.e. verify the certificate
    chain AND the hostname -- which is verify-full, strictly stronger than the
    `require` that was written. That silently upgrades an RDS connection from
    "encrypt, don't verify" to "encrypt and verify", and fails where psycopg2
    succeeded.

    With no sslmode at all, nothing is set and asyncpg falls back to `prefer`
    for TCP connections -- the same default psycopg2 uses, so an unchanged
    DATABASE_URL behaves the same before and after the driver swap.
    """
    split = urlsplit(url)
    scheme = split.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"

    query_pairs = parse_qsl(split.query, keep_blank_values=True)
    kept: list[tuple[str, str]] = []
    connect_args: dict = {}
    for key, value in query_pairs:
        if key.lower() not in _PSYCOPG_ONLY_PARAMS:
            kept.append((key, value))
            continue
        if key.lower() == "sslmode":
            mode = value.strip().lower()
            if mode not in _ASYNCPG_SSL_MODES:
                raise RuntimeError(
                    f"DATABASE_URL has sslmode={value!r}, which is not one of: "
                    + ", ".join(sorted(_ASYNCPG_SSL_MODES))
                )
            connect_args["ssl"] = mode

    async_url = urlunsplit((scheme, split.netloc, split.path, urlencode(kept), split.fragment))
    return async_url, connect_args


DATABASE_URL, _CONNECT_ARGS = _to_async_url(_RAW_DATABASE_URL)

# ASYNC, not sync. Every route handler in this app is `async def`, so FastAPI
# runs it ON the event loop rather than in a threadpool -- which meant that
# with a sync psycopg2 engine, every .commit() and every .query() blocked the
# single uvicorn worker for a network round trip to RDS. One visitor mid-chat
# stalled GET /api/site-content for everybody, which is the same class of bug
# GeminiService already fixed for its HTTPS calls (see its class docstring).
#
# pool_size/max_overflow are explicit rather than left at SQLAlchemy's 5+10
# default: a chat turn holds its connection for the length of the turn, so the
# pool is a real concurrency ceiling and should be a stated one.
# pool_pre_ping survives RDS dropping idle connections, which it does.
engine = create_async_engine(
    DATABASE_URL,
    connect_args=_CONNECT_ARGS,
    pool_size=10,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
)

# expire_on_commit=False, deliberately. With the default (True) every commit
# expires every instance in the identity map, so the NEXT attribute read
# re-SELECTs it one row at a time. ResponseGate commits a review row on every
# rejected attempt, and the conversation history loaded by ContextGatherer is
# still being read after those commits -- which turned a 20-message history
# into ~20 extra round trips per rejected attempt, invisibly, inside what
# reads as a plain list comprehension in prepare_history.
#
# Nothing in this app depends on post-commit refresh semantics: the one place
# that needs a server-generated value back (append_message) calls
# db.refresh(message) explicitly.
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,
)

Base = declarative_base()


async def get_db() -> AsyncIterator[AsyncSession]:
    """
    Provides a database session for the duration of one request, closing it afterward.

    Parameters:
    - none

    Returns:
    - AsyncSession: yielded to FastAPI's dependency injection — used by services, closed automatically when the request finishes.

    NOT for background tasks. FastAPI runs a yield-dependency's exit code
    BEFORE the response's background tasks, so a task holding this session is
    using one that has already been closed. Anything scheduled with
    BackgroundTasks must open its own session -- see
    SummarizationService.summarize_conversation_if_needed, which does.
    """
    async with SessionLocal() as db:
        yield db
