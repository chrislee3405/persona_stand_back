"""Coordinate chat and session rotation across workers without locking message rows.

A transaction-scoped advisory lock uses one connection while work runs. Contended
waiters release their connection before retrying, so they cannot exhaust the pool
and prevent the owner from completing its normal database writes. PostgreSQL
releases the lock on rollback, cancellation or connection/process loss.
"""
import asyncio
import hashlib
from contextlib import asynccontextmanager
from contextvars import ContextVar

from sqlalchemy import text

from app.database import coordination_engine

_owned: ContextVar[frozenset[str]] = ContextVar("session_work", default=frozenset())


@asynccontextmanager
async def session_work(session_id: str):
    if session_id in _owned.get():
        yield
        return
    key = int.from_bytes(hashlib.sha256(("persona-turn:" + session_id).encode()).digest()[:8], "big", signed=True)
    while True:
        async with coordination_engine.connect() as connection:
            async with connection.begin():
                acquired = await connection.scalar(
                    text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": key}
                )
                if acquired:
                    token = _owned.set(_owned.get() | {session_id})
                    try:
                        yield
                    finally:
                        _owned.reset(token)
                    return
        await asyncio.sleep(0.05)
