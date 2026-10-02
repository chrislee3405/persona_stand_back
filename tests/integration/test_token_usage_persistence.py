import asyncio

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import undefer

from app.database import SessionLocal, engine
from app.models.conversation import Message
from app.services.conversation_manage_service import ConversationService
from app.services.schema_readiness import require_chat_schema
from app.services.model_collaborate import turn_metrics

pytestmark = pytest.mark.integration


def instrument_model(fake, monkeypatch):
    for method in ("call_model", "call_model_structured"):
        original = getattr(fake, method)

        async def measured(*args, _original=original, **kwargs):
            with turn_metrics.model_call(kwargs.get("model_name", "test-model")):
                turn_metrics.record_usage(100, 20, thinking_tokens=30, cached_tokens=0, total_tokens=150)
                return await _original(*args, **kwargs)

        monkeypatch.setattr(fake, method, measured)


async def test_wait_and_continue_usage_stays_on_one_user_row(consented_client, fake_gemini, monkeypatch):
    instrument_model(fake_gemini, monkeypatch)
    fake_gemini.readiness_decision = "wait"
    response = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    cid = response.json()["conversationId"]
    assert response.json()["status"] == "wait"
    response = await consented_client.post("/api/guestchat/continue", json={"conversationId": cid})
    assert response.json()["sender"] == "backend"
    assert "token_usage" not in response.text
    async with SessionLocal() as session:
        rows = list((await session.scalars(select(Message).options(undefer(Message.token_usage))
                                          .where(Message.conversation_id == cid).order_by(Message.order_index))).all())
        runs = rows[0].token_usage["runs"]
        assert [r["kind"] for r in runs] == ["message", "continue"]
        assert runs[0]["stages"]["readiness"]["tokens"]["total_tokens"] == 150
        assert runs[1]["stages"]["reply"]["tokens"]["total_tokens"] == 150
        assert runs[1]["stages"]["gate"]["tokens"]["thinking_tokens"] == 30
        assert rows[1].token_usage is None


async def test_failed_model_usage_is_retained_on_retagged_user(consented_client, fake_gemini, monkeypatch):
    instrument_model(fake_gemini, monkeypatch)
    fake_gemini.failure = RuntimeError("test failure")
    response = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    cid = response.json()["conversationId"]
    assert response.json()["userMessageKept"] is False
    async with SessionLocal() as session:
        row = (await session.scalars(select(Message).options(undefer(Message.token_usage))
                                     .where(Message.conversation_id == cid, Message.order_index == 0))).one()
        assert row.sender == "not_saved_user"
        run = row.token_usage["runs"][0]
        assert run["status"] == "failed"
        assert run["stages"]["reply"]["requests"][0]["total_tokens"] == 150


async def test_background_summary_has_separate_usage_run(db, fake_gemini, monkeypatch):
    from app.services.model_collaborate.summarization_service import SummarizationService
    instrument_model(fake_gemini, monkeypatch)
    service = ConversationService(db)
    cid = None
    for index in range(10):
        _, cid = await service.append_message(cid, None, "owner", "user" if index % 2 == 0 else "backend", "example")
    await service.record_token_usage(cid, 8, {"run_id": "earlier-message"})
    await SummarizationService(fake_gemini).summarize_conversation_if_needed(cid)
    async with SessionLocal() as session:
        usage = await session.scalar(select(Message.token_usage).where(Message.conversation_id == cid, Message.order_index == 8))
        assert usage["runs"][0]["run_id"] == "earlier-message"
        run = usage["runs"][1]
        assert run["kind"] == "summary" and run["status"] == "completed"
        assert run["stages"]["summary"]["tokens"]["total_tokens"] == 150


async def test_concurrent_run_append_is_lossless_idempotent_and_not_in_prompt(db):
    service = ConversationService(db)
    _, cid = await service.append_message(None, None, "owner", "user", "question")

    async def save(run_id):
        async with SessionLocal() as session:
            await ConversationService(session).record_token_usage(cid, 0, {"run_id": run_id, "total": {"total_tokens": 123}})

    await asyncio.gather(save("message"), save("continue"), save("summary"))
    await save("message")
    async with SessionLocal() as session:
        message = (await session.scalars(select(Message).options(undefer(Message.token_usage)).where(Message.conversation_id == cid))).one()
        assert len(message.token_usage["runs"]) == 3
        assert {r["run_id"] for r in message.token_usage["runs"]} == {"message", "continue", "summary"}
        assert message.text == "question" and message.sender == "user"
        from app.services.model_collaborate.prepare_history import prepare_history
        assert prepare_history([message], None) == "User: question"


async def test_startup_rejects_unmigrated_database_and_accepts_migration(db):
    from pathlib import Path
    import asyncpg
    import os
    async with engine.begin() as connection:
        await connection.execute(text("ALTER TABLE message DROP COLUMN token_usage"))
    try:
        async with engine.connect() as connection:
            with pytest.raises(RuntimeError, match="Message token usage migration required"):
                await require_chat_schema(connection)
        connection = await asyncpg.connect(os.environ["TEST_DATABASE_URL"].replace("postgresql+asyncpg:", "postgresql:"))
        try:
            sql = (Path(__file__).resolve().parents[2] / "scripts/migrations/20261002_message_token_usage.sql").read_text()
            await connection.execute(sql)
        finally:
            await connection.close()
        async with engine.connect() as connection:
            await require_chat_schema(connection)
    finally:
        async with engine.begin() as connection:
            await connection.execute(text("ALTER TABLE message ADD COLUMN IF NOT EXISTS token_usage jsonb"))
