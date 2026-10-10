from app.services.ai.provider_factory import AIService
"""Real API and disposable PostgreSQL; only model responses are synthetic."""
import json
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import undefer

from app.database import SessionLocal
from app.models.conversation import Conversation, Message
from app.services.conversation_manage_service import ConversationService
from app.services.model_collaborate.summarization_service import SummarizationService
from tests.fakes.fake_gemini import REPLY

pytestmark = pytest.mark.integration
PRIVATE_MARKER = "synthetic_private_provider_detail"


def stage_for(schema):
    properties = schema.get("properties", {})
    for key, stage in [("decision", "readiness"), ("document_topics", "topics"),
                       ("question_type", "grounding"), ("result", "gate")]:
        if key in properties:
            return stage
    return "split" if schema["type"].upper() == "ARRAY" else "example"


@pytest.mark.parametrize("stage", ["topics", "grounding", "reply", "gate"])
@pytest.mark.parametrize("mode", ["exception", "invalid"])
async def test_required_stage_failure_is_retryable_and_never_saved_as_candidate_reply(
        consented_client, fake_gemini, rate_control, monkeypatch, caplog, stage, mode):
    calls = []
    structured, text = fake_gemini.call_model_structured, fake_gemini.call_model
    def fault():
        if mode == "exception":
            raise TimeoutError(PRIVATE_MARKER)
        return None
    async def structured_call(**kwargs):
        current = stage_for(kwargs["schema"])
        calls.append(current)
        return fault() if current == stage else await structured(**kwargs)
    async def text_call(**kwargs):
        calls.append("reply")
        return fault() if stage == "reply" else await text(**kwargs)
    monkeypatch.setattr(fake_gemini, "call_model_structured", structured_call)
    monkeypatch.setattr(fake_gemini, "call_model", text_call)
    response = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    assert response.status_code == 200
    body = response.json()
    assert body["sender"] == "system" and body["userMessageKept"] is False
    assert "try again" in " ".join(body["turns"]).lower()
    assert "No supporting information" not in response.text
    assert calls.count(stage) == 1
    assert "split" not in calls
    if stage == "topics":
        assert not {"grounding", "reply", "gate"}.intersection(calls)
    if stage == "grounding":
        assert not {"reply", "gate"}.intersection(calls)
    if stage == "reply":
        assert "gate" not in calls
    assert rate_control._pending_counts == {} and rate_control._ip_pending_counts == {}
    cid = body["conversationId"]
    async with SessionLocal() as session:
        rows = list((await session.scalars(select(Message).options(undefer(Message.token_usage))
            .where(Message.conversation_id == cid).order_by(Message.order_index))).all())
        assert rows[0].sender == "not_saved_user"
        assert not any(row.sender == "backend" for row in rows)
        assert any(f"stage={stage}" in row.text for row in rows if row.sender == "error")
        run = rows[0].token_usage["runs"][0]
        assert run["status"] == "failed"
        assert run["stages"][stage]["failures"] == [{
            "reason": "call_failed" if mode == "exception" else "invalid_output", "fallback": "abort"}]
        assert PRIVATE_MARKER not in json.dumps(run) + " ".join(row.text for row in rows)
        service = ConversationService(session)
        assert await service.get_recent_messages(cid, exclude_last=False) == []
        assert await service.get_pending_user_messages(cid) == []
    assert PRIVATE_MARKER not in response.text + caplog.text
    # The same session and conversation can make a fresh successful attempt.
    monkeypatch.setattr(fake_gemini, "call_model_structured", structured)
    monkeypatch.setattr(fake_gemini, "call_model", text)
    retry = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project.", "conversationId": cid})
    assert retry.json()["sender"] == "backend" and retry.json()["turns"] == [REPLY]


async def test_successful_no_match_remains_an_honest_missing_information_reply(consented_client, fake_gemini, monkeypatch):
    original = fake_gemini.call_model_structured
    async def empty_search(**kwargs):
        stage = stage_for(kwargs["schema"])
        if stage == "topics":
            return {key: [] for key in kwargs["schema"]["properties"]}
        if stage == "grounding":
            return {"question_type": "factual", "coverage": "none", "claims": [],
                    "response_parts": [{"request": "project details", "action": "answer_factual"}],
                    "missing": "project details"}
        return await original(**kwargs)
    monkeypatch.setattr(fake_gemini, "call_model_structured", empty_search)
    reply = (await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})).json()
    assert reply["sender"] == "backend" and reply["userMessageKept"] is True
    assert reply["turns"] == ["I don't have that information for now."]
    assert "No supporting information" in reply["systemNotice"]


async def test_readiness_and_split_failures_preserve_verified_reply(consented_client, fake_gemini, monkeypatch):
    original = fake_gemini.call_model_structured
    calls = []
    async def failed_optional_stages(**kwargs):
        stage = stage_for(kwargs["schema"])
        calls.append(stage)
        if stage in {"readiness", "split"}:
            raise TimeoutError(PRIVATE_MARKER)
        return await original(**kwargs)
    # Force the otherwise short synthetic reply through bubble splitting.
    from app.services.model_collaborate.response_parser import ResponseParser
    original_parse = ResponseParser.parse
    async def parse(self, reply):
        self.min_chars_per_turn = 10
        return await original_parse(self, reply)
    monkeypatch.setattr(ResponseParser, "parse", parse)
    monkeypatch.setattr(fake_gemini, "call_model_structured", failed_optional_stages)
    body = (await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})).json()
    assert body["sender"] == "backend" and body["turns"] == [REPLY]
    assert calls.count("readiness") == calls.count("grounding") == calls.count("gate") == calls.count("split") == 1
    async with SessionLocal() as session:
        run = (await session.scalar(select(Message.token_usage).where(
            Message.conversation_id == body["conversationId"], Message.order_index == 0)))["runs"][0]
        assert run["stages"]["readiness"]["failures"] == [{"reason": "call_failed", "fallback": "respond"}]
        assert run["stages"]["split"]["failures"] == [{"reason": "call_failed", "fallback": "single_message"}]


@pytest.mark.parametrize("mode,value", [("exception", None), ("invalid", None), ("invalid", "  "), ("invalid", {})])
async def test_summary_failure_preserves_checkpoint_and_all_messages(db, fake_gemini, monkeypatch, caplog, mode, value):
    service = ConversationService(db)
    cid = None
    for index in range(12):
        _, cid = await service.append_message(cid, None, "owner", "user" if index % 2 == 0 else "backend", "Fictional example.")
    await service.mark_summarized_up_to(cid, 1, "Existing summary.")
    call = AsyncMock(side_effect=TimeoutError(PRIVATE_MARKER)) if mode == "exception" else AsyncMock(return_value=value)
    monkeypatch.setattr(fake_gemini, "call_model", call)
    await SummarizationService(AIService({"gemini": fake_gemini})).summarize_conversation_if_needed(cid)
    call.assert_awaited_once()
    async with SessionLocal() as session:
        conv = await session.get(Conversation, cid)
        assert conv.summary == "Existing summary." and conv.last_summarized_index == 1
        rows = list((await session.scalars(select(Message).options(undefer(Message.token_usage))
            .where(Message.conversation_id == cid).order_by(Message.order_index))).all())
        assert len(rows) == 12 and all(row.text == "Fictional example." for row in rows)
        run = rows[10].token_usage["runs"][0]
        assert run["kind"] == "summary" and run["status"] == "failed"
        assert run["stages"]["summary"]["failures"] == [{
            "reason": "call_failed" if mode == "exception" else "invalid_output", "fallback": "keep_summary"}]
    assert PRIVATE_MARKER not in caplog.text + json.dumps(run)
