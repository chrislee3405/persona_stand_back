import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.ai import gemini_service
from app.services.chat_service import ChatService
from app.services.model_collaborate import turn_metrics


def record_example():
    turn_metrics.record_usage(100, 20, thinking_tokens=30, cached_tokens=40, total_tokens=150)


async def test_parallel_stages_and_retries_keep_counts_separate():
    async def work(name, count):
        with turn_metrics.stage(name):
            for _ in range(count):
                with turn_metrics.model_call("test-model"):
                    await asyncio.sleep(0)
                    record_example()
    with turn_metrics.turn() as metrics:
        await asyncio.gather(work("topics", 1), work("gate", 3))
    result = metrics.snapshot(status="respond", kind="message")
    assert result["stages"]["gate"]["calls"] == 3
    assert result["stages"]["topics"]["tokens"]["total_tokens"] == 150
    assert result["total"] == dict(input_tokens=400, output_tokens=80, thinking_tokens=120,
                                  cached_tokens=160, total_tokens=600)
    # Cached is a subset of input, not another charge added to total.


async def test_sdk_usage_is_recorded_even_when_structured_reply_cannot_be_parsed(monkeypatch):
    response = SimpleNamespace(text="invalid json", usage_metadata=SimpleNamespace(
        prompt_token_count=100, candidates_token_count=20, thoughts_token_count=30,
        cached_content_token_count=40, total_token_count=150))
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=response))))
    monkeypatch.setattr(gemini_service, "_get_client", lambda: client)
    with turn_metrics.turn() as metrics, turn_metrics.stage("grounding"):
        with pytest.raises(ValueError):
            await gemini_service.GeminiService().call_model_structured("test-model", "private input", "system", {})
    result = metrics.snapshot(status="failed", kind="message")
    request = result["stages"]["grounding"]["requests"][0]
    assert request["status"] == "failed" and request["total_tokens"] == 150
    assert "private input" not in str(result)


def test_unknown_usage_remains_unknown_and_known_attempt_is_preserved():
    with turn_metrics.turn() as metrics, turn_metrics.stage("gate"):
        with turn_metrics.model_call("model"):
            record_example()
        with pytest.raises(RuntimeError), turn_metrics.model_call("model"):
            raise RuntimeError("provider unavailable")
    result = metrics.snapshot(status="failed", kind="message")
    assert result["total"]["total_tokens"] is None
    assert result["stages"]["gate"]["requests"][0]["total_tokens"] == 150
    assert result["stages"]["gate"]["calls"] == 2


@pytest.mark.parametrize("status", ["respond", "wait", "no_reply", "failed", "timeout"])
async def test_chat_saves_usage_without_needing_a_reply(status, monkeypatch):
    service = ChatService.__new__(ChatService)
    service.conversation_service = SimpleNamespace(record_token_usage=AsyncMock())

    async def operation():
        with turn_metrics.turn(), turn_metrics.stage("readiness"), turn_metrics.model_call("model"):
            record_example()
            if status == "failed":
                raise RuntimeError("failed")
            if status == "timeout":
                await asyncio.sleep(10)
            return SimpleNamespace(decision=status)

    monkeypatch.setattr("app.services.chat_service.TURN_DEADLINE_SECONDS", .02)
    if status in ("failed", "timeout"):
        with pytest.raises((RuntimeError, asyncio.TimeoutError)):
            await service._run_measured_model(operation(), "conv", 4, kind="continue")
    else:
        await service._run_measured_model(operation(), "conv", 4, kind="continue")
    args = service.conversation_service.record_token_usage.await_args.args
    assert args[:2] == ("conv", 4)
    assert args[2]["status"] == status and args[2]["kind"] == "continue"
    assert args[2]["total"]["total_tokens"] == 150


async def test_logging_failure_does_not_replace_reply():
    service = ChatService.__new__(ChatService)
    service.conversation_service = SimpleNamespace(record_token_usage=AsyncMock(side_effect=RuntimeError("write failed")))
    outcome = SimpleNamespace(decision="respond")
    assert await service._run_measured_model(AsyncMock(return_value=outcome)(), "conv", 0, kind="message") is outcome
