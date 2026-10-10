"""Fault injection verifies recovery rules, not a live model's judgement."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.model_collaborate import turn_metrics
from app.services.model_collaborate.grounding_service import GroundingService
from app.services.model_collaborate.readiness_service import ReadinessService
from app.services.model_collaborate.response_gate import ResponseGate
from app.services.model_collaborate.response_parser import ResponseParser
from app.services.model_collaborate.stage_errors import ModelStageError
from app.services.model_collaborate_service import ModelCollaborateService
from tests.unit.test_context_gatherer import build, DOC_TOPICS, SCENARIO_TOPICS, CANDIDATES
from tests.unit.test_evidence_contract import context, contract, review, check, violation
from tests.unit.test_readiness_gate import build_model_service


PRIVATE_MARKER = "synthetic_private_provider_detail"


def failing_model(mode, value=None):
    kwargs = {"side_effect": TimeoutError(PRIVATE_MARKER)} if mode == "exception" else {"return_value": value}
    return SimpleNamespace(call_model=AsyncMock(**kwargs), call_model_structured=AsyncMock(**kwargs))


def failure(metrics, stage, reason, fallback):
    assert metrics.snapshot(status="failed", kind="message")["stages"][stage]["failures"] == [
        {"reason": reason, "fallback": fallback}]


@pytest.mark.parametrize("value", [None, {}, {"document_topics": None, "scenario_topics": []},
    {"document_topics": ["invented"], "scenario_topics": []},
    {"document_topics": [], "scenario_topics": ["study"]},
    {"document_topics": [{}], "scenario_topics": []}])
async def test_invalid_search_is_not_a_successful_empty_search(value):
    model = failing_model("invalid", value)
    gatherer, _ = build(model)
    with turn_metrics.turn() as metrics:
        with pytest.raises(ModelStageError) as error:
            await gatherer._select_relevant_topics(DOC_TOPICS, SCENARIO_TOPICS, "", "Question")
    assert error.value.stage == "topics" and error.value.reason == "invalid_output"
    failure(metrics, "topics", "invalid_output", "abort")
    model.call_model_structured.assert_awaited_once()


async def test_valid_empty_search_and_empty_catalogue_are_successful():
    model = failing_model("invalid", {"document_topics": [], "scenario_topics": []})
    gatherer, _ = build(model)
    with turn_metrics.turn() as metrics:
        assert await gatherer._select_relevant_topics(DOC_TOPICS, SCENARIO_TOPICS, "", "Question") == ([], [])
        assert await gatherer._select_relevant_topics([], [], "", "Question") == ([], [])
    assert not metrics.snapshot(status="respond", kind="message")["stages"]["topics"].get("failures")
    model.call_model_structured.assert_awaited_once()


@pytest.mark.parametrize("mode,reason", [("exception", "call_failed"), ("invalid", "invalid_output")])
async def test_failure_recovery_by_stage_is_content_free(mode, reason, caplog):
    model = failing_model(mode)
    gatherer, _ = build(model)
    service = ModelCollaborateService.__new__(ModelCollaborateService)
    service.ai_service = model
    with turn_metrics.turn() as metrics:
        assert (await ReadinessService(model).check(["Question"], []))["decision"] == "respond"
        with pytest.raises(ModelStageError) as topic_error:
            await gatherer._select_relevant_topics(DOC_TOPICS, SCENARIO_TOPICS, "", "Question")
        selected = await gatherer._select_relevant_example(CANDIDATES, [], "Question")
        assert selected.examples == [] and selected.failure_reason == reason
        grounded = await GroundingService(model).ground("Question", context())
        assert grounded["_failed"] and grounded["_failure_reason"] == reason
        with pytest.raises(ModelStageError) as writer_error:
            await service._generate_reply("system", "user")
        with pytest.raises(ModelStageError) as gate_error:
            await ResponseGate(model, SimpleNamespace())._verify_response("", "Question", "Reply", contract())
        checked_reply = "This is a previously checked reply, long enough to need a formatting call."
        assert await ResponseParser(model, min_chars_per_turn=20).parse(checked_reply) == [checked_reply]
    for stage, fallback in [("readiness", "respond"), ("topics", "abort"), ("example", "omit_example"),
                            ("grounding", "abort"), ("reply", "abort"), ("gate", "abort"), ("split", "single_message")]:
        failure(metrics, stage, reason, fallback)
    assert PRIVATE_MARKER not in caplog.text
    assert PRIVATE_MARKER not in str(metrics.snapshot(status="failed", kind="message"))
    assert all(PRIVATE_MARKER not in str(e.value) for e in (topic_error, writer_error, gate_error))
    assert model.call_model.await_count == 1 and model.call_model_structured.await_count == 6


@pytest.mark.parametrize("value", [[], {}, "-1", "999"])
async def test_invalid_example_selection_keeps_failure_distinct_from_none(value):
    gatherer, _ = build(failing_model("invalid", value))
    selected = await gatherer._select_relevant_example(CANDIDATES, [], "Question")
    assert selected.failure_reason == "invalid_output"
    gatherer.ai_service.call_model_structured.return_value = "none"
    selected = await gatherer._select_relevant_example(CANDIDATES, [], "Question")
    assert selected.examples == [] and selected.failure_reason is None


@pytest.mark.parametrize("coverage,facts,blocked", [("none", [], True), ("partial", ["f"], True), ("full", ["f"], False)])
async def test_failed_optional_example_cannot_become_a_false_factual_gap(coverage, facts, blocked):
    service = build_model_service()
    original = service.context_gatherer.gather
    async def gather(*args, **kwargs):
        return {**await original(*args, **kwargs), "example_failure": "call_failed"}
    service.context_gatherer.gather = gather
    service.grounding_service.ground.result.update(coverage=coverage, facts=facts)
    if blocked:
        with pytest.raises(ModelStageError, match="stage=example"):
            await service.model_orchestration("Question", "conv", "session", "guest")
        assert service._generate_reply.calls == service.response_gate.check.calls == 0
    else:
        outcome = await service.model_orchestration("Question", "conv", "session", "guest")
        assert outcome.reply_text == "a reply"
        assert service.response_gate.check.calls == 1


@pytest.mark.parametrize("mode", ["exception", "invalid"])
async def test_regeneration_failure_does_not_publish_original_or_unchecked_rewrite(mode):
    reply = "AI built the queue; I reviewed it."
    model = failing_model(mode, " ")
    model.call_model_structured = AsyncMock(return_value=review(check(reply),
        violations=[violation("punctuation", ";")], tag="<reject-minor>"))
    gate = ResponseGate(model, SimpleNamespace(append_message=AsyncMock()))
    with pytest.raises(ModelStageError, match="stage=regeneration"):
        await gate.check({**context(), "grounding": contract()}, "Question", reply, "system", "user", "conv", "session", 2)
    model.call_model.assert_awaited_once()
    model.call_model_structured.assert_awaited_once()


@pytest.mark.parametrize("key,value", [("category", {}), ("conflict_source", "unknown"), ("character_evidence", "unknown")])
async def test_malformed_review_violation_is_a_technical_error(key, value):
    bad = {**violation("punctuation", ";"), key: value}
    model = failing_model("invalid", review(violations=[bad]))
    with pytest.raises(ModelStageError, match="reason=invalid_output"):
        await ResponseGate(model, SimpleNamespace())._verify_response("", "Question", "Reply;", contract())


@pytest.mark.parametrize("decision", ["wait", "no_reply", "respond"])
async def test_early_search_failure_is_observed_and_unused_calls_are_cancelled(decision):
    cancelled = asyncio.Event()
    loop = asyncio.get_running_loop()
    errors = []
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, detail: errors.append(detail))
    async def model_call(**kwargs):
        if kwargs["stage"] == "topics":
            raise TimeoutError(PRIVATE_MARKER)
        if kwargs["stage"] == "readiness":
            await asyncio.sleep(0.01)
            return {"decision": decision, "reason": "test"}
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
    try:
        gatherer, _ = build(SimpleNamespace(call_model_structured=model_call))
        if decision == "respond":
            with pytest.raises(ModelStageError, match="stage=topics"):
                await gatherer.gather("Question", "conv")
        else:
            assert (await gatherer.gather("Question", "conv"))["readiness"]["decision"] == decision
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        import gc
        gc.collect()
        await asyncio.sleep(0)
        assert errors == []
    finally:
        loop.set_exception_handler(old_handler)
