"""Synthetic evidence regressions. No calls to a real provider or private seeds.

These test source/action handoffs and review enforcement, not model judgement.
"""
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.model_collaborate.evidence_contract import evidence_sources, validate_grounding, render_contract
from app.services.model_collaborate.grounding_service import GroundingService
from app.services.model_collaborate.prompt_builder import PromptBuilder
from app.services.model_collaborate.response_gate import ResponseGate, _has_rejection_evidence, is_fallback_response
from tests.unit.test_readiness_gate import build_model_service


SOURCE = "AI implemented HarborChat's queue. Alex specified the requirements and reviewed a simulation, not a full browser retest."
CLAIM = {"source_quote": SOURCE, "statement": SOURCE, "status": "documented", "answer_part": "Explain your contribution"}


def context():
    return {"doc_reference_section": "HARBOR OWNERSHIP:\n" + SOURCE,
            "doc_topic_list": ["HARBOR OWNERSHIP"], "similar_examples": [],
            "recent_messages": [], "summary": None, "candidate_identity": "Alex",
            "prefer_name": "Alex", "core_personality": "Direct.", "scenario_reference_section": ""}


def result(claim=None):
    return {"question_type": "factual", "coverage": "full", "missing": "",
            "claims": [copy.deepcopy(CLAIM if claim is None else claim)],
            "response_parts": [{"request": "Explain your contribution", "action": "answer_factual"}]}


def contract():
    return validate_grounding(result())


@pytest.mark.parametrize("statement,reply,verdict", [
    ("Alex has Python backend integration experience.", "I've implemented a Python backend integration.", "supported"),
    ("Alex has ten years of advanced Python experience.", "I have ten years of advanced Python experience.", "unsupported"),
])
async def test_interpretation_survives_handoff_and_ai_verdict_controls_review(statement, reply, verdict):
    source = "Alex implemented the FastAPI connection to an AI provider."
    claim = {"source_quote": source, "statement": statement,
             "status": "documented", "answer_part": "Python experience"}
    ctx = {**context(), "doc_reference_section": source, "doc_topic_list": []}
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=result(claim)))
    grounded = await GroundingService(model).ground("Python experience?", ctx)
    assert grounded["claims"][0] == claim
    _, writer_prompt = PromptBuilder().build_reply("Python experience?", ctx, grounded)
    assert source in writer_prompt and statement in writer_prompt
    model.call_model_structured.return_value = review(check(reply, verdict))
    tag, violations = await ResponseGate(model, SimpleNamespace())._verify_response(
        "", "Python experience?", reply, grounded)
    assert tag == ("<pass>" if verdict == "supported" else "<reject-major>")
    assert bool(violations) == (verdict != "supported")
    assert render_contract(grounded) in model.call_model_structured.await_args.kwargs["user_prompt"]
    assert model.call_model_structured.await_count == 2


def check(quote, verdict="supported"):
    return {"quote": quote, "verdict": verdict,
            "reason": "Compare ownership, uncertainty and causality with the cited source."}


def review(*checks, violations=None, tag="<pass>"):
    return {"result": tag, "violations": violations or [], "claim_checks": list(checks)}


def violation(category, quote, **extra):
    return {"category": category, "quote": quote, "reason": "Test review",
            "conflicting_quote": "", "conflict_source": "none", "character_evidence": "none", **extra}


def test_headings_and_interviewer_assertions_are_not_quotable_evidence():
    ctx = context()
    ctx["recent_messages"] = [SimpleNamespace(sender="user", text="You personally coded it all.")]
    ctx["summary"] = "The candidate claims to have coded it all."
    sources = evidence_sources(ctx)
    assert sources[0]["content"] == SOURCE
    assert sources[0]["label"] == "HARBOR OWNERSHIP"
    assert all("coded it all" not in source["content"] for source in sources)


@pytest.mark.parametrize("mutation", [
    {"source_quote": ""}, {"source_quote": None},
    {"statement": ""}, {"statement": None},
    {"status": "certainly_never"}, {"answer_part": ""},
])
async def test_invalid_claim_cannot_reach_the_writer(mutation):
    raw = result({**CLAIM, **mutation})
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=raw))
    grounded = await GroundingService(model).ground("Explain your contribution", context())
    assert grounded["_failed"] is True
    assert grounded["facts"] == []
    assert model.call_model_structured.await_count == 1


def test_multiple_reference_bodies_keep_separate_source_ids():
    ctx = context()
    ctx["doc_reference_section"] += "\n\nHARBOR LIMITS:\nOnly a simulation was reviewed."
    ctx["doc_topic_list"].append("HARBOR LIMITS")
    sources = evidence_sources(ctx)
    assert sources[0]["content"] == SOURCE
    assert sources[1] == {"id": "doc-2", "kind": "candidate", "label": "HARBOR LIMITS",
                          "content": "Only a simulation was reviewed."}


@pytest.mark.parametrize("mutation,code", [
    ({"status": "PRIVATE_STATUS"}, "claim_status"),
    ({"answer_part": None}, "claim_shape"),
])
async def test_grounding_logs_specific_validation_code_without_private_content(mutation, code, caplog):
    raw = result({**CLAIM, **mutation})
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=raw))
    grounded = await GroundingService(model).ground("Question", context())
    assert grounded["_failed"] is True
    assert f"Grounding validation failed: code={code}" in caplog.text
    assert "PRIVATE_" not in caplog.text
    assert model.call_model_structured.await_count == 1


def test_claim_schema_and_handoff_contain_quote_and_interpretation():
    from app.services.model_collaborate.evidence_contract import CLAIM_SCHEMA
    expected = {"source_quote", "statement", "status", "answer_part"}
    assert set(CLAIM_SCHEMA["properties"]) == set(CLAIM_SCHEMA["required"]) == expected
    raw = result({**CLAIM, "statement": "Alex reviewed the simulation.", "source_id": "unused"})
    grounded = validate_grounding(raw)
    assert set(grounded["claims"][0]) == expected
    assert grounded["facts"] == [SOURCE]


@pytest.mark.parametrize("parts", [None, [], [{"request": "x", "action": "solve_exercise"}], [{"request": "x", "action": {}}]])
def test_missing_or_unknown_actions_are_not_silently_treated_as_behavioural(parts):
    raw = result()
    raw["response_parts"] = parts
    with pytest.raises(ValueError, match="actions"):
        validate_grounding(raw)


async def test_grounding_writer_and_existing_review_share_exact_evidence_without_an_extra_call():
    ctx = context()
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=result()))
    grounded = await GroundingService(model).ground("Explain your contribution", ctx)
    assert model.call_model_structured.await_count == 1
    system, user = PromptBuilder().build_reply("Explain your contribution", ctx, grounded)
    assert render_contract(grounded) in user
    reply = "AI implemented HarborChat's queue; I set the requirements and reviewed a simulation."
    model.call_model_structured.return_value = review(check(reply))
    verdict, violations = await ResponseGate(model, SimpleNamespace())._verify_response("", "Contribution?", reply, grounded)
    assert verdict == "<pass>" and not violations
    assert render_contract(grounded) in model.call_model_structured.await_args.kwargs["user_prompt"]
    assert model.call_model_structured.await_count == 2  # grounding + existing review
    assert "decline_demonstration" in system


@pytest.mark.parametrize("quote", [
    SOURCE.replace(". Alex", ".\nAlex"),
    SOURCE.replace("implemented", "built"),
])
async def test_grounding_wording_differences_reach_review_without_original_sources(quote):
    model = SimpleNamespace(call_model_structured=AsyncMock(
        return_value=result({**CLAIM, "source_quote": quote, "statement": "Alex reviewed a simulation."})))
    grounded = await GroundingService(model).ground("Contribution?", context())
    assert not grounded.get("_failed")
    assert grounded["claims"][0]["source_quote"] == quote
    assert "sources" not in grounded
    assert model.call_model_structured.await_count == 1
    model.call_model_structured.return_value = review(check("I reviewed a simulation."))
    await ResponseGate(model, SimpleNamespace())._verify_response(
        "", "Contribution?", "I reviewed a simulation.", grounded)
    prompt = model.call_model_structured.await_args.kwargs["user_prompt"]
    assert render_contract(grounded) in prompt
    assert "Original selected sources" not in prompt
    assert SOURCE not in prompt


@pytest.mark.parametrize("reply", ["Our trainer uses an AI model.", "AI implemented the queue; I reviewed it."])
async def test_project_description_reported_as_character_break_does_not_spend_a_rewrite(reply):
    report = review(check(reply), violations=[violation("broke_character", reply, character_evidence="project_description")], tag="<reject-minor>")
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=report), call_model=AsyncMock())
    records = SimpleNamespace(append_message=AsyncMock())
    answer = await ResponseGate(model, records).check({**context(), "grounding": contract()}, "Projects?", reply,
                                                    "system", "user", "conv", "session", 2)
    assert answer == reply
    model.call_model.assert_not_awaited()
    records.append_message.assert_not_awaited()


def test_actual_self_identification_remains_blocking():
    reply = "As an AI, I cannot do that."
    assert _has_rejection_evidence(violation("broke_character", reply, character_evidence="self_identification"), reply, [])
    assert not _has_rejection_evidence(violation("broke_character", "model", character_evidence="project_description"), "Our project uses a model.", [])


async def test_minor_repair_retains_original_system_and_evidence_and_is_rechecked_for_actor_loss():
    original = "AI implemented HarborChat's queue; I reviewed it."
    damaged = "The queue was implemented and reviewed."
    model = SimpleNamespace(call_model=AsyncMock(return_value=damaged), call_model_structured=AsyncMock(side_effect=[
        review(check(original), violations=[violation("punctuation", ";")], tag="<reject-minor>"),
        review(check(damaged, "changed_meaning")),  # inconsistent pass cannot hide actor loss
    ]))
    records = SimpleNamespace(append_message=AsyncMock())
    grounded = contract()
    system, user = PromptBuilder().build_reply("Contribution?", context(), grounded)
    answer = await ResponseGate(model, records).check({**context(), "grounding": grounded}, "Contribution?", original,
                                                    system, user, "conv", "session", 2)
    assert is_fallback_response(answer)
    assert "meaning" in answer
    assert model.call_model.await_args.kwargs["system_prompt"].startswith(system)
    assert model.call_model.await_args.kwargs["user_prompt"] == user
    assert model.call_model_structured.await_count == 2
    assert any(damaged in c.kwargs["text"] for c in records.append_message.await_args_list)


@pytest.mark.parametrize("source,bad,status", [
    ("No record says Alex personally compared client IDs with serialization.", "I haven't personally compared those options.", "not_documented"),
    ("The first ref fix failed. A later queue introduced head-of-line blocking.", "The first fix failed because of head-of-line blocking.", "documented"),
])
async def test_uncertainty_and_event_relationship_survive_handoff_and_reported_distortion_is_blocked(source, bad, status):
    ctx = {**context(), "doc_reference_section": "HARBOR OWNERSHIP:\n" + source}
    raw = result({**CLAIM, "source_quote": source, "status": status})
    grounded = validate_grounding(raw)
    system, user = PromptBuilder().build_reply("Explain your contribution", ctx, grounded)
    assert source in user and status in user
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=review(check(bad, "changed_meaning"))))
    answer = await ResponseGate(model, SimpleNamespace(append_message=AsyncMock())).check(
        {**ctx, "grounding": grounded}, "Explain?", bad, system, user, "conv", "session", 1)
    assert is_fallback_response(answer)


async def test_invented_experience_inside_refusal_is_checked_even_with_no_facts():
    grounded = {"claims": [], "response_parts": [{"request": "new design", "action": "decline_demonstration"}], "coverage": "none", "missing": ""}
    bad = "I can explain how I've handled lease systems in past projects."
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=review(check(bad, "unsupported"))))
    verdict, violations = await ResponseGate(model, SimpleNamespace())._verify_response("", "Design a lease system", bad, grounded)
    assert verdict == "<reject-major>"
    assert violations[0]["category"] == "unsupported_fact"


@pytest.mark.parametrize("report", [None, {}, {"result": "<pass>", "violations": []},
    review({"quote": 42, "verdict": "supported", "reason": "x"}),
    review({"quote": "Reply", "verdict": "unknown", "reason": "x"}),
    review({"quote": "Reply", "verdict": "supported"}),
])
async def test_missing_or_invalid_factual_review_never_publishes_unchecked_reply(report):
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=report))
    with pytest.raises(RuntimeError, match="verify reply evidence"):
        await ResponseGate(model, SimpleNamespace())._verify_response("", "Question", "Reply", contract())


async def test_reviewer_outage_uses_technical_failure_path():
    model = SimpleNamespace(call_model_structured=AsyncMock(side_effect=TimeoutError("fake")))
    with pytest.raises(RuntimeError, match="verify reply evidence"):
        await ResponseGate(model, SimpleNamespace())._verify_response("", "Question", "Reply", contract())


async def test_pure_demonstration_declines_without_generated_personal_history_or_extra_calls():
    service = build_model_service()
    service.grounding_service.ground.result = {"question_type": "behavioural", "coverage": "none", "facts": [],
        "claims": [], "missing": "", "response_parts": [{"request": "Design a lease system", "action": "decline_demonstration"}]}
    outcome = await service.model_orchestration("Design a lease system", "conv", "session", "guest")
    assert "can't complete a new technical exercise" in outcome.reply_text
    assert "past projects" not in outcome.reply_text and outcome.system_notice is None
    assert service._generate_reply.calls == service.response_gate.check.calls == service.response_parser.parse.calls == 0


async def test_mixed_request_keeps_historical_answer_and_boundary_in_writer_and_gate():
    grounded = contract()
    grounded["response_parts"].append({"request": "Design a new queue", "action": "decline_demonstration"})
    service = build_model_service()
    service.grounding_service.ground.result = grounded
    service.prompt_builder = PromptBuilder()
    service.response_gate.check = AsyncMock(return_value="AI implemented the queue; I reviewed it. I can't do the new exercise here.")
    outcome = await service.model_orchestration("Explain your contribution and design a new queue", "conv", "session", "guest")
    gate_args = service.response_gate.check.await_args.args
    assert gate_args[0]["grounding"] == grounded
    assert render_contract(grounded) in gate_args[4]
    assert service._generate_reply.calls == 1 and outcome.system_notice is None


async def test_empty_mixed_request_does_not_turn_demonstration_into_factual_gap_notice():
    service = build_model_service()
    service.grounding_service.ground.result = {"question_type": "factual", "coverage": "none", "facts": [],
        "claims": [], "missing": "old fix", "response_parts": [
            {"request": "Explain old fix", "action": "answer_factual"},
            {"request": "Design a new fix", "action": "decline_demonstration"}]}
    outcome = await service.model_orchestration("Explain and design", "conv", "session", "guest")
    assert service._generate_reply.calls == 1
    assert outcome.system_notice is None


async def test_descriptive_answer_part_does_not_need_to_repeat_request():
    raw = result({**CLAIM, "answer_part": "introduction and background"})
    raw["response_parts"] = [{"request": "briefly introduce yourself and share a little bit about your background", "action": "answer_factual"}]
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=raw))
    grounded = await GroundingService(model).ground("Introduce yourself", context())
    assert not grounded.get("_failed")
    assert grounded["claims"][0]["answer_part"] == "introduction and background"
    assert grounded["facts"] == [SOURCE]


def test_repeated_request_labels_do_not_fail_grounding():
    raw = result()
    raw["response_parts"].append(dict(raw["response_parts"][0]))
    assert len(validate_grounding(raw)["response_parts"]) == 2


async def test_review_quote_wording_is_not_compared_by_python():
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value=review(
        check("A differently worded description", "supported"))))
    verdict, _ = await ResponseGate(model, SimpleNamespace())._verify_response(
        "", "Question", "The reply uses different words.", contract())
    assert verdict == "<pass>"
