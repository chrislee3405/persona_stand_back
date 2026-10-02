from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.model_collaborate.response_gate import ResponseGate, _has_rejection_evidence, is_fallback_response


def violation(category="contradiction", quote="I do not have that date.", other="", source="none"):
    return {"category": category, "reason": "test review", "quote": quote,
            "conflicting_quote": other, "conflict_source": source}


@pytest.mark.parametrize("source,other,history", [
    ("none", "", []),
    ("history", "I confirmed that date.", [SimpleNamespace(sender="user", text="I confirmed that date.")]),
    ("history", "I confirmed that date.", [SimpleNamespace(sender="backend", text="I taught before moving into software.")]),
    ("response", "I do not have that date.", []),
    ("response", "not have that", []),
])
def test_unsubstantiated_or_same_claim_cannot_block(source, other, history):
    assert not _has_rejection_evidence(violation(other=other, source=source), "I do not have that date.", history)


def test_explicit_internal_contradiction_is_still_blocking():
    reply = "I do not have that date. I graduated on 12 June 2020."
    assert _has_rejection_evidence(violation(other="I graduated on 12 June 2020.", source="response"), reply, [])


def test_explicit_conflict_with_persona_history_is_still_blocking():
    assert _has_rejection_evidence(violation(quote="I never worked as a teacher.",
        other="I worked as a teacher.", source="history"), "I never worked as a teacher.",
        [SimpleNamespace(sender="backend", text="I worked as a teacher.")])


@pytest.mark.parametrize("category", ["asked_for_private_info", "made_a_promise", "broke_character", "punctuation"])
def test_other_rules_still_need_an_exact_quote(category):
    assert _has_rejection_evidence(violation(category, "actual phrase"), "An actual phrase.", [])
    assert not _has_rejection_evidence(violation(category, "invented phrase"), "An actual phrase.", [])


@pytest.mark.parametrize("category,reply", [
    ("contradiction", "I want to develop responsible AI products. I don't have information about a specific cloud learning plan."),
    ("unnatural", "Over the next five years, I am aiming to grow into an experienced software developer."),
])
async def test_reported_false_positive_passes_without_spending_a_retry(category, reply):
    model = SimpleNamespace(call_model=AsyncMock(), call_model_structured=AsyncMock())
    records = SimpleNamespace(append_message=AsyncMock())
    gate = ResponseGate(model, records)
    gate._verify_response = AsyncMock(return_value=("<reject-minor>", [violation(category, reply)]))
    result = await gate.check({"recent_messages": []}, "Where would you focus?", reply,
                              "system", "user", "conv", "session", regen_counter=2)
    assert result == reply
    gate._verify_response.assert_awaited_once()
    model.call_model.assert_not_awaited()
    records.append_message.assert_not_awaited()


async def test_style_concern_does_not_hide_a_real_conflict():
    reply = "I never worked as a teacher."
    records = SimpleNamespace(append_message=AsyncMock())
    gate = ResponseGate(SimpleNamespace(), records)
    gate._verify_response = AsyncMock(return_value=("<reject-minor>", [
        violation("unnatural", reply), violation(quote=reply, other="I worked as a teacher.", source="history"),
    ]))
    result = await gate.check({"recent_messages": [SimpleNamespace(sender="backend", text="I worked as a teacher.")]},
                              "Background?", reply, "system", "user", "conv", "session", regen_counter=1)
    assert is_fallback_response(result)
    assert "contradicting" in result and "unnatural" not in result


async def test_auditor_requests_two_claims_and_excludes_mild_formality():
    model = SimpleNamespace(call_model_structured=AsyncMock(return_value={"result": "<pass>", "violations": []}))
    await ResponseGate(model, SimpleNamespace())._verify_response("", "Question", "Reply")
    arguments = model.call_model_structured.await_args.kwargs
    assert "same subject, scope and time" in arguments["system_prompt"]
    assert "implied expectation" in arguments["system_prompt"]
    assert "Ordinary interview formality" in arguments["system_prompt"]
    assert "conflicting_quote" in arguments["schema"]["properties"]["violations"]["items"]["required"]
