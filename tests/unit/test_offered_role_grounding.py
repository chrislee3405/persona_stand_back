"""Checks context delivery and prompt assembly, not live model compliance."""
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from app.services.model_collaborate.grounding_service import GroundingService
from app.services.model_collaborate.prompt_builder import PromptBuilder


def test_role_learning_question_has_explicit_hypothetical_boundary_and_writer_guidance():
    question = "What would you prioritise learning to become effective in this position?"
    context = dict(job_context="Junior cloud and platform engineer. Support reliable application delivery.",
                   doc_reference_section="", similar_examples=[], recent_messages=[], summary=None,
                   prefer_name="Chris", candidate_identity="Chris", core_personality="Direct.",
                   scenario_reference_section="")
    system, user = GroundingService(object())._build_prompts(question, context)
    assert question in user and question in system
    assert "classify it as behavioural even if no such plan is stored" in system
    assert "What are you currently learning?" in system
    assert "What learning plan have you committed to?" in system
    assert "also requires factual evidence" in system
    assert "Questions about the offered position are factual too" not in system
    _, writer = PromptBuilder().build_reply(question, context, {
        "question_type": "behavioural", "coverage": "none", "facts": [], "missing": "",
    })
    assert "propose relevant learning or work priorities" in writer
    assert "never as an existing plan" in writer
    assert "No facts are available" not in writer


@pytest.mark.parametrize("role", ["Junior applied AI developer", "Junior cloud engineer"])
async def test_role_question_can_reach_writer_without_candidate_facts(role):
    client = SimpleNamespace(call_model_structured=AsyncMock(return_value={
        "question_type": "factual", "coverage": "full",
        "facts": [f"Offered role: {role}."], "missing": "",
    }))
    context = dict(job_context=f"Role: {role}. Skills: Kubernetes.",
                   doc_reference_section="", similar_examples=[], recent_messages=[], summary=None,
                   prefer_name="Chris", candidate_identity="Chris", core_personality="Direct.",
                   scenario_reference_section="")
    question = "Do you know what position we offer?"
    grounding = await GroundingService(client).ground(question, context)
    call = client.call_model_structured.await_args.kwargs
    assert context["job_context"] in call["user_prompt"]
    assert "Role requirements are never evidence of candidate experience" in call["system_prompt"]
    system, user = PromptBuilder().build_reply(question, context, grounding)
    assert f"Offered role: {role}." in user
    assert "No facts are available" not in user
    assert "not your background" in system


@pytest.mark.parametrize("role", [None, "", "   "])
def test_absent_role_does_not_add_role_context_or_role_instructions(role):
    system, user = GroundingService(object())._build_prompts("What position?", dict(
        job_context=role, doc_reference_section="", similar_examples=[], recent_messages=[], summary=None))
    assert "Role context (quoted data" not in user
    assert "An offered-role brief is supplied" not in system
