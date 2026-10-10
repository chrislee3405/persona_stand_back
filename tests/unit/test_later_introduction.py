"""Prompt assembly regressions, not assertions of live model compliance."""
from types import SimpleNamespace

import pytest

from app.services.model_collaborate.grounding_service import GroundingService
from app.services.model_collaborate.prompt_builder import PromptBuilder
from app.services.model_collaborate.summarization_service import _SYSTEM_PROMPT


@pytest.mark.parametrize("question", [
    "Give me a 3 minutes self intro",
    "Give me your self intro",
    "Give me your self introduction",
])
@pytest.mark.parametrize("summarized", [False, True])
def test_later_introduction_keeps_reference_facts_despite_prior_refusal(question, summarized):
    background = "Completed a Master of Information Technology and previously worked as a teacher."
    refusal = "I don't have that detail to hand; contact Chris directly."
    history = [
        SimpleNamespace(sender="user", text="Tell me your background"),
        SimpleNamespace(sender="backend", text=background),
        SimpleNamespace(sender="user", text="What do you want to achieve in five years?"),
        SimpleNamespace(sender="backend", text=refusal),
    ]
    context = {
        "prefer_name": "Chris", "candidate_identity": "Chris",
        "core_personality": "Answer plainly.", "scenario_reference_section": "",
        "doc_reference_section": background, "similar_examples": [],
        "recent_messages": [] if summarized else history,
        # Deliberately lossy: fresh references must still reach grounding.
        "summary": "The candidate declined to answer." if summarized else None,
    }
    ground_system, ground_user = GroundingService(ai_service=object())._build_prompts(question, context)
    assert background in ground_user
    assert question in ground_user
    assert (context["summary"] if summarized else refusal) in ground_user
    assert "Prior refusals" in ground_system
    assert "not evidence that a fact is unavailable" in ground_system
    assert "does not require a stored script" in ground_system
    assert "format preference, not a missing candidate fact" in ground_system

    system, user = PromptBuilder().build_reply(question, context, {
        "question_type": "factual", "coverage": "full",
        "facts": [background], "missing": "",
    })
    assert background in user
    assert question in user
    assert "reuse relevant approved facts for requested introductions, recaps or explanations" in system
    assert "An earlier refusal does not justify refusing a now-supported request" in system
    assert "do not re-introduce yourself" not in system
    assert "supported shorter answer" in system
    assert "No facts are available" not in user


def test_summary_keeps_facts_and_limits_refusal_scope():
    assert "Retain concrete background facts alongside refusals" in _SYSTEM_PROMPT
    assert "specific question or request part" in _SYSTEM_PROMPT
    assert "not a general knowledge gap or restriction on future answers" in _SYSTEM_PROMPT
