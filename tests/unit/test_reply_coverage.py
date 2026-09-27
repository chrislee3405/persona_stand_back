"""Prompt assembly regressions; these do not measure live model compliance."""
from app.services.model_collaborate.prompt_builder import PromptBuilder, _grounding_section


def test_complete_answer_omits_even_stale_missing_note():
    section = _grounding_section({
        "question_type": "factual", "coverage": "full",
        "facts": ["Integrated Vertex AI with FastAPI."],
        "missing": "The complete team breakdown",
    }, "Chris")
    assert "Integrated Vertex AI with FastAPI." in section
    assert "team breakdown" not in section
    assert "to hand" not in section


def test_partial_answer_keeps_specific_gap_and_supported_fact_without_referral():
    section = _grounding_section({
        "question_type": "factual", "coverage": "partial",
        "facts": ["Operated a forklift."], "missing": "Duration of forklift work",
    }, "Chris")
    assert "Operated a forklift." in section
    assert "Duration of forklift work" in section
    assert "name only the specific requested part" in section
    assert "point them to Chris" not in section
    assert "to hand" not in section


def test_empty_facts_still_decline_even_if_coverage_claims_full():
    section = _grounding_section({
        "question_type": "factual", "coverage": "full",
        "facts": [], "missing": "Duration of forklift work",
    }, "Chris")
    assert "No facts are available" in section
    assert "Chris directly" in section


def test_writer_resolves_blanket_referral_guidance_without_removing_personality():
    system, user = PromptBuilder().build_reply(
        "What machinery did you operate, and for how long?",
        {"prefer_name": "Chris", "candidate_identity": "Chris",
         "core_personality": "If a detail is missing, suggest contacting Chris directly.",
         "scenario_reference_section": "Answer plainly.",
         "recent_messages": [], "summary": None},
        {"question_type": "factual", "coverage": "partial",
         "facts": ["Operated a forklift."], "missing": "Duration of forklift work"},
    )
    assert "If a detail is missing, suggest contacting Chris directly." in system
    assert "even if personality or scenario guidance suggests" in system
    assert "Operated a forklift." in user
    assert "Duration of forklift work" in user
