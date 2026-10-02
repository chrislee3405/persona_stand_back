"""Prompt assembly regressions; these do not measure live model compliance."""
from app.services.model_collaborate.prompt_builder import PromptBuilder, _grounding_section
from app.services.model_collaborate.grounding_service import GroundingService


def test_complete_answer_omits_even_stale_missing_note():
    section = _grounding_section({
        "question_type": "factual", "coverage": "full",
        "facts": ["Integrated Vertex AI with FastAPI."],
        "missing": "The complete team breakdown",
    }, "Chris")
    assert "Integrated Vertex AI with FastAPI." in section
    assert "team breakdown" not in section
    assert "to hand" not in section


def test_boundary_wording_can_adapt_to_history_without_relaxing_the_boundary():
    from types import SimpleNamespace
    previous = "I'd rather discuss the code in a live technical session."
    boundary = "Do not write code here. Explain the format and suggest a live session."
    system, user = PromptBuilder().build_reply("Can you solve this other coding exercise?", {
        "prefer_name": "Chris", "candidate_identity": "Chris", "core_personality": "Direct.",
        "scenario_reference_section": boundary,
        "recent_messages": [SimpleNamespace(sender="backend", text=previous)], "summary": None,
    }, {"question_type": "behavioural", "coverage": "none", "facts": [], "missing": ""})
    assert previous in user and boundary in user
    assert "avoid recycling an opening or explanation" in system
    assert "a short reminder is enough" in system
    assert "overrides scenario style instructions, never its substantive limits" in system
    assert "prohibited partial solution" in system


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
    assert "Say only: I don't have that information for now." in section
    assert "Chris directly" not in section


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


def test_current_intention_can_answer_broad_goal_without_adding_deadline():
    context = {
        "prefer_name": "Chris", "candidate_identity": "Chris",
        "core_personality": "Answer plainly.", "scenario_reference_section": "",
        "recent_messages": [], "summary": None, "similar_examples": [],
        "doc_reference_section": "Looking for a graduate or junior developer role.",
    }
    question = "Whats your target this year"
    ground_system, ground_user = GroundingService(gemini_service=object())._build_prompts(question, context)
    assert context["doc_reference_section"] in ground_user
    assert question in ground_user
    assert "current intention can supply full coverage" in ground_system
    assert "not a promise to obtain it by year-end" in ground_system
    assert "Do not infer ambitions from qualifications, activities or job requirements alone" in ground_system
    system, user = PromptBuilder().build_reply(question, context, {
        "question_type": "factual", "coverage": "full",
        "facts": [context["doc_reference_section"]], "missing": "A separately written annual plan",
    })
    assert context["doc_reference_section"] in user
    assert "A separately written annual plan" not in user
    assert "No facts are available" not in user
    assert "without inventing a deadline or promising an outcome" in system


def test_current_intention_does_not_fill_an_explicit_deadline_gap():
    section = _grounding_section({
        "question_type": "factual", "coverage": "partial",
        "facts": ["Seeking a junior developer role."],
        "missing": "The exact date by which the candidate plans to secure a role",
    }, "Chris")
    assert "Seeking a junior developer role." in section
    assert "exact date" in section
    assert "No facts are available" not in section
