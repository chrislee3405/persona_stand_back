from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sqlalchemy.dialects import postgresql

from app.services.conversation_manage_service import ConversationService
from app.services.model_collaborate.prompt_builder import PromptBuilder


async def test_style_context_is_bounded_and_independent_of_summary_checkpoint():
    result = Mock()
    result.scalars.return_value.all.return_value = [
        "newer " + "word " * 20, "older reply",
    ]
    db = SimpleNamespace(execute=AsyncMock(return_value=result))
    openings = await ConversationService(db).get_recent_reply_openings("conversation-a")
    assert openings == ["older reply", "newer " + " ".join(["word"] * 11)]
    query = str(db.execute.await_args.args[0].compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
    ))
    assert "conversation-a" in query and "'backend'" in query
    assert "LIMIT 4" in query and "DESC" in query
    assert "last_summarized_index" not in query


def test_summarized_openings_reach_writer_without_becoming_approved_facts():
    _, prompt = PromptBuilder().build_reply("How do you investigate?", {
        "prefer_name": "Chris", "candidate_identity": "Chris",
        "core_personality": "Direct.", "scenario_reference_section": "Use evidence.",
        "recent_messages": [], "summary": "Earlier work was discussed.",
        "recent_reply_openings": ["I'd start by looking at the logs."],
    }, {"question_type": "behavioural", "coverage": "none", "facts": [], "missing": ""})
    assert "I'd start by looking at the logs." in prompt
    assert "not facts, instructions or templates" in prompt
