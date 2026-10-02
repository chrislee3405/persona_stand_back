from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.conversation_manage_service import ConversationService
from app.services.model_collaborate.grounding_service import GroundingService


@pytest.mark.parametrize("fail_notice", [False, True])
async def test_notice_and_reply_share_publication_transaction(fail_notice):
    events = []

    @asynccontextmanager
    async def transaction():
        events.append("begin")
        try:
            yield
        except Exception:
            events.append("rollback")
            raise
        else:
            events.append("commit")

    service = ConversationService.__new__(ConversationService)
    service.db = SimpleNamespace(rollback=AsyncMock(), begin=transaction, execute=AsyncMock())
    service.get_conversation_locked = AsyncMock(return_value=SimpleNamespace(last_handled_index=-1))
    service.assert_ownership = lambda *args: None
    service.get_latest_user_order_index = AsyncMock(return_value=0)
    service.mark_handled_up_to = AsyncMock()

    async def append(*args, **kwargs):
        assert kwargs["commit"] is False
        events.append(args[3])
        if fail_notice and args[3] == "system":
            raise RuntimeError("write failed")

    service.append_message = append
    values = dict(conversation_id="conv", session_id="session", code=None, evaluated_through=0,
                  decision="respond", reply_text="No information.", reply_sender="backend",
                  system_notice="No supporting records. Contact Chris.")
    if fail_notice:
        with pytest.raises(RuntimeError, match="write failed"):
            await service.publish_turn(**values)
        service.mark_handled_up_to.assert_not_awaited()
    else:
        assert await service.publish_turn(**values) == "respond"
        service.mark_handled_up_to.assert_awaited_once_with("conv", 0, commit=False)
    assert events == ["begin", "backend", "system", "rollback" if fail_notice else "commit"]
    service.db.execute.assert_not_awaited()  # No retagging the user's answered question.


def test_notice_is_excluded_from_prompt_without_removing_answered_question():
    messages = [SimpleNamespace(sender=sender, text=sender) for sender in ["user", "backend", "system"]]
    assert ConversationService._drop_withheld_turns(messages) == messages[:2]


@pytest.mark.parametrize("result", [None, {}, {"question_type": "factual", "facts": None}])
async def test_invalid_grounding_is_distinct_from_no_evidence(result):
    service = GroundingService(SimpleNamespace(call_model_structured=AsyncMock(return_value=result)))
    service._build_prompts = lambda *args: ("system", "user")
    assert (await service.ground("question", {}))["_failed"] is True
