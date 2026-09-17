import pytest
from sqlalchemy import select

from app.constants import Sender
from app.models.conversation import Message
from app.services.conversation_manage_service import ConversationService
from tests.fakes.fake_gemini import REPLY

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("failure", [TimeoutError("simulated timeout"), ValueError("simulated malformed output")])
async def test_failed_turn_is_excluded_and_slots_released(consented_client, db, fake_gemini, rate_control, failure):
    fake_gemini.failure = failure
    response = await consented_client.post("/api/guestchat", json={"text": "Tell me about testing."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["sender"] == "system"
    assert payload["userMessageKept"] is False
    rows = list((await db.scalars(select(Message))).all())
    assert Sender.UNANSWERED_USER in [row.sender for row in rows]
    assert Sender.BACKEND not in [row.sender for row in rows]
    history = await ConversationService(db).get_recent_messages(payload["conversationId"])
    assert history == []
    await db.rollback()  # release the read transaction before a follow-up request
    assert rate_control._pending_counts == {}
    assert rate_control._ip_pending_counts == {}
    fake_gemini.failure = None
    retry = await consented_client.post("/api/guestchat", json={"text": "A follow-up question.", "conversationId": payload["conversationId"]})
    assert retry.json()["turns"] == [REPLY]
    assert retry.json()["userMessageKept"] is True


async def test_whole_turn_deadline_cancels_generation(consented_client, fake_gemini, monkeypatch, rate_control):
    monkeypatch.setattr("app.services.chat_service.TURN_DEADLINE_SECONDS", 0.01)
    fake_gemini.delay = 10
    response = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    assert response.json()["userMessageKept"] is False
    assert response.json()["sender"] == "system"
    assert rate_control._pending_counts == {}
    assert rate_control._ip_pending_counts == {}
