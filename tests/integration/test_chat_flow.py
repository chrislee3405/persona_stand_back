import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.constants import Sender
from app.main import app
from app.models.conversation import Conversation, Message
from app.services.chat_service import MAX_MESSAGE_LENGTH
from tests.fakes.fake_gemini import REPLY
from tests.support.seed import TERMS

pytestmark = pytest.mark.integration


async def test_chat_reuses_conversation_and_persists_ordered_turns(consented_client, db, fake_gemini):
    first = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload == {"turns": [REPLY], "sender": "backend", "conversationId": payload["conversationId"], "userMessageKept": True}
    second = await consented_client.post("/api/guestchat", json={"text": "What did you test?", "conversationId": payload["conversationId"]})
    assert second.status_code == 200
    assert second.json()["conversationId"] == payload["conversationId"]
    rows = list((await db.scalars(select(Message).order_by(Message.order_index))).all())
    assert [row.sender for row in rows] == [Sender.USER, Sender.BACKEND, Sender.USER, Sender.BACKEND]
    assert [row.order_index for row in rows] == [0, 1, 2, 3]
    assert rows[1].selected_document == "portfolio"
    assert fake_gemini.calls


async def test_foreign_conversation_starts_separate_history(consented_client, db):
    first = (await consented_client.post("/api/guestchat", json={"text": "A project question."})).json()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as other:
        await other.post("/api/consent", json={"conditionText": TERMS["condition"]})
        response = await other.post("/api/guestchat", json={"text": "A different question.", "conversationId": first["conversationId"]})
    assert response.status_code == 200
    assert response.json()["conversationId"] != first["conversationId"]
    conversations = list((await db.scalars(select(Conversation))).all())
    assert len({row.owner_session_id for row in conversations}) == 2
    original = list((await db.scalars(select(Message).where(Message.conversation_id == first["conversationId"]))).all())
    assert len(original) == 2
    assert all(row.text != "A different question." for row in original)


async def test_privacy_gate_precedes_storage_and_model(consented_client, db, fake_gemini):
    response = await consented_client.post("/api/guestchat", json={"text": "Please email tester@example.com about the project."})
    assert response.status_code == 400
    assert list((await db.scalars(select(Message))).all()) == []
    assert fake_gemini.calls == []


async def test_message_length_boundary(consented_client, db, fake_gemini):
    too_long = await consented_client.post("/api/guestchat", json={"text": "x" * (MAX_MESSAGE_LENGTH + 1)})
    assert too_long.status_code == 413
    assert fake_gemini.calls == []
    assert list((await db.scalars(select(Message))).all()) == []
    boundary = await consented_client.post("/api/guestchat", json={"text": "x" * MAX_MESSAGE_LENGTH})
    assert boundary.status_code == 200
    assert boundary.json()["userMessageKept"] is True


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "hello", "code": "forged"}, {"text": "hello", "conversationId": "x" * 65}])
async def test_invalid_payloads_never_call_model(client, fake_gemini, payload):
    assert (await client.post("/api/guestchat", json=payload)).status_code == 422
    assert fake_gemini.calls == []
