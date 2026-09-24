import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.constants import Sender
from app.main import app
from app.models.conversation import Conversation, Message
from app.models.rate_limit import RateLimitCounter
from app.services.chat_service import MAX_MESSAGE_LENGTH
from tests.fakes.fake_gemini import REPLY
from tests.support.seed import TERMS

pytestmark = pytest.mark.integration


async def test_chat_reuses_conversation_and_persists_ordered_turns(consented_client, db, fake_gemini):
    first = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload == {"turns": [REPLY], "sender": "backend", "status": "respond", "conversationId": payload["conversationId"], "userMessageKept": True}
    second = await consented_client.post("/api/guestchat", json={"text": "What did you test?", "conversationId": payload["conversationId"]})
    assert second.status_code == 200
    assert second.json()["conversationId"] == payload["conversationId"]
    rows = list((await db.scalars(select(Message).order_by(Message.order_index))).all())
    assert [row.sender for row in rows] == [Sender.USER, Sender.BACKEND, Sender.USER, Sender.BACKEND]
    assert [row.order_index for row in rows] == [0, 1, 2, 3]
    assert rows[1].selected_document == "portfolio"
    assert fake_gemini.calls


async def test_wait_then_next_message_answers_pending_group(consented_client, db, fake_gemini):
    fragment = "Tell me about the project, specifically"
    completion = "how you tested it."
    fake_gemini.readiness_decision = "wait"

    first = await consented_client.post("/api/guestchat", json={"text": fragment})
    assert first.status_code == 200, first.text
    conversation_id = first.json()["conversationId"]
    assert first.json() == {
        "turns": [], "sender": Sender.SYSTEM, "status": "wait",
        "conversationId": conversation_id, "userMessageKept": True,
    }
    rows = list((await db.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.order_index)
    )).all())
    assert [(row.sender, row.text, row.order_index) for row in rows] == [(Sender.USER, fragment, 0)]
    conversation = await db.get(Conversation, conversation_id)
    assert conversation.last_handled_index == -1
    assert not any(kind == "text" for kind, _ in fake_gemini.calls)

    fake_gemini.calls.clear()
    fake_gemini.readiness_decision = "respond"
    second = await consented_client.post("/api/guestchat", json={
        "text": completion, "conversationId": conversation_id,
    })
    assert second.status_code == 200, second.text
    assert second.json() == {
        "turns": [REPLY], "sender": Sender.BACKEND, "status": "respond",
        "conversationId": conversation_id, "userMessageKept": True,
    }
    reply_prompts = [prompt for kind, prompt in fake_gemini.calls if kind == "text"]
    assert len(reply_prompts) == 1
    assert f"{fragment}\n{completion}" in reply_prompts[0]
    rows = list((await db.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.order_index)
    )).all())
    assert [(row.sender, row.text, row.order_index) for row in rows] == [
        (Sender.USER, fragment, 0),
        (Sender.USER, completion, 1),
        (Sender.BACKEND, REPLY, 2),
    ]
    await db.refresh(conversation)
    assert conversation.last_handled_index == 1


async def test_continue_after_wait_answers_the_held_message(consented_client, db, fake_gemini):
    fragment = "Two questions about the project, first"
    fake_gemini.readiness_decision = "wait"
    held = await consented_client.post("/api/guestchat", json={"text": fragment})
    assert held.json()["status"] == "wait"
    conversation_id = held.json()["conversationId"]

    # The visitor goes quiet. Even a gate that would still say "wait" is not
    # asked again: the silence is the answer.
    fake_gemini.calls.clear()
    response = await consented_client.post("/api/guestchat/continue", json={"conversationId": conversation_id})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "turns": [REPLY], "sender": Sender.BACKEND, "status": "respond",
        "conversationId": conversation_id, "userMessageKept": True,
    }
    assert not any("Unanswered message(s)" in prompt for _, prompt in fake_gemini.calls)
    reply_prompts = [prompt for kind, prompt in fake_gemini.calls if kind == "text"]
    assert len(reply_prompts) == 1 and fragment in reply_prompts[0]
    rows = list((await db.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.order_index)
    )).all())
    assert [(row.sender, row.text) for row in rows] == [(Sender.USER, fragment), (Sender.BACKEND, REPLY)]
    conversation = await db.get(Conversation, conversation_id)
    # The cursor covers the user rows the reply answered (order_index 0).
    assert conversation.last_handled_index == 0
    # One daily unit for the one message; the continue spent none.
    counters = list((await db.scalars(select(RateLimitCounter).where(RateLimitCounter.key.like("session:%")))).all())
    assert [row.count for row in counters] == [1]


async def test_continue_with_nothing_held_calls_no_model(consented_client, fake_gemini):
    first = (await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})).json()
    fake_gemini.calls.clear()

    response = await consented_client.post("/api/guestchat/continue", json={"conversationId": first["conversationId"]})

    assert response.status_code == 200
    assert response.json()["status"] == "no_reply"
    assert response.json()["turns"] == []
    assert fake_gemini.calls == []


async def test_continue_cannot_touch_another_sessions_conversation(consented_client, db, fake_gemini):
    fake_gemini.readiness_decision = "wait"
    held = (await consented_client.post("/api/guestchat", json={"text": "Two questions, first"})).json()
    fake_gemini.calls.clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as other:
        await other.post("/api/consent", json={"conditionText": TERMS["condition"]})
        foreign = await other.post("/api/guestchat/continue", json={"conversationId": held["conversationId"]})
        missing = await other.post("/api/guestchat/continue", json={"conversationId": "no-such-conversation"})

    # The same answer for both, so ids cannot be probed.
    assert (foreign.status_code, foreign.json()) == (missing.status_code, missing.json())
    assert foreign.status_code == 404
    assert fake_gemini.calls == []
    conversation = await db.get(Conversation, held["conversationId"])
    assert conversation.last_handled_index == -1


async def test_invite_continue_requires_a_verified_session(consented_client, fake_gemini):
    response = await consented_client.post("/api/invitechat/continue", json={"conversationId": "anything"})
    assert response.status_code == 401
    assert fake_gemini.calls == []


@pytest.mark.parametrize("payload", [{}, {"conversationId": ""}, {"conversationId": "x", "text": "resent"}, {"conversationId": "x" * 65}])
async def test_invalid_continue_payloads_never_call_model(consented_client, fake_gemini, payload):
    assert (await consented_client.post("/api/guestchat/continue", json=payload)).status_code == 422
    assert fake_gemini.calls == []


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
