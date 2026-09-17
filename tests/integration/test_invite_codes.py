import base64
import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from app.main import app
from app.models.code import InviteCode
from app.models.conversation import Conversation
from tests.support.seed import INVITE_CODE, TERMS

pytestmark = pytest.mark.integration


def session_data(client):
    return json.loads(base64.b64decode(client.cookies.get("session").split(".")[0]))


async def test_invite_requires_verification_and_rejects_invalid_code(client, fake_gemini):
    assert (await client.post("/api/invitechat", json={"text": "Hello"})).status_code == 401
    assert (await client.post("/api/code", json={"inputCode": "wrong"})).status_code == 401
    assert (await client.get("/api/chatroom_initialize")).json()["verified"] is False
    assert fake_gemini.calls == []


async def test_verification_rotates_session_and_transfers_owned_history(consented_client, db):
    client = consented_client
    chat = (await client.post("/api/guestchat", json={"text": "A project question."})).json()
    previous = session_data(client)
    old_cookie = client.cookies.get("session")
    response = await client.post("/api/code", json={"inputCode": INVITE_CODE, "conversationId": chat["conversationId"]})
    assert response.json() == {"status": "success"}
    current = session_data(client)
    assert current["session_id"] != previous["session_id"]
    assert "invite_code_id" in current
    assert INVITE_CODE not in json.dumps(current)
    state = (await client.get("/api/chatroom_initialize")).json()
    assert state["verified"] is True
    assert state["consent"]["consented"] is True
    reply = await client.post("/api/invitechat", json={"text": "Tell me more.", "conversationId": chat["conversationId"]})
    assert reply.json()["conversationId"] == chat["conversationId"]
    conversation = (await db.scalars(select(Conversation))).one()
    assert conversation.owner_session_id == current["session_id"]
    assert conversation.code == INVITE_CODE
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver", cookies={"session": old_cookie}) as replay:
        assert (await replay.post("/api/guestchat", json={"text": "Hello"})).status_code == 403


async def test_valid_code_cannot_upgrade_someone_elses_conversation(consented_client, db):
    chat = (await consented_client.post("/api/guestchat", json={"text": "A project question."})).json()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as other:
        await other.post("/api/consent", json={"conditionText": TERMS["condition"]})
        response = await other.post("/api/code", json={"inputCode": INVITE_CODE, "conversationId": chat["conversationId"]})
        assert response.status_code == 403
        assert (await other.get("/api/chatroom_initialize")).json()["verified"] is False
    assert (await db.scalars(select(Conversation))).one().code == "GUEST"


async def test_revoked_invite_is_refused_on_next_message(consented_client, db, fake_gemini):
    assert (await consented_client.post("/api/code", json={"inputCode": INVITE_CODE})).status_code == 200
    await db.execute(delete(InviteCode))
    await db.commit()
    assert (await consented_client.post("/api/invitechat", json={"text": "Hello"})).status_code == 401
    assert (await consented_client.get("/api/chatroom_initialize")).json()["verified"] is False
    assert fake_gemini.calls == []
