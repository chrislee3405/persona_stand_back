import pytest
from sqlalchemy import delete, select

from app.models.consent import ConsentPolicy, ConsentRecord
from app.models.conversation import Message
from tests.support.seed import TERMS

pytestmark = pytest.mark.integration


async def test_no_consent_blocks_storage_and_model(client, db, fake_gemini):
    response = await client.post("/api/guestchat", json={"text": "Tell me about your project."})
    assert response.status_code == 403
    assert list((await db.scalars(select(Message))).all()) == []
    assert fake_gemini.calls == []


async def test_consent_must_match_current_terms(client, db):
    response = await client.post("/api/consent", json={"conditionText": "Old terms"})
    assert response.status_code == 400
    assert list((await db.scalars(select(ConsentRecord))).all()) == []


async def test_consent_withdrawal_and_reagreement_keep_audit_history(consented_client, db):
    client = consented_client
    for _ in range(2):
        assert (await client.post("/api/consent/withdraw")).json() == {"consented": False}
    assert (await client.post("/api/guestchat", json={"text": "Hello"})).status_code == 403
    response = await client.post("/api/consent", json={"conditionText": TERMS["condition"]})
    assert response.status_code == 200
    records = list((await db.scalars(select(ConsentRecord).order_by(ConsentRecord.id))).all())
    assert len(records) == 2
    assert records[0].withdrawn_at is not None
    assert records[1].withdrawn_at is None
    assert records[1].condition_text == TERMS


async def test_new_policy_requires_new_agreement(consented_client, db, fake_gemini):
    db.add(ConsentPolicy(version="test-v2", condition_text={"condition": "New fictional terms"}))
    await db.commit()
    state = (await consented_client.get("/api/chatroom_initialize")).json()
    assert state["consent"]["consented"] is False
    assert state["consent"]["policyVersion"] == "test-v2"
    assert (await consented_client.post("/api/guestchat", json={"text": "Hello"})).status_code == 403
    assert fake_gemini.calls == []


async def test_missing_policy_fails_closed(client, db):
    await db.execute(delete(ConsentPolicy))
    await db.commit()
    state = (await client.get("/api/chatroom_initialize")).json()
    assert state["consent"] == {"consented": False, "policyVersion": None, "conditionTerms": None}
    assert (await client.post("/api/consent", json={"conditionText": "anything"})).status_code == 503
