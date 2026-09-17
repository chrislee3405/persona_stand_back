from datetime import date

import pytest
from sqlalchemy import select

from app.constants import MAX_DAILY_INVITE_CODE_FAILURES_PER_IP
from app.models.conversation import Message
from app.models.rate_limit import RateLimitCounter
from app.services.rate_control_service import (
    RateControlService,
    TooManyPendingMessagesError,
    _DAILY_QUOTA_PER_IP,
    _DAILY_QUOTA_PER_SESSION,
    _MAX_PENDING_PER_SESSION,
)
from tests.integration.test_invite_codes import session_data

pytestmark = pytest.mark.integration


async def test_session_daily_limit_survives_new_service_instance(consented_client, db, rate_control, fake_gemini):
    session_id = session_data(consented_client)["session_id"]
    db.add(RateLimitCounter(key=f"session:{session_id}", day=date.today(), count=_DAILY_QUOTA_PER_SESSION["guest"]))
    await db.commit()
    response = await consented_client.post("/api/guestchat", json={"text": "A project question."})
    assert response.status_code == 429
    assert "daily" in response.json()["detail"]
    assert rate_control._pending_counts == {}
    assert fake_gemini.calls == []
    replacement = RateControlService()
    assert await replacement._peek_daily(db, f"session:{session_id}") >= _DAILY_QUOTA_PER_SESSION["guest"]
    assert list((await db.scalars(select(Message))).all()) == []


async def test_ip_daily_rejection_releases_both_pending_slots(consented_client, db, rate_control, fake_gemini):
    db.add(RateLimitCounter(key="ip:127.0.0.1", day=date.today(), count=_DAILY_QUOTA_PER_IP))
    await db.commit()
    response = await consented_client.post("/api/guestchat", json={"text": "A project question."})
    assert response.status_code == 429
    assert rate_control._pending_counts == {}
    assert rate_control._ip_pending_counts == {}
    assert fake_gemini.calls == []


async def test_pending_cap_rejects_excess_and_recovers(db, rate_control):
    session_id = "fictional-session"
    for _ in range(_MAX_PENDING_PER_SESSION["guest"]):
        await rate_control.reserve_slot(db, session_id, "guest")
    with pytest.raises(TooManyPendingMessagesError):
        await rate_control.reserve_slot(db, session_id, "guest")
    rate_control.release_slot(session_id)
    await rate_control.reserve_slot(db, session_id, "guest")
    for _ in range(_MAX_PENDING_PER_SESSION["guest"]):
        rate_control.release_slot(session_id)
    assert rate_control._pending_counts == {}


async def test_invite_lockout_is_durable_and_refuses_even_valid_code(client, db):
    db.add(RateLimitCounter(key="code_fail_ip:127.0.0.1", day=date.today(), count=MAX_DAILY_INVITE_CODE_FAILURES_PER_IP))
    await db.commit()
    response = await client.post("/api/code", json={"inputCode": "TEST-INVITE"})
    assert response.status_code == 429
    assert (await client.get("/api/chatroom_initialize")).json()["verified"] is False
