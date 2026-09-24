"""Real Postgres regressions for the September audit; all data is fictional."""
import asyncio
import logging

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.constants import Sender
from app.database import SessionLocal, engine
from app.main import app
from app.models.conversation import Conversation, Message
from app.services.conversation_manage_service import ConversationService
from app.services.session_work import session_work

pytestmark = pytest.mark.integration


async def pause_reply(monkeypatch, fake_gemini):
    entered, resume = asyncio.Event(), asyncio.Event()
    original = fake_gemini.call_model

    async def paused(*args, **kwargs):
        entered.set()
        await resume.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(fake_gemini, 'call_model', paused)
    return entered, resume


async def test_rejection_cannot_retag_an_active_question(consented_client, fake_gemini, monkeypatch, db):
    first = (await consented_client.post('/api/guestchat', json={'text': 'Describe the portfolio.'})).json()
    cid = first['conversationId']
    entered, resume = await pause_reply(monkeypatch, fake_gemini)
    active = asyncio.create_task(consented_client.post('/api/guestchat', json={
        'conversationId': cid, 'text': 'Describe its tests.'}))
    await asyncio.wait_for(entered.wait(), 5)
    rejected = asyncio.create_task(consented_client.post('/api/guestchat', json={
        'conversationId': cid, 'text': 'a' * 751}))
    try:
        await asyncio.sleep(0.1)
        assert not rejected.done()
    finally:
        resume.set()
    good, bad = await asyncio.wait_for(asyncio.gather(active, rejected), 10)
    assert good.status_code == 200 and good.json()['userMessageKept'] is True
    assert bad.status_code == 413
    row = await db.scalar(select(Message).where(Message.conversation_id == cid, Message.text == 'Describe its tests.'))
    assert row.sender == Sender.USER


@pytest.mark.parametrize('model_fails', [False, True])
async def test_invite_rotation_waits_for_active_turn(consented_client, fake_gemini, monkeypatch, model_fails):
    cid = (await consented_client.post('/api/guestchat', json={'text': 'Describe the portfolio.'})).json()['conversationId']
    if model_fails:
        fake_gemini.failure = RuntimeError('fictional model failure')
    entered, resume = await pause_reply(monkeypatch, fake_gemini)
    old_cookie = dict(consented_client.cookies)
    active = asyncio.create_task(consented_client.post('/api/guestchat', json={
        'conversationId': cid, 'text': 'Describe the architecture.'}))
    await asyncio.wait_for(entered.wait(), 5)
    upgrade = asyncio.create_task(consented_client.post('/api/code', json={
        'conversationId': cid, 'inputCode': 'TEST-INVITE'}))
    try:
        await asyncio.sleep(0.1)
        assert not upgrade.done()
    finally:
        resume.set()
    reply, verified = await asyncio.wait_for(asyncio.gather(active, upgrade), 10)
    assert reply.status_code == 200
    assert verified.status_code == 200
    initialized = (await consented_client.get('/api/chatroom_initialize')).json()
    assert initialized['verified'] and initialized['consent']['consented']
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver', cookies=old_cookie) as old:
        assert (await old.post('/api/guestchat', json={'text': 'Another question.', 'conversationId': cid})).status_code == 403


async def test_cursor_is_monotonic_with_a_retained_stale_orm_object():
    async with SessionLocal() as creator:
        _, cid = await ConversationService(creator).append_message(None, None, 'owner', Sender.USER, 'fictional')
    async with SessionLocal() as older, SessionLocal() as newer:
        retained = await ConversationService(older).get_conversation_unlocked(cid)
        await ConversationService(newer).mark_handled_up_to(cid, 2)
        await ConversationService(older).mark_handled_up_to(cid, 0)
        assert retained is not None
    async with SessionLocal() as check:
        assert (await check.get(Conversation, cid)).last_handled_index == 2


@pytest.mark.parametrize('fallback', [False, True])
async def test_publication_rolls_back_every_write_and_can_be_retried(monkeypatch, fallback):
    async with SessionLocal() as db:
        service = ConversationService(db)
        _, cid = await service.append_message(None, None, 'owner', Sender.USER, 'first part')
        await service.append_message(cid, None, 'owner', Sender.USER, 'second part')
        values = dict(conversation_id=cid, session_id='owner', code=None, evaluated_through=1,
                      decision='respond', reply_text='fictional reply',
                      reply_sender=Sender.SYSTEM if fallback else Sender.BACKEND)
        original = service.mark_handled_up_to

        async def fail_after_writes(*args, **kwargs):
            raise RuntimeError('fictional cursor failure')

        monkeypatch.setattr(service, 'mark_handled_up_to', fail_after_writes)
        with pytest.raises(RuntimeError):
            await service.publish_turn(**values)
        rows = list((await db.scalars(select(Message).where(Message.conversation_id == cid))).all())
        assert [m.sender for m in rows] == [Sender.USER, Sender.USER]
        assert (await db.get(Conversation, cid)).last_handled_index == -1
        monkeypatch.setattr(service, 'mark_handled_up_to', original)
        assert await service.publish_turn(**values) == 'respond'
        # Same evaluated group cannot publish a duplicate, including after restart.
    async with SessionLocal() as db:
        assert await ConversationService(db).publish_turn(**values) == 'superseded'
        rows = list((await db.scalars(select(Message).where(Message.conversation_id == cid).order_by(Message.order_index))).all())
        assert len(rows) == 3
        assert [m.sender for m in rows[:2]] == ([Sender.UNANSWERED_USER] * 2 if fallback else [Sender.USER] * 2)
        assert (await db.get(Conversation, cid)).last_handled_index == 1


async def test_cursor_migration_is_required_for_startup_and_readiness(client):
    assert (await client.get('/api/health/ready')).status_code == 200
    async with engine.begin() as connection:
        await connection.execute(text('ALTER TABLE conversation DROP COLUMN last_handled_index'))
    try:
        with pytest.raises(RuntimeError, match='cursor migration required'):
            async with app.router.lifespan_context(app):
                pytest.fail('An incompatible schema must not start')
        assert (await client.get('/api/health/ready')).status_code == 503
    finally:
        # Data-reset fixtures truncate tables; they do not repair schema changes.
        async with engine.begin() as connection:
            await connection.execute(text(
                'ALTER TABLE conversation ADD COLUMN last_handled_index INTEGER NOT NULL DEFAULT -1'))
    assert (await client.get('/api/health/ready')).status_code == 200


async def test_exception_handler_never_logs_or_copies_sql_parameters(consented_client, fake_gemini, caplog, db):
    canary = 'fictional-private-sql-canary'
    fake_gemini.failure = IntegrityError('INSERT fictional', (canary,), Exception(canary))
    with caplog.at_level(logging.ERROR):
        response = await consented_client.post('/api/guestchat', json={'text': 'Describe the project.'})
    assert response.status_code == 200
    assert canary not in caplog.text
    assert 'IntegrityError' in caplog.text
    review = list((await db.scalars(select(Message).where(Message.sender == Sender.ERROR))).all())
    assert review and all(canary not in row.text for row in review)


async def test_cancelled_lock_owner_releases_database_coordination():
    entered = asyncio.Event()
    async def hold():
        async with session_work('fictional-lock-owner'):
            entered.set()
            await asyncio.Event().wait()
    owner = asyncio.create_task(hold())
    await asyncio.wait_for(entered.wait(), 5)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    async with asyncio.timeout(5):
        async with session_work('fictional-lock-owner'):
            pass
