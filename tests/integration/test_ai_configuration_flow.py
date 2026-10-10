from sqlalchemy import select, update

from app.database import engine, SessionLocal
from app.models.model_configuration import ModelConfiguration


async def test_activation_validates_before_updating_and_clears_old_overrides(db, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import pytest
    from scripts import configure_ai
    from app.services.ai.configuration import AIConfiguration
    from app.services.ai.provider_factory import AIService
    provider = SimpleNamespace(call_model=AsyncMock(return_value="OK"),
                               call_model_structured=AsyncMock(return_value="wrong"))
    monkeypatch.setattr(configure_ai, "get_ai_service", lambda _: AIService({"openai": provider}))
    configuration = AIConfiguration("openai", "checked-model")
    with pytest.raises(ValueError, match="probe"):
        await configure_ai.activate(configuration)
    assert await db.scalar(select(ModelConfiguration.provider)) == "gemini"
    await db.rollback()
    provider.call_model_structured.return_value = "OK"
    await configure_ai.activate(configuration, check_only=True)
    assert await db.scalar(select(ModelConfiguration.provider)) == "gemini"
    await db.rollback()
    await configure_ai.activate(configuration)
    row = (await db.execute(select(ModelConfiguration).where(ModelConfiguration.provider == "openai"))).scalar_one()
    assert row.provider == "openai" and row.default_model == "checked-model"
    assert row.summary_model is None
    enabled = (await db.scalars(select(ModelConfiguration).where(ModelConfiguration.is_active.is_(True)))).all()
    assert len(enabled) == 2
    assert len({row.use_priority for row in enabled}) == 2


async def test_context_connections_released_and_changes_apply_next_turn(consented_client, fake_gemini, monkeypatch):
    calls = []
    for method in ("call_model", "call_model_structured"):
        original = getattr(fake_gemini, method)
        async def checked(*args, _original=original, **kwargs):
            # Coordination locks have their own pool. No application query
            # connection should be checked out at any of these model calls.
            assert engine.pool.checkedout() == 0
            calls.append(kwargs["model_name"])
            return await _original(*args, **kwargs)
        monkeypatch.setattr(fake_gemini, method, checked)
    response = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    assert response.json()["sender"] == "backend", response.text
    assert set(calls) == {"test-model"}
    calls.clear()
    async with SessionLocal() as session:
        await session.execute(update(ModelConfiguration).values(default_model="new-default", response_model="writer"))
        await session.commit()
    response = await consented_client.post("/api/guestchat", json={"text": "Explain the project."})
    assert response.json()["sender"] == "backend", response.text
    assert set(calls) == {"new-default", "writer"}


async def test_configuration_edit_during_turn_does_not_mix_models(consented_client, fake_gemini, monkeypatch):
    original = fake_gemini.call_model_structured
    seen = []
    changed = False
    async def change_during_call(**kwargs):
        nonlocal changed
        seen.append(kwargs["model_name"])
        if not changed:
            changed = True
            async with SessionLocal() as session:
                await session.execute(update(ModelConfiguration).values(default_model="next-turn"))
                await session.commit()
        return await original(**kwargs)
    monkeypatch.setattr(fake_gemini, "call_model_structured", change_during_call)
    response = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    assert response.json()["sender"] == "backend", response.text
    assert set(seen) == {"test-model"}
    async with SessionLocal() as session:
        assert await session.scalar(select(ModelConfiguration.default_model)) == "next-turn"


async def test_database_provider_selects_other_adapter(consented_client, fake_gemini, monkeypatch):
    from app.services.ai import provider_factory
    calls = []
    original = fake_gemini.call_model
    async def capture(**kwargs):
        calls.append(kwargs["model_name"])
        return await original(**kwargs)
    monkeypatch.setattr(fake_gemini, "call_model", capture)
    monkeypatch.setattr(provider_factory, "GrokService", lambda: fake_gemini)
    async with SessionLocal() as session:
        await session.execute(update(ModelConfiguration).values(provider="grok", default_model="grok-test"))
        await session.commit()
    response = await consented_client.post("/api/guestchat", json={"text": "Tell me about the project."})
    assert response.json()["sender"] == "backend", response.text
    assert calls and set(calls) == {"grok-test"}


async def test_seed_adds_presets_without_overwriting_existing_selection(db):
    from app.models.seed.load import _load_model_configuration
    from app.services.ai.configuration import load_configuration
    assert await _load_model_configuration(db, False) == 1
    await db.commit()
    rows = list((await db.scalars(select(ModelConfiguration))).all())
    assert {r.provider for r in rows} == {"gemini", "openai"}
    assert (await load_configuration(db)).default_model == "test-model"
    await db.execute(update(ModelConfiguration).values(is_active=False))
    await db.execute(update(ModelConfiguration).where(ModelConfiguration.provider == "openai").values(is_active=True, default_model="owner-openai"))
    await db.commit()
    assert await _load_model_configuration(db, False) == 0
    await db.commit()
    assert (await load_configuration(db)).default_model == "owner-openai"
    assert (await load_configuration(db)).provider == "openai"


async def test_fresh_seed_enables_both_providers_with_stage_selection(db):
    from sqlalchemy import delete
    from app.models.seed.load import _load_model_configuration
    from app.services.ai.configuration import load_configuration
    await db.execute(delete(ModelConfiguration))
    assert await _load_model_configuration(db, False) == 2
    await db.commit()
    rows = {row.provider: row for row in (await db.scalars(select(ModelConfiguration))).all()}
    assert set(rows) == {"gemini", "openai"}
    assert all(row.is_active for row in rows.values())
    assert rows["gemini"].use_priority == 1
    assert rows["openai"].use_priority == 2
    gemini = rows["gemini"]
    assert gemini.default_model == "gemini-3.5-flash-lite"
    assert gemini.summary_model == "gemini-2.5-flash"
    selection = await load_configuration(db)
    assert selection.provider == "gemini"
    for stage in ("readiness", "topics", "grounding", "reply", "regeneration", "gate", "split"):
        assert selection.selection_for(stage) == ("gemini", "gemini-3.5-flash-lite")
    assert selection.selection_for("summary") == ("gemini", "gemini-2.5-flash")
    assert selection.selection_for("example") == ("openai", "gpt-4.1-mini")
