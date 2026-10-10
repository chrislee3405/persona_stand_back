from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.ai.configuration import AIConfiguration, AIConfigurationError, load_configuration, selection_from_rows
from app.models.model_configuration import MODEL_COLUMNS
from app.services.ai.provider_factory import AIService
from app.services.read_phase import finish_read_phase


async def test_stage_models_and_retries_preserve_explicit_selection():
    provider = SimpleNamespace(call_model=AsyncMock(return_value="answer"),
                               call_model_structured=AsyncMock(return_value={}))
    service = AIService({"openai": provider})
    config = AIConfiguration("openai", "default", response_model="writer", response_check_model="checker")
    service.configure(config)
    for stage in ("reply", "regeneration", "grounding", "gate", "gate"):
        await service.call_model(stage, "question")
    assert [call.kwargs["model_name"] for call in provider.call_model.await_args_list] == [
        "writer", "writer", "default", "checker", "checker"]
    with pytest.raises(FrozenInstanceError):
        config.default_model = "changed"
    with pytest.raises(AIConfigurationError):
        service.configure(AIConfiguration("openai", "changed"))
    next_run = service.new_run()
    next_run.configure(AIConfiguration("openai", "changed"))
    await next_run.call_model("reply", "question")
    assert provider.call_model.await_args.kwargs["model_name"] == "changed"
    assert service.configuration == config


@pytest.mark.parametrize("settings", [dict(provider="unknown", default_model="x"),
    dict(provider="gemini", default_model=""), dict(provider="gemini", default_model="x", grounding_model=" ")])
def test_invalid_configuration_is_rejected(settings):
    with pytest.raises(AIConfigurationError):
        AIConfiguration(**settings)


async def test_missing_configuration_never_silently_uses_a_model():
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: []))))
    with pytest.raises(AIConfigurationError, match="missing"):
        await load_configuration(db)


@pytest.mark.parametrize("pending", ["new", "dirty", "deleted"])
async def test_read_boundary_never_commits_pending_writes(pending):
    db = SimpleNamespace(new=set(), dirty=set(), deleted=set(), commit=AsyncMock())
    setattr(db, pending, {object()})
    with pytest.raises(RuntimeError, match="pending writes"):
        await finish_read_phase(db)
    db.commit.assert_not_awaited()


async def test_context_releases_both_read_phases_before_returning():
    from tests.unit.test_context_gatherer import build, FakeGemini
    class BoundaryModel(FakeGemini):
        async def call_model_structured(self, *args, **kwargs):
            assert db.commits == 1
            return await super().call_model_structured(*args, **kwargs)
    gatherer, db = build(gemini=BoundaryModel())
    await gatherer.gather("Question", "conversation")
    assert db.commits == 2


async def test_wait_also_releases_initial_read_phase():
    from tests.unit.test_context_gatherer import build, FakeGemini
    gatherer, db = build(gemini=FakeGemini(decision="wait"))
    await gatherer.gather("Question", "conversation")
    assert db.commits == 1


def provider_row(provider, priority, default, **models):
    return {"provider": provider, "use_priority": priority, "default_model": default,
            **dict.fromkeys(MODEL_COLUMNS), **models}


async def test_priority_selection_routes_each_stage_and_keeps_retry_assignment():
    rows = [provider_row("openai", 2, "openai-default", response_model="writer"),
            provider_row("gemini", 1, "gemini-default", grounding_model="grounder")]
    selection = selection_from_rows(rows)
    providers = {name: SimpleNamespace(call_model=AsyncMock(return_value="ok"),
                                      call_model_structured=AsyncMock(return_value={}))
                 for name in ("gemini", "openai")}
    service = AIService(providers)
    service.configure(selection)
    await service.call_model("grounding", "q")
    for stage in ("reply", "regeneration"):
        await service.call_model(stage, "q")
    await service.call_model_structured("gate", "q", "s", {})
    assert providers["gemini"].call_model.await_args.kwargs["model_name"] == "grounder"
    assert [c.kwargs["model_name"] for c in providers["openai"].call_model.await_args_list] == ["writer", "writer"]
    assert providers["gemini"].call_model_structured.await_args.kwargs["model_name"] == "gemini-default"
    rows[0]["response_model"] = "changed"
    assert selection.selection_for("reply") == ("openai", "writer")


def test_highest_priority_explicit_stage_wins_and_unknown_stage_fails():
    selection = selection_from_rows([
        provider_row("gemini", 1, "default", response_model="first"),
        provider_row("openai", 2, "other", response_model="second")])
    assert selection.selection_for("reply") == ("gemini", "first")
    with pytest.raises(AIConfigurationError, match="Unknown"):
        selection.selection_for("unknown")


def test_duplicate_enabled_priorities_are_rejected():
    with pytest.raises(AIConfigurationError, match="unique"):
        selection_from_rows([provider_row("gemini", 1, "a"), provider_row("openai", 1, "b")])


async def test_loader_fetches_only_enabled_rows_without_holding_orm_objects():
    rows = [provider_row("gemini", 1, "a"), provider_row("openai", 2, "b", response_model="writer")]
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
        mappings=lambda: SimpleNamespace(all=lambda: rows))))
    selected = await load_configuration(db)
    assert selected.selection_for("reply") == ("openai", "writer")
    assert "is_active IS true" in str(db.execute.await_args.args[0])
