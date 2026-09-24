"""Conversation content reaches the logs only through the trace logger, and only with CHAT_TRACE on."""
import logging

import pytest

from app import chat_trace
from app.runtime_settings import chat_trace_enabled
from app.services.model_collaborate_service import ModelCollaborateService
from tests.unit.test_readiness_gate import build_model_service

VISITOR_TEXT = "my private question about the canary project"
REASON = "the visitor wrote about the canary project"


@pytest.fixture
def trace_env(monkeypatch):
    """Sets CHAT_TRACE and re-applies it, restoring the default afterwards."""
    def set_trace(value):
        if value is None:
            monkeypatch.delenv("CHAT_TRACE", raising=False)
        else:
            monkeypatch.setenv("CHAT_TRACE", value)
        chat_trace.configure()

    yield set_trace
    monkeypatch.delenv("CHAT_TRACE", raising=False)
    chat_trace.configure()


class EchoGemini:
    async def call_model(self, model_name, user_prompt, system_prompt=None):
        return f"reply to: {user_prompt}"


async def generate(caplog):
    service = ModelCollaborateService.__new__(ModelCollaborateService)
    service.gemini_service = EchoGemini()
    with caplog.at_level(logging.DEBUG):
        await service._generate_reply("system prompt", VISITOR_TEXT)


def test_trace_is_off_by_default_even_at_debug(trace_env):
    trace_env(None)
    logging.getLogger().setLevel(logging.DEBUG)
    assert chat_trace_enabled() is False
    assert not chat_trace.trace.isEnabledFor(logging.CRITICAL)


async def test_prompts_and_replies_stay_out_of_debug_logs(trace_env, caplog):
    trace_env(None)
    await generate(caplog)
    assert VISITOR_TEXT not in caplog.text


async def test_chat_trace_is_the_explicit_way_to_see_them(trace_env, caplog):
    trace_env("true")
    await generate(caplog)
    assert VISITOR_TEXT in caplog.text
    assert all(r.name == chat_trace.LOGGER_NAME for r in caplog.records if VISITOR_TEXT in r.getMessage())


@pytest.mark.parametrize("decision", ["wait", "no_reply"])
async def test_the_readiness_reason_is_not_in_ordinary_logs(trace_env, caplog, decision):
    # The reason is the model's own description of the visitor's message.
    trace_env(None)
    service = build_model_service(decision=decision, evaluated_through=0)
    service.context_gatherer.gather = _gather_with_reason(service.context_gatherer.gather)
    with caplog.at_level(logging.DEBUG):
        outcome = await service.model_orchestration("hi", "conv-1", "sess-1", "guest")
    assert outcome.decision == decision
    assert REASON not in caplog.text
    # The decision itself is still logged.
    assert f"Readiness gate: {decision}" in caplog.text


def _gather_with_reason(gather):
    async def run(*args, **kwargs):
        context = await gather(*args, **kwargs)
        context["readiness"] = {"decision": context["readiness"]["decision"], "reason": REASON}
        return context
    return run


def test_an_unclear_chat_trace_value_is_refused(monkeypatch):
    monkeypatch.setenv("CHAT_TRACE", "maybe")
    with pytest.raises(RuntimeError, match="CHAT_TRACE"):
        chat_trace_enabled()


@pytest.mark.parametrize('provider_error', [False, True])
def test_exception_details_are_removed_before_later_handlers(trace_env, provider_error):
    import io
    from sqlalchemy.exc import IntegrityError
    from app.safe_logging import configure_safe_logging
    trace_env(None)
    configure_safe_logging()
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger = logging.getLogger('audit.later-handler')
    logger.addHandler(handler)
    try:
        try:
            canary = 'fictional-exception-content'
            if provider_error:
                raise RuntimeError(canary)
            raise IntegrityError('INSERT fictional', (canary,), Exception(canary))
        except Exception:
            logger.exception('Model or database operation failed')
        assert canary not in output.getvalue()
        assert 'details omitted' in output.getvalue()
    finally:
        logger.removeHandler(handler)
