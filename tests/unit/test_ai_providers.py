import json

import httpx
import pytest

from app.services.ai.base import AIProviderError
from app.services.ai.openai_service import OpenAIService, strict_schema
from app.services.ai.grok_service import GrokService
from app.services.model_collaborate import turn_metrics


@pytest.mark.parametrize("adapter", [OpenAIService, GrokService])
async def test_adapters_translate_scalar_schema_and_record_provider(monkeypatch, adapter):
    captured = []
    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '{"result":"0"}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 17 if adapter is OpenAIService else 19}})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setenv(adapter.key_variable, "test-key")
    with turn_metrics.turn() as metrics:
        with turn_metrics.stage("example"):
            result = await adapter().call_model_structured("configured-model", "question", "instructions",
                                                          {"type": "STRING", "enum": ["0", "none"]})
    assert result == "0"
    assert captured[0][adapter.token_parameter] == 2048
    schema = captured[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["result"]["type"] == "string"
    request = metrics.snapshot(status="completed", kind="message")["stages"]["example"]["requests"][0]
    assert request["provider"] == adapter.provider
    assert request["model"] == "configured-model"
    assert request["output_tokens"] == (5 if adapter is OpenAIService else 7)
    assert request["thinking_tokens"] == 2
    assert request["response"] == '{"result":"0"}'


@pytest.mark.parametrize("response", [
    {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"refusal": "refused"}}]},
    {"choices": []},
])
async def test_unusable_outputs_fail_without_retries(monkeypatch, response):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=response)
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(AIProviderError):
        await OpenAIService().call_model("model", "question")
    assert len(requests) == 1


def test_schema_translation_does_not_mutate_stage_schema():
    schema = {"type": "OBJECT", "properties": {"answer": {"type": "STRING"}}}
    wire = strict_schema(schema)
    assert schema["type"] == "OBJECT"
    assert "required" not in schema
    assert wire["required"] == ["answer"]
    assert wire["properties"]["answer"]["anyOf"][1] == {"type": "null"}


@pytest.mark.parametrize("content,finish,refusal", [
    ("{incomplete", "stop", None),
    ("partial output", "length", None),
    (None, "stop", "Cannot answer this request"),
])
@pytest.mark.parametrize("adapter", [OpenAIService, GrokService])
async def test_failed_outputs_remain_in_request_record(monkeypatch, adapter, content, finish, refusal):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"finish_reason": finish, "message": {"content": content, "refusal": refusal}}]}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    monkeypatch.setenv(adapter.key_variable, "test-key")
    with turn_metrics.turn() as metrics, turn_metrics.stage("grounding"):
        with pytest.raises(AIProviderError):
            await adapter().call_model_structured("model", "private prompt", "system", {"type": "OBJECT"})
    entry = metrics.snapshot(status="failed", kind="message")["stages"]["grounding"]["requests"][0]
    assert entry["response"] == content
    assert entry["status"] == "failed"
    if refusal:
        assert entry["refusal"] == refusal
    assert "private prompt" not in str(entry)
