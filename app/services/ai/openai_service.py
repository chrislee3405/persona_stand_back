"""OpenAI Chat Completions adapter with bounded calls and no implicit retries."""
import asyncio
import json
import os

import httpx

from app.services.ai.base import AIProviderError, MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS
from app.services.model_collaborate import turn_metrics


def strict_schema(schema):
    """Convert our Gemini-style types to the strict JSON Schema wire format."""
    if isinstance(schema, list):
        return [strict_schema(value) for value in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: strict_schema(value) for key, value in schema.items()}
    if isinstance(result.get("type"), str):
        result["type"] = result["type"].lower()
    if result.get("type") == "object":
        properties = result.get("properties", {})
        required = result.get("required", [])
        for name, value in properties.items():
            if name not in required:
                properties[name] = {"anyOf": [value, {"type": "null"}]}
        result["required"] = list(properties)
        result["additionalProperties"] = False
    return result


class OpenAIService:
    provider = "openai"
    endpoint = "https://api.openai.com/v1/chat/completions"
    key_variable = "OPENAI_API_KEY"
    token_parameter = "max_completion_tokens"
    completion_includes_reasoning = True

    async def _call(self, model_name, user_prompt, system_prompt, schema=None):
        with turn_metrics.model_call(model_name, provider=self.provider):
            key = os.getenv(self.key_variable)
            if not key:
                raise AIProviderError(f"{self.key_variable} is not configured")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            payload = {"model": model_name, "messages": messages, self.token_parameter: MAX_OUTPUT_TOKENS}
            if schema is not None:
                # Some stages expect a scalar or array. Always use an object root
                # on the wire and unwrap it so existing stage contracts stay intact.
                wrapped = {"type": "object", "properties": {"result": strict_schema(schema)},
                           "required": ["result"], "additionalProperties": False}
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "stage_result", "strict": True, "schema": wrapped}}
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                        response = await client.post(self.endpoint, json=payload,
                                                     headers={"Authorization": f"Bearer {key}"})
                        response.raise_for_status()
                        data = response.json()
                usage = data.get("usage") or {}
                def count(value):
                    return value if type(value) is int and value >= 0 else None
                completion = count(usage.get("completion_tokens"))
                thinking = count((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
                visible = completion
                if self.completion_includes_reasoning:
                    visible = (completion - thinking if completion is not None and thinking is not None
                               and completion >= thinking else None)
                turn_metrics.record_usage(
                    count(usage.get("prompt_tokens")),
                    visible,
                    thinking_tokens=thinking,
                    cached_tokens=count((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
                    total_tokens=count(usage.get("total_tokens")),
                )
                choice = data["choices"][0]
                message = choice["message"]
                text = message.get("content")
                turn_metrics.record_response(text, refusal=message.get("refusal"))
                if message.get("refusal") or choice.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
                    raise AIProviderError("AI provider returned no complete usable response")
                if schema is None:
                    return text
                return json.loads(text)["result"]
            except AIProviderError:
                raise
            except (httpx.HTTPError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
                # Never propagate HTTP response bodies or visitor prompts into logs.
                raise AIProviderError("AI provider request failed or returned invalid output") from None

    async def call_model(self, model_name, user_prompt, system_prompt=None):
        return await self._call(model_name, user_prompt, system_prompt)

    async def call_model_structured(self, model_name, user_prompt, system_prompt, schema):
        return await self._call(model_name, user_prompt, system_prompt, schema)
