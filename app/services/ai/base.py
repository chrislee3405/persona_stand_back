"""Provider contract. Adapters receive concrete model names, never stage names."""
from typing import Protocol, Union

JSONValue = Union[dict, list, str, int, float, bool, None]
REQUEST_TIMEOUT_SECONDS = 30
MAX_OUTPUT_TOKENS = 2048


class AIProviderError(RuntimeError):
    """Content-free provider failure, safe for existing error logging."""


class AIProvider(Protocol):
    async def call_model(self, model_name: str, user_prompt: str, system_prompt: str | None = None) -> str: ...
    async def call_model_structured(self, model_name: str, user_prompt: str, system_prompt: str, schema: dict) -> JSONValue: ...
