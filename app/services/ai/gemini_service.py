import os
import json
from typing import Union

from google import genai
from google.genai import types

# Per-call wall-clock ceiling, milliseconds. Without it a hung Vertex
# connection stalls a chat turn indefinitely: nothing else in the pipeline
# sets a deadline, and a turn makes 6-11 of these calls in sequence, so one
# stuck call holds the visitor's request open until the reverse proxy gives
# up on it. 30s is generous against the ~10s a real turn's slowest call
# takes, while keeping the worst case (a turn that exhausts its
# regeneration budget) inside nginx's proxy_read_timeout -- which is why
# that timeout is now set explicitly in persona_stand_front/nginx.conf
# rather than left at its 60s default. Keep the two in step.
_REQUEST_TIMEOUT_MS = 30_000

# Hard ceiling on generated length, in tokens, applied to every call. This
# is a COST control, not a quality one: nothing else bounds output size, so
# a prompt that induces a runaway generation is billed for whatever the
# model decides to produce. Comfortably above a normal persona reply (a
# few sentences) and above the longest structured payload any caller asks
# for (ResponseGate's violation list). A truncated response surfaces as a
# malformed/short reply, which every caller already handles -- see
# GroundingService's GROUND_FALLBACK and ResponseParser's single-turn
# fallback.
_MAX_OUTPUT_TOKENS = 2048

# vertexai=True routes through Vertex AI (using this environment's GCP
# credentials -- Workload Identity Federation in production) rather than the
# public Gemini Developer API, which would need a separate API key and bill
# differently. Replaces the old `vertexai.init(...)` call -- see
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk
# (vertexai.generative_models is deprecated, removed June 24, 2026).
_client = genai.Client(
    vertexai=True,
    project=os.getenv("GCP_PROJECT_ID"),
    location="global",
    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
)

# Whatever json.loads() can produce -- which is exactly what
# call_model_structured returns, since it's a direct pass-through of the
# parsed response. A schema of {"type": "string", "enum": [...]} yields a
# str; {"type": "object", ...} yields a dict; {"type": "array", ...}
# yields a list; and so on. The actual shape returned depends entirely on
# the `schema` argument the caller passes in.
JSONValue = Union[dict, list, str, int, float, bool, None]


class GeminiService:
    """
    Owns raw calls to Gemini via Vertex AI. Knows nothing about
    conversation flow, topics, or retrieval — just sends prompts and
    returns whatever Gemini responds with.

    BOTH METHODS ARE ASYNC, and use the SDK's async surface
    (`_client.aio`). They were synchronous, called without any threadpool
    offload from `async def` request handlers -- which meant the blocking
    HTTPS request ran ON the event loop. For the whole duration of a chat
    turn (6-11 sequential calls) the single uvicorn worker could serve
    nothing else, including GET /api/site-content, the public portfolio's
    only data source. Every caller in the reply pipeline is already async,
    so awaiting here costs nothing structurally.
    """

    async def call_model(self, model_name: str, user_prompt: str, system_prompt: str | None = None) -> str:
        """
        Sends a prompt to a Gemini model and returns its free-text response.

        Parameters:
        - model_name (str): which Gemini model to call — comes from the caller (e.g. ModelCollaborateService, SummarizationService)
        - user_prompt (str): the prompt content — comes from the caller
        - system_prompt (str | None): optional system instruction — comes from the caller

        Returns:
        - str: the model's generated text — goes back to the calling service. Awaited, not blocking: see the class docstring.
        """
        response = await _client.aio.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            )
        )
        return response.text

    async def call_model_structured(self, model_name: str, user_prompt: str, system_prompt: str, schema: dict) -> JSONValue:
        """
        Sends a prompt to a Gemini model and parses its response as JSON matching the given schema.

        Parameters:
        - model_name (str): which Gemini model to call — comes from the caller (e.g. ModelCollaborateService.find_topic)
        - user_prompt (str): the prompt content — comes from the caller
        - system_prompt (str): system instruction — comes from the caller
        - schema (dict): the expected response JSON schema — comes from the caller

        Returns:
        - JSONValue: the parsed JSON response, shaped by `schema` — goes back to the calling service. Awaited, not blocking: see the class docstring.
        """
        response = await _client.aio.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            )
        )
        return json.loads(response.text)
