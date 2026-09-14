import os
import json
from typing import Union

from google import genai
from google.genai import types

# Per-call wall-clock ceiling, milliseconds. Bounds ONE hung Vertex
# connection; 30s is generous against the ~10s a real turn's slowest call
# takes. It does NOT bound a whole turn: a turn makes 4-12 of these calls in
# sequence, so the per-call ceiling alone allows ~360s. The whole turn is
# bounded by TURN_DEADLINE_SECONDS (app/constants.py), which is the value
# that has to stay under nginx's proxy_read_timeout -- not this one.
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


class GeminiEmptyResponseError(Exception):
    """
    Raised when a Gemini call comes back with no usable text.

    `response.text` is None whenever the candidate carries no text part --
    a safety-filter block, a recitation stop, or finish_reason=MAX_TOKENS
    with nothing emitted (reachable here: _MAX_OUTPUT_TOKENS caps every
    call at 2048, and ResponseGate's violation list is the longest
    structured payload any caller asks for).

    Both methods below are annotated `-> str` / `-> JSONValue`, and both
    used to return that None straight to the caller, where it travelled
    several frames before failing somewhere unrelated: `json.loads(None)`
    raised TypeError; a None reply reached ResponseGate.check and died on
    `v["quote"] in current_response` with "argument of type 'NoneType' is
    not iterable"; `is_fallback_response(None)` died on .startswith. Every
    one of those surfaced to the visitor as the generic "something went
    wrong", for what is an ordinary event for a chatbot.

    Raising a named error at the boundary instead means the callers that
    ALREADY degrade gracefully -- GroundingService's GROUND_FALLBACK,
    ResponseParser's single-turn fallback, ResponseGate passing the
    response through unaudited -- catch it with the except clauses they
    already have, and only the callers that genuinely cannot continue fail
    the turn.
    """

    def __init__(self, model_name: str, finish_reason: object = None):
        self.model_name = model_name
        self.finish_reason = finish_reason
        super().__init__(
            f"{model_name} returned no text (finish_reason={finish_reason!r})"
        )


def _require_text(response, model_name: str) -> str:
    """
    Pulls the text out of a Gemini response, raising rather than returning None when there is none.

    Parameters:
    - response: the SDK's GenerateContentResponse -- comes from either call below
    - model_name (str): which model produced it -- comes from the caller, for the error message

    Returns:
    - str: the generated text -- goes back to the calling method

    Raises:
    - GeminiEmptyResponseError: the response carries no text part. The
      candidate's finish_reason is attached where the SDK exposes one, since
      that is what distinguishes a safety block from a truncation and is the
      first thing anyone reading the log will want.
    """
    text = response.text
    if text:
        return text

    finish_reason = None
    candidates = getattr(response, "candidates", None)
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
    raise GeminiEmptyResponseError(model_name, finish_reason)


class GeminiService:
    """
    Owns raw calls to Gemini via Vertex AI. Knows nothing about
    conversation flow, topics, or retrieval — just sends prompts and
    returns whatever Gemini responds with.

    BOTH METHODS ARE ASYNC, and use the SDK's async surface
    (`_client.aio`). They were synchronous, called without any threadpool
    offload from `async def` request handlers -- which meant the blocking
    HTTPS request ran ON the event loop. For the whole duration of a chat
    turn (up to 12 sequential calls) the single uvicorn worker could serve
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

        Raises:
        - GeminiEmptyResponseError: the model returned no text at all (safety block, truncation). Never returns None despite the SDK being able to.
        """
        response = await _client.aio.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            )
        )
        return _require_text(response, model_name)

    async def call_model_structured(self, model_name: str, user_prompt: str, system_prompt: str, schema: dict) -> JSONValue:
        """
        Sends a prompt to a Gemini model and parses its response as JSON matching the given schema.

        Parameters:
        - model_name (str): which Gemini model to call — comes from the caller (e.g. ContextGatherer._select_relevant_topics, GroundingService.ground)
        - user_prompt (str): the prompt content — comes from the caller
        - system_prompt (str): system instruction — comes from the caller
        - schema (dict): the expected response JSON schema — comes from the caller

        Returns:
        - JSONValue: the parsed JSON response, shaped by `schema` — goes back to the calling service. Awaited, not blocking: see the class docstring.

        Raises:
        - GeminiEmptyResponseError: the model returned no text at all, so there is nothing to parse.
        - json.JSONDecodeError: the model returned text that is not valid JSON (a truncated payload does this). Callers that can degrade already catch broadly; see GroundingService.ground.
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
        return json.loads(_require_text(response, model_name))
