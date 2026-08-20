import os
import json
from typing import Union

from vertexai.generative_models import GenerativeModel
import vertexai

vertexai.init(project=os.getenv("GCP_PROJECT_ID"), location="global")

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
    """

    def call_model(self, model_name: str, user_prompt: str, system_prompt: str | None = None) -> str:
        model = GenerativeModel(model_name, system_instruction=system_prompt)
        response = model.generate_content(contents=user_prompt)
        return response.text

    def call_model_structured(self, model_name: str, user_prompt: str, system_prompt: str, schema: dict) -> JSONValue:
        model = GenerativeModel(model_name, system_instruction=system_prompt)
        response = model.generate_content(
            contents=user_prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        )
        return json.loads(response.text)