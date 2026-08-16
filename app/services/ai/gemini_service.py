import os
import json

from vertexai.generative_models import GenerativeModel
import vertexai

vertexai.init(project=os.getenv("GCP_PROJECT_ID"), location="global")


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

    def call_model_structured(
        self, model_name: str, user_prompt: str, system_prompt: str, schema: dict
    ) -> dict:
        model = GenerativeModel(model_name, system_instruction=system_prompt)
        response = model.generate_content(
            contents=user_prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        )
        return json.loads(response.text)