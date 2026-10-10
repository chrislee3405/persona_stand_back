"""Request-scoped routing; changing DB settings never mutates an in-flight turn."""
from fastapi import Depends

from app.services.ai.base import AIProvider
from app.services.ai.configuration import AIConfiguration, AISelection, AIConfigurationError
from app.services.ai.gemini_service import GeminiService
from app.services.ai.openai_service import OpenAIService
from app.services.ai.grok_service import GrokService


class AIService:
    def __init__(self, providers: dict[str, AIProvider]):
        self.providers = providers
        self.configuration: AIConfiguration | AISelection | None = None

    def configure(self, configuration: AIConfiguration | AISelection):
        if self.configuration is not None and self.configuration != configuration:
            raise AIConfigurationError("Cannot change AI configuration during a run")
        self.configuration = configuration

    def _resolve(self, stage):
        if self.configuration is None:
            raise AIConfigurationError("AI configuration has not been loaded")
        provider, model = self.configuration.selection_for(stage)
        return self.providers[provider], model

    async def call_model(self, stage: str, user_prompt: str, system_prompt: str | None = None):
        provider, model = self._resolve(stage)
        return await provider.call_model(model_name=model, user_prompt=user_prompt, system_prompt=system_prompt)

    async def call_model_structured(self, stage: str, user_prompt: str, system_prompt: str, schema: dict):
        provider, model = self._resolve(stage)
        return await provider.call_model_structured(model_name=model, user_prompt=user_prompt,
                                                    system_prompt=system_prompt, schema=schema)

    def new_run(self):
        return AIService(self.providers)


def get_ai_service(gemini: GeminiService = Depends()) -> AIService:
    # Constructors do no networking and require no inactive-provider credentials.
    return AIService({"gemini": gemini, "openai": OpenAIService(), "grok": GrokService()})
