from fastapi import Depends
from app.services.ai.gemini_service import GeminiService


class ModelCollaborateService:
    """
    Manages how a raw user message moves through the multi-model flow —
    which models get called, in what order, and how outputs feed into the
    next step. Delegates every actual model call to GeminiService; doesn't
    know or care which AI provider is behind that call.
    """

    def __init__(self, gemini_service: GeminiService = Depends()):
        self.gemini_service = gemini_service

    async def route_user_message(self, user_message: str) -> str:
        """
        Basic single-step flow for now: send the raw user message straight
        through and return the response. Expand this later to insert
        topic classification, retrieval, etc. between steps.
        """
        response = self.gemini_service.call_model(
            model_name="gemini-2.5-flash",
            user_prompt=user_message,
            system_prompt="You are a helpful assistant."
        )
        return response