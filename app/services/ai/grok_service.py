"""Grok's compatible text/structured endpoint with its own credential and limit."""
from app.services.ai.openai_service import OpenAIService


class GrokService(OpenAIService):
    provider = "grok"
    endpoint = "https://api.x.ai/v1/chat/completions"
    key_variable = "XAI_API_KEY"
    token_parameter = "max_tokens"
    # xAI reports visible completion and reasoning separately in Chat Completions.
    completion_includes_reasoning = False
