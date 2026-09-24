"""A deterministic stand-in at the external model boundary, not the API layer."""
import asyncio

REPLY = "I built a fictional portfolio with automated tests."
FAILURE_MESSAGE = "Please simulate a model failure."


class FakeGemini:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.delay = 0
        self.malformed_split = False
        # What the readiness gate decides. "respond" keeps every existing
        # test running the full pipeline exactly as before the gate existed;
        # a test that wants a held or ignored message sets one of the others.
        self.readiness_decision = "respond"

    async def call_model(self, model_name, user_prompt, system_prompt=None):
        self.calls.append(("text", user_prompt))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failure:
            raise self.failure
        # Only the explicit fictional E2E message selects failure. There is
        # no test route, secret header or production environment switch.
        if FAILURE_MESSAGE in user_prompt:
            raise TimeoutError("simulated model failure")
        return REPLY

    async def call_model_structured(self, model_name, user_prompt, system_prompt, schema):
        self.calls.append(("structured", user_prompt))
        properties = schema.get("properties", {})
        if "decision" in properties:
            return {"decision": self.readiness_decision, "reason": "fake readiness verdict"}
        if "question_type" in properties:
            return {"question_type": "factual", "coverage": "full", "facts": [REPLY], "missing": ""}
        if "result" in properties:
            return {"result": "<pass>", "violations": []}
        if "document_topics" in properties or "scenario_topics" in properties:
            return {key: value["items"]["enum"][:1] for key, value in properties.items()}
        if schema.get("type", "").upper() == "STRING":
            return "none"
        if schema.get("type", "").upper() == "ARRAY":
            return None if self.malformed_split else [user_prompt.removeprefix("Original response:\n")]
        raise AssertionError(f"Unhandled Gemini schema: {schema}")
