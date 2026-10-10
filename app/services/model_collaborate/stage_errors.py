"""Content-free failure records and technical errors for model stages."""
import logging

from app.services.model_collaborate import turn_metrics

logger = logging.getLogger(__name__)
_STAGES = {"readiness", "topics", "example", "grounding", "reply", "gate", "regeneration", "split", "summary"}
_REASONS = {"call_failed", "invalid_output"}
_FALLBACKS = {"abort", "respond", "omit_example", "single_message", "keep_summary"}


class ModelStageError(RuntimeError):
    """Safe to log: never contains provider exception text, prompts or replies."""

    def __init__(self, stage: str, reason: str):
        if stage not in _STAGES or reason not in _REASONS:
            raise ValueError("Unknown model failure code")
        self.stage = stage
        self.reason = reason
        description = {
            "grounding": "Unable to verify supporting information",
            "gate": "Unable to verify reply evidence",
        }.get(stage, "Unable to complete model stage")
        super().__init__(f"{description} (stage={stage}, reason={reason})")


def record_failure(stage: str, reason: str, fallback: str) -> None:
    if stage not in _STAGES or reason not in _REASONS or fallback not in _FALLBACKS:
        raise ValueError("Unknown model failure code")
    turn_metrics.record_failure(stage, reason=reason, fallback=fallback)
    logger.warning("Model stage failed: stage=%s reason=%s fallback=%s", stage, reason, fallback)


def stage_error(stage: str, reason: str) -> ModelStageError:
    record_failure(stage, reason, "abort")
    return ModelStageError(stage, reason)
