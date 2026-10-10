"""Load a detached, immutable configuration once per turn or summary run."""
from dataclasses import dataclass

from sqlalchemy import select

from app.models.model_configuration import ModelConfiguration, MODEL_COLUMNS


class AIConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class AIConfiguration:
    provider: str
    default_model: str
    readiness_model: str | None = None
    topic_selection_model: str | None = None
    example_selection_model: str | None = None
    grounding_model: str | None = None
    response_model: str | None = None
    response_check_model: str | None = None
    response_split_model: str | None = None
    summary_model: str | None = None

    def __post_init__(self):
        if self.provider not in {"gemini", "openai", "grok"}:
            raise AIConfigurationError("Unsupported AI provider")
        for name in ("default_model", *MODEL_COLUMNS):
            value = getattr(self, name)
            if value is None and name != "default_model":
                continue
            if not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 200:
                raise AIConfigurationError(f"Invalid {name}; use NULL for an unset stage model")

    def model_for(self, stage: str) -> str:
        return getattr(self, stage_column(stage)) or self.default_model

    def selection_for(self, stage: str) -> tuple[str, str]:
        return self.provider, self.model_for(stage)


def stage_column(stage: str) -> str:
    column = {
        "readiness": "readiness_model", "topics": "topic_selection_model",
        "example": "example_selection_model", "grounding": "grounding_model",
        "reply": "response_model", "regeneration": "response_model",
        "gate": "response_check_model", "split": "response_split_model",
        "summary": "summary_model",
    }.get(stage)
    if column is None:
        raise AIConfigurationError("Unknown AI stage")
    return column


@dataclass(frozen=True)
class AISelection:
    """Enabled provider snapshots in ascending, unique priority order."""
    configurations: tuple[AIConfiguration, ...]

    @property
    def provider(self):
        return self.configurations[0].provider

    @property
    def default_model(self):
        return self.configurations[0].default_model

    def selection_for(self, stage: str) -> tuple[str, str]:
        column = stage_column(stage)
        for config in self.configurations:
            model = getattr(config, column)
            if model is not None:
                return config.provider, model
        return self.provider, self.default_model

    def model_for(self, stage: str) -> str:
        return self.selection_for(stage)[1]


def selection_from_rows(rows) -> AISelection:
    rows = sorted(rows, key=lambda row: row["use_priority"])
    if not rows:
        raise AIConfigurationError("Active AI configuration missing; enable at least one provider")
    priorities = [row["use_priority"] for row in rows]
    if any(type(p) is not int or p < 1 for p in priorities) or len(set(priorities)) != len(priorities):
        raise AIConfigurationError("Enabled providers must have unique positive use_priority values")
    return AISelection(tuple(AIConfiguration(provider=row["provider"], default_model=row["default_model"],
                              **{name: row[name] for name in MODEL_COLUMNS}) for row in rows))


async def load_configuration(db) -> AISelection:
    result = await db.execute(select(ModelConfiguration.__table__).where(ModelConfiguration.is_active.is_(True)))
    return selection_from_rows(result.mappings().all())
