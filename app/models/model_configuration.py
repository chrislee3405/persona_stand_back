"""Owner-controlled provider configurations; credentials live outside the DB."""
from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Index, Integer, String, func, text

from app.database import Base


MODEL_COLUMNS = (
    "readiness_model", "topic_selection_model", "example_selection_model",
    "grounding_model", "response_model", "response_check_model",
    "response_split_model", "summary_model",
)


class ModelConfiguration(Base):
    __tablename__ = "model_configuration"
    __table_args__ = (
        Index("mc_provider_unique", "provider", unique=True),
        Index("mc_enabled_priority_unique", "use_priority", unique=True, postgresql_where=text("is_active")),
        CheckConstraint("use_priority > 0", name="mc_use_priority_positive"),
        CheckConstraint("provider IN ('gemini', 'openai', 'grok')", name="model_configuration_provider"),
        CheckConstraint("length(trim(default_model)) > 0", name="model_configuration_default"),
        *(CheckConstraint(f"{name} IS NULL OR length(trim({name})) > 0", name=f"mc_{name}_nonempty")
          for name in MODEL_COLUMNS),
    )

    id = Column(Integer, primary_key=True, autoincrement=False)
    provider = Column(String(16), nullable=False)
    is_active = Column(Boolean, nullable=False, default=False, server_default="false")
    # Lower values are checked first among enabled providers for each stage.
    use_priority = Column(Integer, nullable=False, default=1, server_default="1")
    default_model = Column(String(200), nullable=False)
    readiness_model = Column(String(200))
    topic_selection_model = Column(String(200))
    example_selection_model = Column(String(200))
    grounding_model = Column(String(200))
    response_model = Column(String(200))
    response_check_model = Column(String(200))
    response_split_model = Column(String(200))
    summary_model = Column(String(200))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
