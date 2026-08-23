from sqlalchemy import Column, Integer, String, Text
from app.database import Base


class ScenarioReference(Base):
    __tablename__ = "scenario_reference"

    id = Column(Integer, primary_key=True, index=True)
    scenario_topic = Column(String, nullable=False, unique=True)
    topic_description = Column(Text, nullable=False, unique=True)
    content = Column(Text, nullable=False)