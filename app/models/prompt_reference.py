from sqlalchemy import Column, Integer, String, Text
from app.database import Base


class QuestionBank(Base):
    __tablename__ = "question_bank"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)


class DocReference(Base):
    __tablename__ = "doc_reference"

    id = Column(Integer, primary_key=True, index=True)
    document_topic = Column(String, nullable=False, unique=True)
    topic_description = Column(Text, nullable=False, unique=True)
    content = Column(Text, nullable=False)


class PersonalityReference(Base):
    __tablename__ = "personality_reference"

    id = Column(Integer, primary_key=True, index=True)
    legal_name = Column(Text, nullable=False)
    prefer_name = Column(Text, nullable=False)
    cluture_background = Column(Text, nullable=False)
    core_personality = Column(Text, nullable=False)


class ScenarioReference(Base):
    __tablename__ = "scenario_reference"

    id = Column(Integer, primary_key=True, index=True)
    scenario_topic = Column(String, nullable=False, unique=True)
    topic_description = Column(Text, nullable=False, unique=True)
    content = Column(Text, nullable=False)
