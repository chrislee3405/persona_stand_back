from sqlalchemy import Column, Integer, Text
from app.database import Base


class PersonalityReference(Base):
    __tablename__ = "personality_reference"

    id = Column(Integer, primary_key=True, index=True)
    core_personality = Column(Text, nullable=False, unique=True)