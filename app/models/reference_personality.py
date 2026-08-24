from sqlalchemy import Column, Integer, Text
from app.database import Base


class PersonalityReference(Base):
    __tablename__ = "personality_reference"

    id = Column(Integer, primary_key=True, index=True)
    legal_name = Column(Text, nullable=False)
    prefer_name = Column(Text, nullable=False)
    cluture_background = Column(Text, nullable=False)
    core_personality = Column(Text, nullable=False)