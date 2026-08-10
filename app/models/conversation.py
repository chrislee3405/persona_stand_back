from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base

class UserInput(Base):
    __tablename__ = "conversation"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, nullable=True, default="GUEST")
    content = Column(JSONB, nullable=False)