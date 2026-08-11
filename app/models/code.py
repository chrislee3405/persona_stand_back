from sqlalchemy import Column, Integer, String, Text
from app.database import Base

class InviteCode(Base):
    __tablename__ = "code"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)