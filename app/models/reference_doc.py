from sqlalchemy import Column, Integer, String, Text
from app.database import Base


class DocReference(Base):
    __tablename__ = "doc_reference"

    id = Column(Integer, primary_key=True, index=True)
    document_topic = Column(String, nullable=False, unique=True)
    topic_description = Column(Text, nullable=False, unique=True)
    content = Column(Text, nullable=False)
