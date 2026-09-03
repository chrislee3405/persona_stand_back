from sqlalchemy import Column, Integer, String, Text, ForeignKey, UniqueConstraint, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Conversation(Base):
    __tablename__ = "conversation"

    conversation_id = Column(String, primary_key=True)  # backend-generated UUID (see ConversationService)
    owner_session_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    code = Column(String, nullable=True, default="GUEST")
    last_summarized_index = Column(Integer, nullable=False, default=-1)
    summary = Column(Text, nullable=True)  # running summary, appended every N pairs

    messages = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.order_index"
    )


class Message(Base):
    __tablename__ = "message"
    __table_args__ = (
        UniqueConstraint("conversation_id", "order_index", name="uq_conversation_order"),
    )

    message_id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversation.conversation_id"), nullable=False, index=True)
    order_index = Column(Integer, nullable=False)   # 0, 1, 2... strictly increasing per conversation
    sender = Column(String, nullable=False)          # 'user' | 'backend' | 'system' | 'error' | 'regen'
    text = Column(Text, nullable=False)
    selected_scenario = Column(String, nullable=True)
    selected_document = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


