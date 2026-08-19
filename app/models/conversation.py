from sqlalchemy import Column, Integer, String, Text, ForeignKey, UniqueConstraint, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Conversation(Base):
    __tablename__ = "conversation"

    conversation_id = Column(String, primary_key=True)  # backend-generated UUID (see ConversationService)
    code = Column(String, nullable=True, default="GUEST")
    summary = Column(Text, nullable=True)  # running summary, appended every N pairs
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # The session_id (from the signed httpOnly session cookie) that created
    # this conversation. Used by ConversationService.assert_ownership to
    # decide who's allowed to read/write it — for guest conversations this
    # is the sole authorization check; for code-owned conversations it's
    # kept for audit/creator tracking even though ownership there is
    # decided by matching verified_code instead. Nullable only so existing
    # rows created before this column existed don't break; new rows always
    # set it.
    owner_session_id = Column(String, nullable=True, index=True)

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
    sender = Column(String, nullable=False)          # 'user' | 'backend'
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")