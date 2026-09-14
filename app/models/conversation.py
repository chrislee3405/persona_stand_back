from sqlalchemy import Column, Integer, String, Text, ForeignKey, UniqueConstraint, DateTime
from sqlalchemy.sql import func
from app.constants import GUEST_CODE
from app.database import Base


# No ORM relationship() between these two. Messages are always read through
# ConversationService's explicit queries; a relationship nobody used was an
# implicit lazy load waiting to happen, and lazy loads raise under AsyncSession.


class Conversation(Base):
    __tablename__ = "conversation"

    conversation_id = Column(String, primary_key=True)  # backend-generated UUID (see ConversationService)
    owner_session_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    code = Column(String, nullable=True, default=GUEST_CODE)
    last_summarized_index = Column(Integer, nullable=False, default=-1)
    summary = Column(Text, nullable=True)  # running summary, appended every N pairs


class Message(Base):
    __tablename__ = "message"
    __table_args__ = (
        UniqueConstraint("conversation_id", "order_index", name="uq_conversation_order"),
    )

    message_id = Column(Integer, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversation.conversation_id"), nullable=False, index=True)
    order_index = Column(Integer, nullable=False)   # 0, 1, 2... strictly increasing per conversation
    sender = Column(String, nullable=False)          # one of app.constants.Sender
    text = Column(Text, nullable=False)
    # Owner-review metadata, written on persona replies and never read by the
    # application: which scenario/document topics grounded this reply, so a
    # questionable answer can be traced back to its references in pgAdmin.
    selected_scenario = Column(String, nullable=True)
    selected_document = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
