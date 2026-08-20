from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base


class QuestionBank(Base):
    __tablename__ = "question_bank"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)


# INSERT INTO question_bank (topic, question, answer) VALUES
# (
#     'personality',
#     'What can this negotiation terminal help me with?',
#     'This terminal helps you work through pricing and terms discussions, offering guidance on tactics and generating responses based on your conversation.'
# ),
# (
#     'certificate',
#     'What negotiation strategies does this app support?',
#     'Common strategies include anchoring (opening with a strong but defensible position), mirroring (repeating the last few words as a question), and strategic use of silence after making an offer.'
# ),
# (
#     'past projects',
#     'How much discount can be offered?',
#     'Discounts are supported up to a configured maximum. Anything above the standard threshold typically requires additional approval before being finalized.'
# ),
# (
#     'achievement',
#     'Is my conversation saved?',
#     'Yes — your conversation is tied to your current session and persists as long as your session remains active.'
# ),
# (
#     'expectation',
#     'How does the terminal confirm a final agreement?',
#     'Before finalizing, the terminal restates the agreed-upon terms and asks for explicit confirmation before locking them in.'
# )
# ON CONFLICT (topic) DO UPDATE
# SET question = EXCLUDED.question,
#     answer = EXCLUDED.answer;