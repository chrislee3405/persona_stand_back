from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base


class ModelReference(Base):
    __tablename__ = "model_reference"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False, unique=True)
    content = Column(JSONB, nullable=False)



# INSERT INTO model_reference (topic, content) VALUES
# (
#     'personality',
#     '{
#         "tone": "professional_warm",
#         "opening_lines": [
#             "Now we are talking about personality"
#         ]
#     }'::jsonb
# ),
# (
#     'certificate',
#     '{
#         "tone": "professional_warm",
#         "opening_lines": [
#             "Now we are talking about cert"
#         ]
#     }'::jsonb
# )
# ON CONFLICT (topic) DO UPDATE
# SET content = EXCLUDED.content;