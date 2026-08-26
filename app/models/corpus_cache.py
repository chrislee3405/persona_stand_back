from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base


class CorpusCache(Base):
    """
    Holds the precomputed BM25 corpus derived from question_bank (per-question
    term frequencies/size, plus corpus-wide avg_doc_length/df_dict/idf_dict),
    so a fresh app process can load a ready-made corpus instead of
    recomputing it from every question_bank row on first use.

    The app owner deletes this row after updating question_bank to force
    recomputation on the next request -- see BM25Service._get_corpus. Rows
    are never updated in place, only inserted/deleted, so no uniqueness is
    enforced: a rare concurrent-first-request race could momentarily leave
    two rows, but reads always take the first one and a DELETE removes all
    of them regardless, so a stray duplicate is harmless.
    """
    __tablename__ = "corpus_cache"

    id = Column(Integer, primary_key=True)
    data = Column(JSONB, nullable=False)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
