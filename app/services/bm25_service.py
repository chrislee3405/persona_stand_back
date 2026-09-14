import os
import math
import string
import logging

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stemming.porter2 import stem

from app.database import get_db
from app.models import prompt_reference as question_bank_models
from app.models.corpus_cache import CorpusCache

logger = logging.getLogger(__name__)

_STOP_WORDS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "resources", "common-english-words.txt"
)


def _load_stop_words(path: str = _STOP_WORDS_PATH) -> frozenset:
    """
    Loads the stop-word list used to filter out common terms during BM25 scoring.

    Parameters:
    - path (str): path to the comma-separated stop-words file — defaults to _STOP_WORDS_PATH

    Returns:
    - frozenset: the loaded stop words — goes into _DEFAULT_STOP_WORDS, used by every BM25Service instance
    """
    try:
        with open(path, "r") as f:
            words = [w.strip().lower() for w in f.read().split(",") if w.strip()]
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Stop word file not found at {path}. QuestionBankService "
            "requires this file to produce correct BM25 scores — check "
            "that app/resources/common-english-words.txt is present and "
            "was copied into the deployed image/container."
        ) from e

    if not words:
        raise RuntimeError(f"Stop word file at {path} was read but contains no words.")

    return frozenset(words)


_DEFAULT_STOP_WORDS = _load_stop_words()


def process_line(line: str) -> str:
    """
    Strips <p>/</p> tags, removes digits, and replaces punctuation with spaces in a line of text.

    Parameters:
    - line (str): raw text — comes from _extract_terms

    Returns:
    - str: the cleaned line — goes to _extract_terms for tokenization
    """
    line = line.replace("<p>", "").replace("</p>", "")
    line = line.translate(str.maketrans('', '', string.digits)).translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    )
    return line


def _extract_terms(text: str, stop_words: frozenset) -> list[str]:
    """
    Tokenizes text into stemmed, filtered terms for BM25 scoring.

    Parameters:
    - text (str): raw text to tokenize — comes from find_similar_questions (a question or the user's message)
    - stop_words (frozenset): words to exclude — comes from BM25Service.stop_words

    Returns:
    - list[str]: stemmed terms, length > 2 and not stop words — goes to _term_freqs / find_similar_questions
    """
    processed = process_line(text.strip())
    terms = []
    for term in processed.split():
        term = stem(term.lower())
        if len(term) > 2 and term not in stop_words:
            terms.append(term)
    return terms


def _term_freqs(terms: list[str]) -> dict[str, int]:
    """
    Counts how many times each term occurs in a list of terms.

    Parameters:
    - terms (list[str]): terms to count — comes from _extract_terms

    Returns:
    - dict[str, int]: term to occurrence count — goes to _bm25_score / find_similar_questions
    """
    freqs: dict[str, int] = {}
    for term in terms:
        freqs[term] = freqs.get(term, 0) + 1
    return freqs


def _document_frequency(term_freq_list: list[dict[str, int]]) -> dict[str, int]:
    """
    Counts how many documents each term appears in at least once.

    Parameters:
    - term_freq_list (list[dict[str, int]]): per-document term frequency maps — comes from find_similar_questions

    Returns:
    - dict[str, int]: term to document count — goes to _compute_idf
    """
    df_dict: dict[str, int] = {}
    for term_freqs in term_freq_list:
        for term in term_freqs.keys():
            df_dict[term] = df_dict.get(term, 0) + 1
    return df_dict


def _compute_idf(df_dict: dict[str, int], ndocs: int, epsilon: float = 0.25) -> dict[str, float]:
    """
    Computes per-term IDF weights, flooring negative values so common terms don't penalize scores.

    Parameters:
    - df_dict (dict[str, int]): term to document frequency — comes from _document_frequency
    - ndocs (int): total number of documents in the corpus — comes from find_similar_questions
    - epsilon (float): floor multiplier applied to the average IDF — defaults to 0.25

    Returns:
    - dict[str, float]: term to IDF weight — goes to _bm25_score via find_similar_questions

    THE SMALL-CORPUS FLOOR. Normally a non-positive IDF is floored to
    `epsilon x average positive IDF`. That average does not exist when NO term
    has a positive IDF -- and with fewer than three documents that is always
    the case: one document gives every term log(1/((1.5)/(0.5))) = -1.10, and
    two give a term in one of them log(1/1.0) = 0.0, which is not > 0 either.
    The average then came out as 0, the floor as 0, every IDF as 0, every
    score as 0, and find_similar_questions -- whose floor is exclusive -- threw
    every candidate away. Retrieval silently returned nothing, so the persona
    answered without the stored example it should have had.

    With no positive IDF to average, the floor is `epsilon` itself. Every term
    then carries the same small weight, so ranking falls back to plain term
    overlap -- a question sharing more words with the message still scores
    higher -- which is the only meaningful signal a one- or two-document
    corpus has. A corpus of three or more documents is unaffected.
    """
    raw_idf = {}
    for term, n_i in df_dict.items():
        left_denominator = (n_i + 0.5) / (ndocs - n_i + 0.5)
        raw_idf[term] = math.log(1 / left_denominator)

    positive_values = [v for v in raw_idf.values() if v > 0]
    if positive_values:
        floor = epsilon * (sum(positive_values) / len(positive_values))
    else:
        floor = epsilon
        if raw_idf:
            logger.warning(
                "BM25 corpus has %d document(s) and no term with a positive IDF -- "
                "falling back to a flat IDF of %.2f so retrieval still ranks by term overlap. "
                "Add question_bank rows (3 or more) for real IDF weighting.",
                ndocs, epsilon,
            )

    return {term: (v if v > 0 else floor) for term, v in raw_idf.items()}


def _bm25_score(query_term_freqs: dict[str, int], doc_term_freqs: dict[str, int], doc_size: int, avg_doc_length: float, idf_dict: dict[str, float], k1: float = 1.2, k2: float = 100, b: float = 0.4) -> float:
    """
    Computes the BM25 relevance score of one document against one query.

    Parameters:
    - query_term_freqs (dict[str, int]): query term counts — comes from find_similar_questions
    - doc_term_freqs (dict[str, int]): document term counts — comes from find_similar_questions
    - doc_size (int): number of terms in the document — comes from find_similar_questions
    - avg_doc_length (float): average document length across the corpus — comes from find_similar_questions
    - idf_dict (dict[str, float]): term to IDF weight — comes from _compute_idf
    - k1 (float): term frequency saturation constant — defaults to 1.2
    - k2 (float): query term frequency saturation constant — defaults to 100
    - b (float): document length normalization constant — defaults to 0.4

    Returns:
    - float: the BM25 score — goes to find_similar_questions for ranking
    """
    if avg_doc_length == 0:
        return 0.0

    K = k1 * ((1 - b) + b * (doc_size / avg_doc_length))
    score = 0.0

    for q_term, qf_i in query_term_freqs.items():
        if q_term in doc_term_freqs:
            idf = idf_dict.get(q_term, 0.0)
            mid = ((k1 + 1) * doc_term_freqs[q_term]) / (K + doc_term_freqs[q_term])
            right = ((k2 + 1) * qf_i) / (k2 + qf_i)
            score += idf * mid * right

    return score


class BM25Service:
    """
    Keyword-overlap (BM25) retrieval over QuestionBank rows, using the
    same preprocessing (stopword filtering, Porter2 stemming) and BM25
    formula as the original assignment — reimplemented against QuestionBank
    rows instead of parsed XML documents, without the linked-list/printer
    infrastructure that was specific to writing the assignment's output
    files.

    This is lexical matching, not semantic matching: it scores shared
    stemmed word roots, not meaning. "How much discount can I get?" and
    "What's the maximum markdown available?" ask the same thing but share
    almost no stemmed terms, so this would likely score that pair near
    zero. If paraphrase-level matching matters, an embedding-based
    approach is the next step up from this.
    """

    # Class-level, not instance-level: FastAPI constructs a fresh BM25Service
    # per request, so an instance attribute would never survive between
    # requests. This persists for the lifetime of the running process, shared
    # across every request handled by it. None means "not loaded into this
    # process yet" -- distinct from an empty-but-loaded corpus.
    _corpus_cache: dict | None = None
    # The corpus_cache.id the in-process copy above was built from, so a
    # cleared or replaced row is noticed. Without it the class attribute was
    # consulted first and never re-checked, which made the documented
    # invalidation procedure -- "delete the corpus_cache row after updating
    # question_bank" (see app/models/corpus_cache.py) -- a silent no-op until
    # the container restarted: new question_bank rows were never retrieved and
    # nothing said so. -1 means "nothing cached yet".
    _corpus_cache_id: int = -1

    def __init__(self, db: AsyncSession = Depends(get_db)):
        """
        Stores the injected database session and default stop-word set.

        Parameters:
        - db (AsyncSession): SQLAlchemy async session — injected by FastAPI via get_db

        Returns:
        - None: sets self.db and self.stop_words
        """
        self.db = db
        self.stop_words = _DEFAULT_STOP_WORDS

    async def _load_all(self) -> list[question_bank_models.QuestionBank]:
        """
        Fetches every row from the question_bank table.

        Parameters:
        - none

        Returns:
        - list[QuestionBank]: all question_bank rows — goes to _compute_corpus
        """
        result = await self.db.execute(select(question_bank_models.QuestionBank))
        return list(result.scalars().all())

    async def _compute_corpus(self) -> dict | None:
        """
        Computes the BM25 corpus (per-question term frequencies/size, plus corpus-wide stats) from every question_bank row.

        Parameters:
        - none

        Returns:
        - dict | None: {"documents": [{"question", "answer", "term_freqs", "size"}, ...], "avg_doc_length", "idf_dict"}, or None if question_bank is empty — goes to _get_corpus, which caches it in memory and persists it to corpus_cache
        """
        rows = await self._load_all()
        if not rows:
            return None

        documents = []
        for row in rows:
            terms = _extract_terms(row.question, self.stop_words)
            documents.append({
                "question": row.question,
                "answer": row.answer,
                "term_freqs": _term_freqs(terms),
                "size": len(terms)
            })

        doc_sizes = [doc["size"] for doc in documents]
        avg_doc_length = sum(doc_sizes) / len(doc_sizes) if doc_sizes else 0
        df_dict = _document_frequency([doc["term_freqs"] for doc in documents])
        idf_dict = _compute_idf(df_dict, ndocs=len(rows))

        # df_dict is an intermediate for _compute_idf only -- it is never read
        # back by find_similar_questions, so it is deliberately NOT stored:
        # keeping it bloated every cached corpus row by roughly the vocabulary
        # size for nothing. Older corpus_cache rows may still contain it; the
        # extra key is simply ignored on read.
        return {
            "documents": documents,
            "avg_doc_length": avg_doc_length,
            "idf_dict": idf_dict
        }

    async def _get_corpus(self) -> dict | None:
        """
        Returns the BM25 corpus, reusing the in-process copy when the corpus_cache row it was built from is still the current one, and otherwise loading or recomputing it.

        Parameters:
        - none

        Returns:
        - dict | None: the corpus (see _compute_corpus for shape), or None if question_bank is empty — goes to find_similar_questions

        THE POINT OF THE ID CHECK. The in-process copy used to be consulted
        first and never re-checked, so once any request in this process had
        populated it, deleting the corpus_cache row -- the invalidation
        procedure app/models/corpus_cache.py documents -- did nothing until
        the container restarted. New question_bank rows were silently never
        retrieved, so `similar_examples` stayed empty, so the persona answered
        without the stored example it should have had. No error, no log line,
        nothing to notice from the outside.

        One indexed `SELECT id ... LIMIT 1` per turn buys the fix. That is
        nothing beside the 4-12 Gemini calls the same turn makes, and it
        catches BOTH ways the corpus can change: the row being deleted (the
        owner forcing a rebuild) and the row being replaced (a newer corpus
        computed by another worker).

        The three outcomes:
        - no row at all      -> the owner cleared it: recompute from
                                question_bank, persist, and adopt the new row.
        - a different row id -> somebody else recomputed it: load theirs.
        - the same row id    -> the in-process copy is current: use it.
        """
        result = await self.db.execute(
            select(CorpusCache.id).order_by(CorpusCache.id.desc()).limit(1)
        )
        current_id = result.scalar_one_or_none()

        if current_id is None:
            # Cleared (or never built). Drop whatever this process was holding
            # BEFORE recomputing, so a failure here cannot leave a stale copy
            # looking current.
            BM25Service._corpus_cache = None
            BM25Service._corpus_cache_id = -1

            corpus = await self._compute_corpus()
            if corpus is None:
                # question_bank is empty -- nothing to cache, and nothing to
                # persist either. Retried on the next turn, which is correct:
                # the owner may still be loading rows.
                return None

            row = CorpusCache(data=corpus)
            self.db.add(row)
            await self.db.commit()
            await self.db.refresh(row)
            BM25Service._corpus_cache = corpus
            BM25Service._corpus_cache_id = row.id
            logger.info(
                "Recomputed the BM25 corpus from question_bank (%d documents) and persisted it as corpus_cache id=%d.",
                len(corpus["documents"]), row.id,
            )
            return corpus

        if BM25Service._corpus_cache is not None and BM25Service._corpus_cache_id == current_id:
            return BM25Service._corpus_cache

        result = await self.db.execute(
            select(CorpusCache).where(CorpusCache.id == current_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            # Deleted between the two statements above -- the owner clearing
            # it at exactly this moment. Treat it as "not cached" and let the
            # next turn recompute; refusing to guess is cheaper than a
            # half-second race window is worth.
            return BM25Service._corpus_cache

        BM25Service._corpus_cache = row.data
        BM25Service._corpus_cache_id = row.id
        logger.debug("Loaded BM25 corpus from corpus_cache id=%d.", row.id)
        return BM25Service._corpus_cache

    async def find_similar_questions(self, user_message: str, top_k: int = 3) -> list[dict]:
        """
        Ranks question_bank rows by BM25 score against a user message and returns the top matches.

        Parameters:
        - user_message (str): the text to match against — comes from the caller (e.g. ModelCollaborateService)
        - top_k (int): maximum number of results to return — defaults to 3

        Returns:
        - list[dict]: up to top_k matches as {question, answer, score}, all with score > 0 — goes to the caller (e.g. ContextGatherer.gather)

        A result must score STRICTLY ABOVE zero, i.e. "some term overlap or
        nothing". It used to be `score < 0.0`, so a document scoring exactly 0.0 -- meaning it
        shares NO stemmed term with the message, the definition of an
        irrelevant hit -- was still returned as a candidate. ContextGatherer
        then spent a Gemini call asking the model to re-rank rows that could
        not possibly match, on every turn where retrieval found nothing. An
        empty list here skips that call entirely.
        """
        corpus = await self._get_corpus()
        if corpus is None:
            return []

        query_terms = _extract_terms(user_message, self.stop_words)
        if not query_terms:
            return []
        query_term_freqs = _term_freqs(query_terms)

        documents = corpus["documents"]
        avg_doc_length = corpus["avg_doc_length"]
        idf_dict = corpus["idf_dict"]

        scored = []
        for doc in documents:
            score = _bm25_score(
                query_term_freqs=query_term_freqs,
                doc_term_freqs=doc["term_freqs"],
                doc_size=doc["size"],
                avg_doc_length=avg_doc_length,
                idf_dict=idf_dict
            )
            scored.append((doc, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)

        results = []
        for doc, score in scored[:top_k]:
            # Exclusive: a zero score means no shared terms at all. `scored` is
            # sorted descending, so the first failure ends the run.
            if score <= 0.0:
                break
            results.append({
                "question": doc["question"],
                "answer": doc["answer"],
                "score": score
            })
        return results