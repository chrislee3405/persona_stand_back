import os
import math
import string
import logging

from fastapi import Depends
from sqlalchemy.orm import Session
from stemming.porter2 import stem

from app.database import get_db
from app.models import question_bank as question_bank_models
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
    """
    raw_idf = {}
    for term, n_i in df_dict.items():
        left_denominator = (n_i + 0.5) / (ndocs - n_i + 0.5)
        raw_idf[term] = math.log(1 / left_denominator)

    positive_values = [v for v in raw_idf.values() if v > 0]
    avg_idf = sum(positive_values) / len(positive_values) if positive_values else 0.0
    floor = epsilon * avg_idf

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
    # requests. This dict persists for the lifetime of the running process,
    # shared across every request handled by it. None means "not loaded into
    # this process yet" -- distinct from an empty-but-loaded corpus.
    _corpus_cache: dict | None = None

    def __init__(self, db: Session = Depends(get_db)):
        """
        Stores the injected database session and default stop-word set.

        Parameters:
        - db (Session): SQLAlchemy session — injected by FastAPI via get_db

        Returns:
        - None: sets self.db and self.stop_words
        """
        self.db = db
        self.stop_words = _DEFAULT_STOP_WORDS

    def _load_all(self) -> list[question_bank_models.QuestionBank]:
        """
        Fetches every row from the question_bank table.

        Parameters:
        - none

        Returns:
        - list[QuestionBank]: all question_bank rows — goes to _compute_corpus
        """
        return self.db.query(question_bank_models.QuestionBank).all()

    def _compute_corpus(self) -> dict | None:
        """
        Computes the BM25 corpus (per-question term frequencies/size, plus corpus-wide stats) from every question_bank row.

        Parameters:
        - none

        Returns:
        - dict | None: {"documents": [{"question", "answer", "term_freqs", "size"}, ...], "avg_doc_length", "df_dict", "idf_dict"}, or None if question_bank is empty — goes to _get_corpus, which caches it in memory and persists it to corpus_cache
        """
        rows = self._load_all()
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

        return {
            "documents": documents,
            "avg_doc_length": avg_doc_length,
            "df_dict": df_dict,
            "idf_dict": idf_dict
        }

    def _get_corpus(self) -> dict | None:
        """
        Returns the BM25 corpus, preferring the in-memory cache, then the corpus_cache table, then computing it fresh from question_bank as a last resort.

        Parameters:
        - none

        Returns:
        - dict | None: the corpus (see _compute_corpus for shape), or None if question_bank is empty — goes to find_similar_questions
        """
        if BM25Service._corpus_cache is not None:
            return BM25Service._corpus_cache

        row = self.db.query(CorpusCache).first()
        if row is not None:
            logger.debug("Loaded BM25 corpus from corpus_cache table.")
            BM25Service._corpus_cache = row.data
            return BM25Service._corpus_cache

        corpus = self._compute_corpus()
        if corpus is None:
            return None

        BM25Service._corpus_cache = corpus
        self.db.add(CorpusCache(data=corpus))
        self.db.commit()
        logger.debug("Computed BM25 corpus from question_bank and persisted it to corpus_cache.")
        return corpus

    def find_similar_questions(self, user_message: str, top_k: int = 3, min_score: float = 0.0) -> list[dict]:
        """
        Ranks question_bank rows by BM25 score against a user message and returns the top matches.

        Parameters:
        - user_message (str): the text to match against — comes from the caller (e.g. ModelCollaborateService)
        - top_k (int): maximum number of results to return — defaults to 3
        - min_score (float): minimum score to include a result — defaults to 0.0

        Returns:
        - list[dict]: up to top_k matches as {question, answer, score} — goes to the caller (e.g. ContextGatherer.gather)
        """
        corpus = self._get_corpus()
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

        logger.debug(
            "BM25 scores for %r: %s",
            user_message,
            [(doc["question"], round(score, 4)) for doc, score in scored]
        )

        results = []
        for doc, score in scored[:top_k]:
            if score < min_score:
                break
            results.append({
                "question": doc["question"],
                "answer": doc["answer"],
                "score": score
            })
        return results