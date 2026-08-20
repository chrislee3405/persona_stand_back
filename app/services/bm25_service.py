import os
import math
import string
import logging

from fastapi import Depends
from sqlalchemy.orm import Session
from stemming.porter2 import stem

from app.database import get_db
from app.models import question_bank as question_bank_models

logger = logging.getLogger(__name__)

_STOP_WORDS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "resources", "common-english-words.txt"
)


def _load_stop_words(path: str = _STOP_WORDS_PATH) -> frozenset:
    """
    Loads the same way the original assignment's __main__ block did:
    comma-separated terms in one file.

    No silent fallback here on purpose: a missing or unreadable stopword
    file is a deployment bug, not something to paper over with a
    different list that would quietly change every score without anyone
    noticing. Fail at import time (app startup) instead — same principle
    as SESSION_SECRET_KEY failing loudly if unset, rather than falling
    back to something weaker.
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
    Reused verbatim from the original assignment's process_line: strips
    <p>/</p> tags, removes digits, and replaces punctuation with spaces.
    """
    line = line.replace("<p>", "").replace("</p>", "")
    line = line.translate(str.maketrans('', '', string.digits)).translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    )
    return line


def _extract_terms(text: str, stop_words: frozenset) -> list[str]:
    """
    Same per-term pipeline as the assignment's docParser/queryParser inner
    loop: process_line -> split -> lowercase -> Porter2 stem -> drop terms
    of length <= 2 or in the stop word list. Kept as one shared helper
    since the original applied the exact same steps to both documents and
    queries (queryParser is literally docParser's inner loop applied to
    one string instead of a file).
    """
    processed = process_line(text.strip())
    terms = []
    for term in processed.split():
        term = stem(term.lower())
        if len(term) > 2 and term not in stop_words:
            terms.append(term)
    return terms


def _term_freqs(terms: list[str]) -> dict[str, int]:
    freqs: dict[str, int] = {}
    for term in terms:
        freqs[term] = freqs.get(term, 0) + 1
    return freqs


def _document_frequency(term_freq_list: list[dict[str, int]]) -> dict[str, int]:
    """
    Reused from the assignment's df(): for each term, how many documents
    (not occurrences) contain it at least once.
    """
    df_dict: dict[str, int] = {}
    for term_freqs in term_freq_list:
        for term in term_freqs.keys():
            df_dict[term] = df_dict.get(term, 0) + 1
    return df_dict


def _compute_idf(df_dict: dict[str, int], ndocs: int, epsilon: float = 0.25) -> dict[str, float]:
    """
    Precomputes IDF per term, with a floor for terms whose raw IDF would
    be negative. The textbook Robertson-Sparck-Jones IDF used in the
    original assignment goes negative when a term appears in MORE than
    half the corpus — harmless in a large document collection (rare),
    but a real problem here: QuestionBank is a small table, so a common
    word appearing in just over half the stored questions is completely
    plausible, and a negative IDF term SUBTRACTS from the score instead
    of adding, actively penalizing documents for containing a query word.

    Standard fix (same approach rank_bm25 and most production BM25
    implementations use): replace any negative raw IDF with
    epsilon * average_idf_across_vocabulary, so common terms contribute
    a small positive weight instead of a penalty.
    """
    raw_idf = {}
    for term, n_i in df_dict.items():
        left_denominator = (n_i + 0.5) / (ndocs - n_i + 0.5)
        raw_idf[term] = math.log(1 / left_denominator)

    positive_values = [v for v in raw_idf.values() if v > 0]
    avg_idf = sum(positive_values) / len(positive_values) if positive_values else 0.0
    floor = epsilon * avg_idf

    return {term: (v if v > 0 else floor) for term, v in raw_idf.items()}


def _bm25_score(
    query_term_freqs: dict[str, int],
    doc_term_freqs: dict[str, int],
    doc_size: int,
    avg_doc_length: float,
    idf_dict: dict[str, float],
    k1: float = 1.2,
    k2: float = 100,
    b: float = 0.4
) -> float:
    """
    Same BM25 formula shape as the assignment's bm25() (k1/k2 defaults
    match the original), but b is lowered from the assignment's 0.75 to
    0.4. b controls how harshly document-length differences are
    normalized — 0.75 was tuned for a large newspaper-article corpus
    with genuinely long, variable-length documents. QuestionBank entries
    are all short and roughly similar length, where near-full
    normalization over-penalizes a question that's just a couple of
    words longer for no meaningful reason. k2 is left untouched: it only
    affects terms where the QUERY itself repeats a word (qf_i > 1), which
    essentially never happens with short user messages — for qf_i == 1
    the k2 term is always exactly 1.0 regardless of k2's value, so it has
    no practical effect on this use case either way.
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

    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
        self.stop_words = _DEFAULT_STOP_WORDS

    def _load_all(self) -> list[question_bank_models.QuestionBank]:
        return self.db.query(question_bank_models.QuestionBank).all()

    def find_similar_questions(
        self,
        user_message: str,
        top_k: int = 3,
        min_score: float = 0.0
    ) -> list[dict]:
        """
        Returns up to top_k QuestionBank rows ranked by BM25 score against
        user_message, each as {topic, question, answer, score}.

        min_score defaults to 0.0 rather than a guessed positive cutoff —
        BM25 scores aren't bounded to a fixed range; their scale depends on
        corpus size and term rarity. Log real scores against your actual
        question set (debug log below) before picking a real threshold.
        """
        rows = self._load_all()
        if not rows:
            return []

        doc_term_freqs_list = [
            _term_freqs(_extract_terms(row.question, self.stop_words)) for row in rows
        ]
        doc_sizes = [len(_extract_terms(row.question, self.stop_words)) for row in rows]
        avg_doc_length = sum(doc_sizes) / len(doc_sizes) if doc_sizes else 0
        df_dict = _document_frequency(doc_term_freqs_list)
        idf_dict = _compute_idf(df_dict, ndocs=len(rows))

        query_terms = _extract_terms(user_message, self.stop_words)
        if not query_terms:
            return []
        query_term_freqs = _term_freqs(query_terms)

        scored = []
        for row, doc_term_freqs, doc_size in zip(rows, doc_term_freqs_list, doc_sizes):
            score = _bm25_score(
                query_term_freqs=query_term_freqs,
                doc_term_freqs=doc_term_freqs,
                doc_size=doc_size,
                avg_doc_length=avg_doc_length,
                idf_dict=idf_dict
            )
            scored.append((row, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)

        logger.debug(
            "BM25 scores for %r: %s",
            user_message,
            [(row.topic, round(score, 4)) for row, score in scored]
        )

        results = []
        for row, score in scored[:top_k]:
            if score < min_score:
                break
            results.append({
                "topic": row.topic,
                "question": row.question,
                "answer": row.answer,
                "score": score
            })
        return results