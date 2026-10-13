"""Hybrid retrieval: keyword search (BM25) + semantic search (embeddings), merged with
Reciprocal Rank Fusion.

- BM25 is good at exact terms: "INR 1,500", "L5", "POSH".
- Embeddings are good at meaning: "vacation days" finds "annual leave".
- Rank fusion combines both lists without having to compare their very different scores.
"""

import logging
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

log = logging.getLogger(__name__)

_STOPWORDS = set("""
a an and are as at be by can do does for from how i if in is it its many may me much my
of on or our should so than that the their there this to was we what when where which who
why will with you your am any have has get got need does did
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, with very common words removed. "1,500" becomes "1500"."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS]


def _stem(token: str) -> str:
    """Crude plural trimming so 'leaves' matches 'leave' and 'policies' matches 'policy'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def keyword_terms(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text)]


@dataclass
class Hit:
    chunk: Chunk
    score: float                    # fused score: higher is better
    bm25_rank: int | None = None    # 1 = best keyword match
    dense_rank: int | None = None   # 1 = best semantic match
    dense_similarity: float | None = None


class DenseIndex:
    """Embeddings via fastembed (small ONNX models, runs on CPU, no GPU or PyTorch needed)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported here so keyword-only mode needs no model
        self.model = TextEmbedding(model_name=model_name)
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def build(self, texts: list[str]) -> None:
        self.vectors = self._embed(texts)

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity between the query and every chunk."""
        return self.vectors @ self._embed([query])[0]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], use_dense: bool = True,
                 embedding_model: str = "BAAI/bge-small-en-v1.5", rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self.dense: DenseIndex | None = None
        self.dense_error: str | None = None
        if use_dense:
            try:
                self.dense = DenseIndex(embedding_model)
            except Exception as exc:  # no internet to download the model, etc.
                self.dense_error = str(exc)
                log.warning("Semantic search unavailable, using keyword search only: %s", exc)
        self.rebuild(chunks)

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + embeddings)" if self.dense else "keyword (BM25 only)"

    def rebuild(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        # The title is added to each chunk's keyword text, so "travel policy" matches its chunks.
        corpus = [keyword_terms(f"{c.doc_title} {c.text}") for c in self.chunks]
        self.bm25 = BM25Okapi(corpus) if self.chunks else None
        if self.dense and self.chunks:
            self.dense.build([f"{c.doc_title}: {c.text}" for c in self.chunks])

    @property
    def idf(self) -> dict[str, float]:
        return dict(self.bm25.idf) if self.bm25 else {}

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        n = len(self.chunks)
        fused = np.zeros(n)
        bm25_rank = [None] * n
        dense_rank = [None] * n
        sims = None

        terms = keyword_terms(query)
        if terms:
            bm25_scores = self.bm25.get_scores(terms)
            for rank, i in enumerate(np.argsort(-bm25_scores), start=1):
                if bm25_scores[i] <= 0:
                    break  # no shared keywords: don't reward it just for being in the list
                bm25_rank[i] = rank
                fused[i] += 1.0 / (self.rrf_k + rank)

        if self.dense:
            sims = self.dense.similarities(query)
            for rank, i in enumerate(np.argsort(-sims), start=1):
                dense_rank[i] = rank
                fused[i] += 1.0 / (self.rrf_k + rank)

        order = [i for i in np.argsort(-fused) if fused[i] > 0][:top_k]
        return [Hit(self.chunks[i], float(fused[i]), bm25_rank[i], dense_rank[i],
                     None if sims is None else float(sims[i])) for i in order]


def has_enough_evidence(query: str, hits: list[Hit], min_keyword_coverage: float,
                        min_dense_similarity: float,
                        idf: dict[str, float] | None = None) -> tuple[bool, dict]:
    """Decide whether the retrieved chunks are relevant enough to answer from.

    Passes if enough of the question's keywords appear in the top 3 chunks, OR the best
    semantic match is similar enough. Failing both means the documents probably don't
    cover the question, so the assistant refuses instead of letting the LLM guess.

    With `idf`, rare words count more than common ones: matching "gym" matters, matching
    "company" (which appears everywhere) hardly does. Words never seen in the documents
    get the highest weight, because they are the clearest sign of an off-topic question.
    """
    terms = set(keyword_terms(query))
    top_text = set(keyword_terms(" ".join(f"{h.chunk.doc_title} {h.chunk.text}" for h in hits[:3])))
    if idf:
        unseen = max(idf.values())
        weight = {t: max(idf.get(t, unseen), 0.0) for t in terms}
    else:
        weight = {t: 1.0 for t in terms}
    total = sum(weight.values())
    coverage = sum(w for t, w in weight.items() if t in top_text) / total if total else 0.0
    best_sim = max((h.dense_similarity for h in hits if h.dense_similarity is not None),
                   default=None)
    ok = bool(hits) and (coverage >= min_keyword_coverage or
                         (best_sim is not None and best_sim >= min_dense_similarity))
    return ok, {"keyword_coverage": round(coverage, 2),
                "best_similarity": None if best_sim is None else round(best_sim, 3)}
