"""The full question-answering flow: retrieve, check evidence, generate, verify citations."""

import logging
import re
import time
from dataclasses import asdict, dataclass, field

from .config import Settings
from .ingest import Chunk, chunk_document, load_directory, read_pdf
from .llm import NOT_FOUND, ChatLLM, build_messages, cited_numbers, remove_invalid_citations
from .retrieval import Hit, HybridRetriever, has_enough_evidence, keyword_terms

log = logging.getLogger(__name__)


@dataclass
class Citation:
    number: int
    doc_title: str
    page: int
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    status: str = "answered"   # answered | not_found | unverified | error
    mode: str = ""             # "llm" or "extractive"
    retrieval: dict = field(default_factory=dict)
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _snippet(text: str, limit: int = 900) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def extractive_answer(question: str, hits: list[Hit], idf: dict[str, float] | None = None,
                      max_sentences: int = 2, min_coverage: float = 0.35) -> str:
    """No-LLM fallback: return the source sentences that best match the question.

    Used when no API key is set or the LLM call fails, so the app still gives a cited
    (if less fluent) answer instead of an error. A sentence is only used if it covers at
    least `min_coverage` of the question's words, weighted by rarity; otherwise the
    answer is "not found" rather than a loosely related quote.
    """
    terms = set(keyword_terms(question))
    if not terms:
        return NOT_FOUND
    unseen = max(idf.values()) if idf else 1.0
    weight = {t: max((idf or {}).get(t, unseen), 0.0) if idf else 1.0 for t in terms}
    total = sum(weight.values()) or 1.0

    scored = []
    for n, hit in enumerate(hits[:3], start=1):
        for position, sentence in enumerate(re.split(r"(?<=[.!?])\s+", hit.chunk.text)):
            words = set(keyword_terms(sentence))
            coverage = sum(w for t, w in weight.items() if t in words) / total
            if coverage >= min_coverage:
                scored.append((coverage, -n, -position, sentence.strip(), n))
    if not scored:
        return NOT_FOUND
    best = sorted(scored, reverse=True)[:max_sentences]
    # Keep the chosen sentences in document order so the answer reads naturally.
    best.sort(key=lambda item: (-item[1], -item[2]))
    return " ".join(f"{sentence} [{n}]" for *_, sentence, n in best)


class PolicyAssistant:
    def __init__(self, settings: Settings | None = None, chunks: list[Chunk] | None = None,
                 llm=None):
        self.settings = settings or Settings.from_env()
        s = self.settings
        if chunks is None:
            chunks = load_directory(s.policies_dir, s.chunk_words, s.chunk_overlap_words)
        self.retriever = HybridRetriever(chunks, s.use_dense, s.embedding_model)
        if llm is not None:
            self.llm = llm
        elif s.llm_api_key:
            self.llm = ChatLLM(s.llm_api_key, s.llm_base_url, s.llm_model, s.llm_temperature)
        else:
            self.llm = None
            log.warning("No LLM_API_KEY set: answers will be extractive quotes, not generated text.")

    # ----- documents -------------------------------------------------------------------
    @property
    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.retriever.chunks:
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.doc_title,
                                           "pages": 0, "chunks": 0})
            d["pages"] = max(d["pages"], c.page)
            d["chunks"] += 1
        return list(docs.values())

    def add_pdf(self, data: bytes, filename: str) -> dict:
        """Index an uploaded PDF. A file with the same name replaces the earlier version."""
        doc_id, title, pages = read_pdf(data, filename)
        if not any(p.strip() for p in pages):
            raise ValueError("No text found in this PDF. Scanned PDFs need OCR first.")
        new = chunk_document(doc_id, title, pages, self.settings.chunk_words,
                             self.settings.chunk_overlap_words)
        kept = [c for c in self.retriever.chunks if c.doc_id != doc_id]
        self.retriever.rebuild(kept + new)
        return {"doc_id": doc_id, "title": title, "pages": len(pages), "chunks": len(new)}

    # ----- answering -------------------------------------------------------------------
    def ask(self, question: str, top_k: int | None = None) -> Answer:
        started = time.perf_counter()
        question = question.strip()
        s = self.settings
        hits = self.retriever.search(question, top_k or s.top_k)
        ok, stats = has_enough_evidence(question, hits, s.min_keyword_coverage,
                                        s.min_dense_similarity,
                                        self.retriever.idf)
        result = Answer(question, NOT_FOUND, status="not_found",
                        retrieval={"mode": self.retriever.mode, **stats,
                                   "chunks": [h.chunk.chunk_id for h in hits]})

        if ok:
            text, result.mode = None, "extractive"
            if self.llm is not None:
                try:
                    out = self.llm.generate(build_messages(question, hits))
                    text, result.mode = out.text.strip(), "llm"
                    result.prompt_tokens, result.completion_tokens = (out.prompt_tokens,
                                                                      out.completion_tokens)
                except Exception as exc:
                    log.error("LLM call failed, falling back to extractive answer: %s", exc)
                    result.retrieval["llm_error"] = type(exc).__name__
            if text is None:
                text = extractive_answer(question, hits, self.retriever.idf)

            text, invalid = remove_invalid_citations(text, len(hits))
            if invalid:
                result.retrieval["invalid_citations_removed"] = invalid
            result.answer = text
            numbers = cited_numbers(text)
            result.citations = [Citation(n, hits[n - 1].chunk.doc_title, hits[n - 1].chunk.page,
                                         _snippet(hits[n - 1].chunk.text)) for n in numbers]
            if NOT_FOUND.rstrip(".") in text:
                result.status = "not_found"
            elif not numbers:
                result.status = "unverified"  # the model answered without citing anything
            else:
                result.status = "answered"

        result.latency_ms = round((time.perf_counter() - started) * 1000)
        return result
