"""Turn PDFs into page-aware chunks.

Every chunk remembers which document and page it came from. Chunks never cross a
page boundary, so a citation always points to exactly one page.
"""

import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # e.g. "leave_policy:p2:c1"
    doc_id: str        # file name without extension
    doc_title: str     # human-readable title used in citations
    page: int          # 1-based page number
    text: str


_BULLET = re.compile(r"^[\x00-\x1f\x7f•▪●]\s*")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _clean_line(line: str) -> str:
    """Normalise bullet glyphs (PDFs encode them in many ways) and drop control characters."""
    return _CONTROL.sub(" ", _BULLET.sub("- ", line.strip()))


def _strip_headers_and_footers(pages: list[list[str]]) -> list[list[str]]:
    """Remove lines that repeat on most pages (running headers/footers) and bare page numbers.

    Digits are masked before comparing, so "Page 1" and "Page 2" count as the same line.
    """
    if len(pages) < 2:
        return [[ln for ln in p if not _PAGE_NUMBER.match(ln)] for p in pages]
    mask = lambda ln: re.sub(r"\d+", "#", ln.strip().lower())
    counts = Counter(m for page in pages for m in {mask(ln) for ln in page if ln.strip()})
    repeated = {m for m, n in counts.items() if n >= max(2, 0.6 * len(pages))}
    return [[ln for ln in page if mask(ln) not in repeated and not _PAGE_NUMBER.match(ln)]
            for page in pages]


def read_pdf(source: str | Path | bytes, name: str | None = None) -> tuple[str, str, list[str]]:
    """Read a PDF from a path or raw bytes. Returns (doc_id, title, one text string per page)."""
    if isinstance(source, (bytes, bytearray)):
        reader = PdfReader(io.BytesIO(source))
        doc_id = Path(name or "uploaded.pdf").stem
    else:
        reader = PdfReader(str(source))
        doc_id = Path(source).stem

    raw_pages = [[_clean_line(ln) for ln in (page.extract_text() or "").splitlines()]
                 for page in reader.pages]
    pages = [" ".join(ln.strip() for ln in lines if ln.strip())
             for lines in _strip_headers_and_footers(raw_pages)]

    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    title = meta_title.strip() or doc_id.replace("_", " ").replace("-", " ").title()
    return doc_id, title, pages


def chunk_page(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Pack whole sentences into chunks of about `chunk_words` words.

    Neighbouring chunks share roughly `overlap_words` words, so a fact that sits on a
    chunk boundary still appears whole in at least one chunk.
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    chunks, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if sum(len(s.split()) for s in current) >= chunk_words:
            chunks.append(" ".join(current))
            # Carry the last sentences (up to overlap_words) into the next chunk.
            carry, words = [], 0
            for s in reversed(current):
                words += len(s.split())
                if words > overlap_words:
                    break
                carry.insert(0, s)
            current = carry
    if current and (not chunks or " ".join(current) not in chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc_id: str, title: str, pages: list[str],
                   chunk_words: int = 120, overlap_words: int = 25) -> list[Chunk]:
    chunks = []
    for page_number, text in enumerate(pages, start=1):
        for i, piece in enumerate(chunk_page(text, chunk_words, overlap_words)):
            chunks.append(Chunk(f"{doc_id}:p{page_number}:c{i}", doc_id, title, page_number, piece))
    return chunks


def load_directory(directory: str | Path, chunk_words: int = 120,
                   overlap_words: int = 25) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(directory).glob("*.pdf")):
        doc_id, title, pages = read_pdf(path)
        chunks.extend(chunk_document(doc_id, title, pages, chunk_words, overlap_words))
    return chunks
