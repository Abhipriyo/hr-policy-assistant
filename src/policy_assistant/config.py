"""Settings, read from environment variables (or a .env file) so nothing secret lives in code."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=value lines, ignoring comments. Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # LLM: any OpenAI-compatible endpoint. Groq is the default because it has a free tier.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # Retrieval
    use_dense: bool = True                       # False = keyword (BM25) search only
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5                               # chunks sent to the LLM
    chunk_words: int = 60                        # target chunk size
    chunk_overlap_words: int = 25                # words shared between neighbouring chunks

    # Evidence gate: refuse instead of guessing when retrieval finds nothing relevant
    min_keyword_coverage: float = 0.25           # rarity-weighted share of question words found in top chunks
    min_dense_similarity: float = 0.62           # cosine similarity; tune with eval/run_eval.py

    policies_dir: Path = field(default_factory=lambda: Path("data/policies"))

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        s = cls()
        s.llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
        s.llm_base_url = os.getenv("LLM_BASE_URL", s.llm_base_url)
        s.llm_model = os.getenv("LLM_MODEL", s.llm_model)
        s.use_dense = _flag("USE_DENSE", s.use_dense)
        s.embedding_model = os.getenv("EMBEDDING_MODEL", s.embedding_model)
        s.top_k = int(os.getenv("TOP_K", s.top_k))
        s.min_keyword_coverage = float(os.getenv("MIN_KEYWORD_COVERAGE", s.min_keyword_coverage))
        s.min_dense_similarity = float(os.getenv("MIN_DENSE_SIMILARITY", s.min_dense_similarity))
        s.policies_dir = Path(os.getenv("POLICIES_DIR", str(s.policies_dir)))
        return s
