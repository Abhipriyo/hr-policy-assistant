"""Prompting the LLM and checking its citations.

The client uses the OpenAI-compatible chat API, which Groq, OpenAI, Together, OpenRouter
and local Ollama all accept. Switching provider only means changing LLM_BASE_URL,
LLM_MODEL and LLM_API_KEY.
"""

import re
from dataclasses import dataclass

from .retrieval import Hit

NOT_FOUND = "I couldn't find this in the policy documents."

SYSTEM_PROMPT = f"""You are an HR policy assistant. Answer employees' questions using ONLY the numbered sources provided.

Rules:
- Every factual sentence must end with the number of the source that supports it, like [1] or [2][3].
- Quote numbers, limits and deadlines exactly as written in the sources.
- If the sources do not contain the answer, reply with exactly: "{NOT_FOUND}"
- If the sources answer only part of the question, answer that part and say what is not covered.
- Do not give legal advice and do not invent policy. Be concise: 1 to 4 sentences, or a short list."""


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{n}] {h.chunk.doc_title}, page {h.chunk.page}:\n{h.chunk.text}"
                       for n, h in enumerate(hits, start=1))


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_numbers(answer: str) -> list[int]:
    """Source numbers cited in the answer, in order of first appearance, without repeats."""
    seen = []
    for n in re.findall(r"\[(\d+)\]", answer):
        if int(n) not in seen:
            seen.append(int(n))
    return seen


def remove_invalid_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Strip citations to sources that don't exist (e.g. [7] when only 5 were given)."""
    invalid = [n for n in cited_numbers(answer) if not 1 <= n <= n_sources]
    for n in invalid:
        answer = answer.replace(f"[{n}]", "")
    return answer.strip(), invalid


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatLLM:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0,
                 timeout: float = 30.0):
        from openai import OpenAI
        self.model = model
        self.temperature = temperature
        # Ask for gzip, not brotli: mismatched brotli versions break response decoding on some machines.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2,
                             default_headers={"Accept-Encoding": "gzip, deflate"})

    def generate(self, messages: list[dict]) -> LLMResult:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=400)
        usage = response.usage
        return LLMResult(response.choices[0].message.content or "",
                         getattr(usage, "prompt_tokens", 0) or 0,
                         getattr(usage, "completion_tokens", 0) or 0)
