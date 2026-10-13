from pathlib import Path

import pytest

from policy_assistant.config import Settings
from policy_assistant.pipeline import PolicyAssistant

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings() -> Settings:
    # Keyword-only search and no API key: tests run offline, fast and free.
    return Settings(use_dense=False, llm_api_key="", policies_dir=ROOT / "data" / "policies")


@pytest.fixture
def assistant(settings) -> PolicyAssistant:
    return PolicyAssistant(settings)


class FakeLLM:
    """Stands in for the real LLM, returning a fixed reply and recording what it was sent."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def generate(self, messages):
        from policy_assistant.llm import LLMResult
        self.calls.append(messages)
        if self.error:
            raise self.error
        return LLMResult(self.reply, prompt_tokens=100, completion_tokens=20)
