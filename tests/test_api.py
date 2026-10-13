import pytest
from fastapi.testclient import TestClient

from policy_assistant import api
from policy_assistant.pipeline import PolicyAssistant


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setitem(api.state, "assistant", PolicyAssistant(settings))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 5


def test_ask_returns_cited_answer(client):
    body = client.post("/ask", json={"question": "How many sick leave days do I get?"}).json()
    assert body["status"] == "answered"
    assert body["citations"][0]["page"] == 2


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_upload_rejects_non_pdf(client):
    r = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_upload_needs_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    r = client.post("/documents", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 401


def test_unreadable_pdf_gives_clear_error(client):
    r = client.post("/documents", files={"file": ("broken.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 422
