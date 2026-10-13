"""REST API, so other apps (an intranet page, a Slack bot) can use the assistant.

Run:  uvicorn policy_assistant.api:app --reload
Docs: http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .pipeline import PolicyAssistant

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "assistant" not in state:  # tests put their own assistant here first
        state["assistant"] = PolicyAssistant()
    yield


app = FastAPI(title="HR Policy Assistant API", version="1.0.0", lifespan=lifespan)


def assistant() -> PolicyAssistant:
    return state["assistant"]


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Uploads change what every user sees, so they need ADMIN_TOKEN when one is configured."""
    expected = os.getenv("ADMIN_TOKEN")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Missing or wrong X-Admin-Token header.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)


@app.get("/health")
def health(a: PolicyAssistant = Depends(assistant)):
    return {"status": "ok", "retrieval": a.retriever.mode,
            "llm": a.settings.llm_model if a.llm else "none (extractive answers)",
            "documents": len(a.documents)}


@app.get("/documents")
def list_documents(a: PolicyAssistant = Depends(assistant)):
    return a.documents


@app.post("/documents", dependencies=[Depends(require_admin)])
async def upload_document(file: UploadFile = File(...), a: PolicyAssistant = Depends(assistant)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF is larger than 10 MB.")
    try:
        return a.add_pdf(data, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read this PDF.")


@app.post("/ask")
def ask(request: AskRequest, a: PolicyAssistant = Depends(assistant)):
    return a.ask(request.question, request.top_k).to_dict()
