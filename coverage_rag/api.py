"""HTTP API.  uvicorn coverage_rag.api:app --reload"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .rag import default_assistant

app = FastAPI(title="Coverage RAG", version="0.1.0",
              description="Cited answers over Medicare NCDs and IRDAI health insurance documents.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    jurisdiction: Literal["US", "IN"] | None = Field(
        default=None, description="US = Medicare NCDs, IN = IRDAI / Arogya Sanjeevani. Omit to search both.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider, "search_backend": settings.search_backend}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return default_assistant().ask(req.question, req.jurisdiction).to_dict()
