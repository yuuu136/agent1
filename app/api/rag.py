from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.rag.service import rag_service


router = APIRouter(prefix="/agent/rag", tags=["rag"])


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int | None = None


@router.post("/build")
def build_rag_index() -> dict[str, Any]:
    result = rag_service.build_index()
    return {
        "status": "ok",
        **result,
    }


@router.post("/search")
def search_knowledge(request: RAGQueryRequest) -> dict[str, Any]:
    chunks = rag_service.retrieve(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "results": [
            {
                "id": chunk.id,
                "score": chunk.score,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    }


@router.post("/answer")
def answer_with_rag(request: RAGQueryRequest) -> dict[str, Any]:
    return rag_service.answer(request.query, top_k=request.top_k)
