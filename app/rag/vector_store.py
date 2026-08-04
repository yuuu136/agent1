from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.rag.documents import KnowledgeChunk, RetrievedChunk
from app.utils.tool_path import get_project_abs_path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def to_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in to_jsonable(metadata).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
        else:
            normalized[str(key)] = str(value)
    return normalized


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        distance_metric: str = "cosine",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for RAG vector search. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.persist_directory = Path(get_project_abs_path(persist_directory))
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.store_path = self.persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def save(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        self._replace_collection(metadata or {})
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[to_chroma_metadata(chunk.metadata) for chunk in chunks],
            embeddings=embeddings,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        if not query_embedding or top_k <= 0 or self.collection.count() == 0:
            return []

        payload = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = payload.get("ids", [[]])[0]
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for chunk_id, content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            score = self._distance_to_score(float(distance))
            if score < score_threshold:
                continue
            results.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    content=str(content or ""),
                    metadata=dict(metadata or {}),
                    score=score,
                )
            )
        return results

    def count(self) -> int:
        return self.collection.count()

    def _replace_collection(self, metadata: dict[str, Any]) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": self.distance_metric,
                **to_chroma_metadata(metadata),
            },
        )

    def _distance_to_score(self, distance: float) -> float:
        if self.distance_metric == "cosine":
            return 1.0 - distance
        return -distance
