import hashlib
import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from app.agent.intent_catalog import intent_catalog
from app.rag.documents import KnowledgeChunk
from app.rag.embeddings import QwenEmbeddingClient
from app.rag.vector_store import ChromaVectorStore
from app.utils.config_handler import chroma_config, rag_config


logger = logging.getLogger(__name__)


class IntentRAGUnavailable(RuntimeError):
    """Raised when intent RAG was required but the embedding service failed."""


@dataclass(frozen=True)
class IntentExample:
    text: str
    intent: str


@dataclass(frozen=True)
class IntentMatch:
    intent: str
    score: float
    example: str
    runner_up_score: float


class IntentRAGRetriever:
    def __init__(
        self,
        examples: list[IntentExample] | None = None,
        score_threshold: float | None = None,
        margin_threshold: float | None = None,
    ) -> None:
        settings = rag_config.get("intent_rag", {})
        document_path = settings.get(
            "document_path",
            "data/intent/intent_catalog.md",
        )
        document_examples = [
            IntentExample(text=item.text, intent=item.intent)
            for item in intent_catalog.intent_examples()
        ]
        self.examples = examples or document_examples
        if not self.examples:
            raise ValueError("Intent catalog has no intent examples")
        self.enabled = bool(settings.get("enabled", True))
        self.document_path = document_path
        shared_embedding_settings = rag_config.get("embedding", {})
        intent_embedding_settings = settings.get("embedding", {})
        self.embedding_settings = {
            **shared_embedding_settings,
            **intent_embedding_settings,
        }
        self.embedding_model_name = str(
            self.embedding_settings.get(
                "embedding_model_name",
                "text-embedding-v4",
            )
        )
        self.embedding_client = QwenEmbeddingClient(
            api_key_env=str(
                self.embedding_settings.get("api_key_env", "DASHSCOPE_API_KEY")
            ),
            base_url=str(
                self.embedding_settings.get(
                    "base_url",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            ),
            model_name=self.embedding_model_name,
            timeout_seconds=int(
                self.embedding_settings.get("timeout_seconds", 60)
            ),
        )
        self.corpus_hash = self._corpus_hash(self.examples)
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else float(settings.get("score_threshold", 0.42))
        )
        self.margin_threshold = (
            margin_threshold
            if margin_threshold is not None
            else float(settings.get("margin_threshold", 0.04))
        )
        self.top_k = int(settings.get("top_k", 8))
        self.max_retries = max(0, int(settings.get("max_retries", 1)))
        self.retry_backoff_seconds = max(
            0.0,
            float(settings.get("retry_backoff_seconds", 0.3)),
        )
        self._query_cache: dict[str, IntentMatch | None] = {}
        self._query_cache_size = max(1, int(settings.get("query_cache_size", 256)))
        self.store = ChromaVectorStore(
            persist_directory=chroma_config.get("chroma", {}).get(
                "persist_directory",
                "data/vector_store/chroma",
            ),
            collection_name=settings.get(
                "collection_name",
                "movie_ticket_agent_intents_v3",
            ),
            distance_metric=chroma_config.get("chroma", {}).get(
                "distance_metric",
                "cosine",
            ),
        )
        self._ensure_index()

    def retrieve(self, text: str) -> IntentMatch | None:
        if not self.enabled:
            return None

        cache_key = text.strip()
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        query_vector = self._embed_query_with_retry(text)
        if not query_vector:
            self._remember_query(cache_key, None)
            return None

        retrieved = self.store.search(
            query_embedding=query_vector,
            top_k=self.top_k,
            score_threshold=0.0,
        )
        if not retrieved:
            self._remember_query(cache_key, None)
            return None

        best_by_intent: dict[str, tuple[float, str]] = {}
        for item in retrieved:
            intent = str(item.metadata.get("intent") or "")
            if not intent:
                continue
            current = best_by_intent.get(intent)
            if current is None or item.score > current[0]:
                best_by_intent[intent] = (item.score, item.content)
        ranked = sorted(
            best_by_intent.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )
        if not ranked:
            self._remember_query(cache_key, None)
            return None

        best_intent, (best_score, best_example) = ranked[0]
        runner_up_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
        if best_score < self.score_threshold:
            self._remember_query(cache_key, None)
            return None
        if best_score - runner_up_score < self.margin_threshold:
            self._remember_query(cache_key, None)
            return None
        match = IntentMatch(
            intent=best_intent,
            score=best_score,
            example=best_example,
            runner_up_score=runner_up_score,
        )
        self._remember_query(cache_key, match)
        return match

    def _remember_query(
        self,
        cache_key: str,
        value: IntentMatch | None,
    ) -> None:
        if not cache_key:
            return
        self._query_cache[cache_key] = value
        while len(self._query_cache) > self._query_cache_size:
            self._query_cache.pop(next(iter(self._query_cache)))

    def _ensure_index(self) -> None:
        current_metadata = getattr(self.store.collection, "metadata", {}) or {}
        if (
            self.store.count() > 0
            and current_metadata.get("intent_corpus_hash") == self.corpus_hash
        ):
            return
        chunks = [
            KnowledgeChunk(
                id=f"intent-{index}-{self._stable_id(example.text)}",
                content=example.text,
                metadata={
                    "source": "intent_examples",
                    "document_path": self.document_path,
                    "intent": example.intent,
                    "example": example.text,
                    "index": index,
                },
            )
            for index, example in enumerate(self.examples)
        ]
        batch_size = int(self.embedding_settings.get("batch_size", 10))
        embeddings: list[list[float]] = []
        for start in range(0, len(self.examples), batch_size):
            batch = self.examples[start : start + batch_size]
            embeddings.extend(
                self._embed_texts_with_retry(
                    [example.text for example in batch]
                )
            )
        self.store.save(
            chunks=chunks,
            embeddings=embeddings,
            metadata={
                "source_type": "intent_examples",
                "document_path": self.document_path,
                "intent_corpus_hash": self.corpus_hash,
                "example_count": len(self.examples),
                "embedding_model_name": self.embedding_model_name,
            },
        )

    def _corpus_hash(self, examples: list[IntentExample]) -> str:
        payload = "\n".join(
            f"{example.intent}:{example.text}" for example in examples
        )
        payload = f"{self.embedding_model_name}\n{payload}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _stable_id(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

    def _embed_query_with_retry(self, text: str) -> list[float]:
        vectors = self._embed_texts_with_retry([text])
        return vectors[0] if vectors else []

    def _embed_texts_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.embedding_client.embed_texts(texts)
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                if attempt >= self.max_retries or status_code != 429:
                    break
                time.sleep(self.retry_backoff_seconds * (attempt + 1))
        message = str(last_exc or "unknown error")
        if getattr(last_exc, "status_code", None) == 429:
            message = f"Embedding rate limited: {message}"
        raise IntentRAGUnavailable(message) from last_exc


@lru_cache(maxsize=1)
def get_intent_rag_retriever() -> IntentRAGRetriever:
    return IntentRAGRetriever()
