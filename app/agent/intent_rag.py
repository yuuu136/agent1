import hashlib
import math
import re
from dataclasses import dataclass

from app.agent.intent_catalog import intent_catalog
from app.rag.documents import KnowledgeChunk
from app.rag.vector_store import ChromaVectorStore
from app.utils.config_handler import chroma_config, rag_config


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
        self.embedding_dimension = int(settings.get("embedding_dimension", 512))
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

        query_vector = self._embed(text)
        if not query_vector:
            return None

        retrieved = self.store.search(
            query_embedding=query_vector,
            top_k=self.top_k,
            score_threshold=0.0,
        )
        if not retrieved:
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
            return None

        best_intent, (best_score, best_example) = ranked[0]
        runner_up_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
        if best_score < self.score_threshold:
            return None
        if best_score - runner_up_score < self.margin_threshold:
            return None
        return IntentMatch(
            intent=best_intent,
            score=best_score,
            example=best_example,
            runner_up_score=runner_up_score,
        )

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
        self.store.save(
            chunks=chunks,
            embeddings=[self._embed(example.text) for example in self.examples],
            metadata={
                "source_type": "intent_examples",
                "document_path": self.document_path,
                "intent_corpus_hash": self.corpus_hash,
                "example_count": len(self.examples),
                "embedding_dimension": self.embedding_dimension,
            },
        )

    def _corpus_hash(self, examples: list[IntentExample]) -> str:
        payload = "\n".join(
            f"{example.intent}:{example.text}" for example in examples
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _embed(self, text: str) -> list[float]:
        normalized = re.sub(r"[\s，。,.!?！？、:：;；]+", "", text.strip()).casefold()
        if not normalized:
            return []
        vector = [0.0] * self.embedding_dimension
        for size in (1, 2, 3):
            if len(normalized) < size:
                continue
            for index in range(0, len(normalized) - size + 1):
                token = normalized[index:index + size]
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.embedding_dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return []
        return [value / norm for value in vector]

    def _stable_id(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


intent_rag_retriever = IntentRAGRetriever()
