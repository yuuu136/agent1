import hashlib
import math
import re
from dataclasses import dataclass

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


INTENT_EXAMPLES = [
    IntentExample("我想看电影", "search_movies"),
    IntentExample("想找点片子看", "search_movies"),
    IntentExample("有什么电影", "search_movies"),
    IntentExample("最近有什么上映", "search_movies"),
    IntentExample("帮我推荐几部电影", "search_movies"),
    IntentExample("今天有什么片子", "search_movies"),
    IntentExample("想看部电影", "search_movies"),
    IntentExample("想看部片子", "search_movies"),
    IntentExample("想看点片子", "search_movies"),
    IntentExample("想看一部片", "search_movies"),
    IntentExample("有没有电影看", "search_movies"),
    IntentExample("给我买两张今晚八点的电影票", "book_ticket"),
    IntentExample("帮我订一张功夫女足影票", "book_ticket"),
    IntentExample("买两张蜘蛛侠电影票", "book_ticket"),
    IntentExample("今晚去看科幻片", "book_ticket"),
    IntentExample("明晚八点订票", "book_ticket"),
    IntentExample("附近有什么影院", "nearby_cinema"),
    IntentExample("离我近的电影院", "nearby_cinema"),
    IntentExample("周边有哪些影院", "nearby_cinema"),
    IntentExample("找附近影城", "nearby_cinema"),
    IntentExample("我现在在哪里", "location_query"),
    IntentExample("我的当前位置", "location_query"),
    IntentExample("当前经纬度是多少", "location_query"),
    IntentExample("票价多少", "price_query"),
    IntentExample("这一场多少钱", "price_query"),
    IntentExample("有便宜一点的吗", "select_or_modify"),
    IntentExample("换个便宜的", "select_or_modify"),
    IntentExample("晚一点的场次", "select_or_modify"),
    IntentExample("来点爆米花", "snack"),
    IntentExample("我要加一瓶可乐", "snack"),
    IntentExample("有没有零食", "snack"),
    IntentExample("我的订单", "order_query"),
    IntentExample("查一下订单", "order_query"),
    IntentExample("支付结果怎么样", "order_query"),
    IntentExample("退票规则是什么", "faq"),
    IntentExample("退款多久能到", "faq"),
    IntentExample("改签规则", "faq"),
]


class IntentRAGRetriever:
    def __init__(
        self,
        examples: list[IntentExample] | None = None,
        score_threshold: float | None = None,
        margin_threshold: float | None = None,
    ) -> None:
        self.examples = examples or INTENT_EXAMPLES
        settings = rag_config.get("intent_rag", {})
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
                "movie_ticket_agent_intents_v2",
            ),
            distance_metric=chroma_config.get("chroma", {}).get(
                "distance_metric",
                "cosine",
            ),
        )
        self._ensure_index()

    def retrieve(self, text: str) -> IntentMatch | None:
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
        if self.store.count() > 0:
            return
        chunks = [
            KnowledgeChunk(
                id=f"intent-{index}-{self._stable_id(example.text)}",
                content=example.text,
                metadata={
                    "source": "intent_examples",
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
                "embedding_dimension": self.embedding_dimension,
            },
        )

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
