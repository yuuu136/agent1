import os
from typing import Any

from openai import OpenAI

from app.rag.documents import RetrievedChunk
from app.rag.embeddings import QwenEmbeddingClient
from app.rag.loaders import load_knowledge_documents
from app.rag.splitter import split_documents
from app.rag.vector_store import ChromaVectorStore
from app.prompts import prompt_manager
from app.utils.config_handler import agent_config, chroma_config, rag_config


class RAGService:
    def __init__(self) -> None:
        self.rag_settings = rag_config.get("rag", {})
        self.embedding_settings = rag_config.get("embedding", {})
        self.chroma_settings = chroma_config.get("chroma", {})
        self.store = ChromaVectorStore(
            persist_directory=self.chroma_settings.get(
                "persist_directory",
                "data/vector_store/chroma",
            ),
            collection_name=self.chroma_settings.get(
                "collection_name",
                "movie_ticket_agent_knowledge",
            ),
            distance_metric=self.chroma_settings.get("distance_metric", "cosine"),
        )

    def _embedding_client(self) -> QwenEmbeddingClient:
        return QwenEmbeddingClient(
            api_key_env=self.embedding_settings.get("api_key_env", "DASHSCOPE_API_KEY"),
            base_url=self.embedding_settings.get(
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model_name=self.embedding_settings.get("embedding_model_name", "text-embedding-v4"),
            timeout_seconds=self.embedding_settings.get("timeout_seconds", 60),
        )

    def build_index(self) -> dict[str, Any]:
        documents = load_knowledge_documents(
            knowledge_dir=self.rag_settings.get("knowledge_dir", "data/knowledge"),
            allowed_extensions=self.rag_settings.get("allowed_extensions", [".md", ".txt"]),
        )
        chunks = split_documents(
            documents=documents,
            chunk_size=self.rag_settings.get("chunk_size", 800),
            chunk_overlap=self.rag_settings.get("chunk_overlap", 120),
        )

        client = self._embedding_client()
        batch_size = self.embedding_settings.get("batch_size", 16)
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings.extend(client.embed_texts([chunk.content for chunk in batch]))

        self.store.save(
            chunks=chunks,
            embeddings=embeddings,
            metadata={
                "embedding_model_name": self.embedding_settings.get("embedding_model_name"),
                "knowledge_dir": self.rag_settings.get("knowledge_dir"),
                "chunk_size": self.rag_settings.get("chunk_size"),
                "chunk_overlap": self.rag_settings.get("chunk_overlap"),
            },
        )

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "vector_store_path": str(self.store.store_path),
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        client = self._embedding_client()
        query_embedding = client.embed_query(query)
        return self.store.search(
            query_embedding=query_embedding,
            top_k=top_k or self.rag_settings.get("top_k", 5),
            score_threshold=self.rag_settings.get("score_threshold", 0.0),
        )

    def answer(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        context = self._format_context(retrieved_chunks)
        prompt = self._load_prompt()
        message = self._call_chat_model(prompt=prompt, query=query, context=context)

        return {
            "message": message,
            "contexts": [self._serialize_chunk(chunk) for chunk in retrieved_chunks],
        }

    def _load_prompt(self) -> str:
        return prompt_manager.get_content("rag_answer")

    def _call_chat_model(self, prompt: str, query: str, context: str) -> str:
        llm_settings = agent_config.get("llm", {})
        api_key_env = llm_settings.get("api_key_env", "DASHSCOPE_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable: {api_key_env}")

        client = OpenAI(
            api_key=api_key,
            base_url=llm_settings.get(
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            timeout=llm_settings.get("timeout_seconds", 60),
        )
        response = client.chat.completions.create(
            model=llm_settings.get("chat_model_name", "qwen-max"),
            temperature=llm_settings.get("temperature", 0.3),
            max_tokens=llm_settings.get("max_tokens", 1200),
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"用户问题：{query}\n\nRAG 检索片段：\n{context}",
                },
            ],
        )
        return response.choices[0].message.content or ""

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "没有检索到相关知识片段。"

        parts = []
        for index, chunk in enumerate(chunks, start=1):
            title = chunk.metadata.get("title") or chunk.metadata.get("file_name") or chunk.id
            parts.append(
                f"[片段 {index}] 标题：{title}\n"
                f"来源：{chunk.metadata.get('source', '')}\n"
                f"相关度：{chunk.score:.4f}\n"
                f"内容：{chunk.content}"
            )
        return "\n\n".join(parts)

    def _serialize_chunk(self, chunk: RetrievedChunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "score": chunk.score,
            "content": chunk.content,
            "metadata": chunk.metadata,
        }


rag_service = RAGService()
if __name__ == "__main__":
    rag_service.build_index()
    print(rag_service.answer("退票"))
