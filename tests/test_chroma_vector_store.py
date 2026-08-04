from app.rag.documents import KnowledgeChunk
from app.rag.vector_store import ChromaVectorStore


def test_chroma_vector_store_persists_and_queries_chunks(tmp_path) -> None:
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )
    chunks = [
        KnowledgeChunk(
            id="refund-policy",
            content="退票规则说明",
            metadata={"source": "refund.md", "tags": ["ticket", "refund"]},
        ),
        KnowledgeChunk(
            id="snack-policy",
            content="零食套餐说明",
            metadata={"source": "snack.md"},
        ),
    ]

    store.save(
        chunks=chunks,
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    reloaded = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )

    results = reloaded.search([0.9, 0.1, 0.0], top_k=1)

    assert reloaded.count() == 2
    assert results[0].id == "refund-policy"
    assert results[0].content == "退票规则说明"
    assert results[0].metadata["source"] == "refund.md"
