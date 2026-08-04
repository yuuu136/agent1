import hashlib
import re

from app.rag.documents import KnowledgeChunk, KnowledgeDocument


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_id(source: str, index: int, content: str) -> str:
    digest = hashlib.md5(f"{source}:{index}:{content}".encode("utf-8")).hexdigest()
    return f"{PathSafeId.from_source(source)}-{index}-{digest[:8]}"


class PathSafeId:
    @staticmethod
    def from_source(source: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9]+", "-", source).strip("-").lower()
        return value[-80:] or "chunk"


def split_document(
    document: KnowledgeDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[KnowledgeChunk]:
    text = _normalize_text(document.content)
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[KnowledgeChunk] = []
    source = str(document.metadata.get("source", "unknown"))
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            heading_break = text.rfind("\n#", start, end)
            split_at = max(paragraph_break, heading_break)
            if split_at > start + int(chunk_size * 0.45):
                end = split_at

        content = text[start:end].strip()
        if content:
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            metadata["chunk_start"] = start
            metadata["chunk_end"] = end
            chunks.append(
                KnowledgeChunk(
                    id=_chunk_id(source, index, content),
                    content=content,
                    metadata=metadata,
                )
            )
            index += 1

        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)

    return chunks


def split_documents(
    documents: list[KnowledgeDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(split_document(document, chunk_size, chunk_overlap))
    return chunks
