from pathlib import Path
from typing import Any

import yaml

from app.rag.documents import KnowledgeDocument
from app.utils.tool_path import get_project_abs_path


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    raw_metadata = parts[1].strip()
    content = parts[2].strip()
    metadata = yaml.safe_load(raw_metadata) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    return metadata, content


def load_text_file(path: Path) -> KnowledgeDocument:
    text = path.read_text(encoding="utf-8")
    metadata, content = _parse_frontmatter(text)
    metadata.setdefault("source", str(path))
    metadata.setdefault("file_name", path.name)
    metadata.setdefault("file_ext", path.suffix.lower())
    return KnowledgeDocument(content=content, metadata=metadata)


def load_knowledge_documents(
    knowledge_dir: str,
    allowed_extensions: list[str],
) -> list[KnowledgeDocument]:
    root = Path(get_project_abs_path(knowledge_dir))
    if not root.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {root}")

    allowed = {ext.lower() for ext in allowed_extensions}
    documents: list[KnowledgeDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        documents.append(load_text_file(path))

    return documents
