from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeDocument:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeChunk:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float
