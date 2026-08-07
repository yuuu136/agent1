import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.utils.config_handler import rag_config
from app.utils.tool_path import get_project_abs_path


SECTION_PATTERN = re.compile(
    r"^##\s+\[(?P<kind>intent|lexicon|map)\]\s+"
    r"(?P<name>[A-Za-z][A-Za-z0-9_-]*)\s*$"
)
BULLET_PATTERN = re.compile(r"^-\s+(.+?)\s*$")


@dataclass(frozen=True)
class CatalogIntentExample:
    text: str
    intent: str


@dataclass(frozen=True)
class IntentCatalog:
    intents: dict[str, tuple[str, ...]]
    lexicons: dict[str, tuple[str, ...]]
    mappings: dict[str, dict[str, tuple[str, ...]]]

    def intent_examples(self) -> list[CatalogIntentExample]:
        return [
            CatalogIntentExample(text=text, intent=intent)
            for intent, examples in self.intents.items()
            for text in examples
        ]

    def terms(self, name: str) -> tuple[str, ...]:
        return self.lexicons.get(name, ())

    def mapping(self, name: str) -> dict[str, tuple[str, ...]]:
        return self.mappings.get(name, {})


@lru_cache(maxsize=1)
def load_intent_catalog() -> IntentCatalog:
    settings = rag_config.get("intent_rag", {})
    document_path = settings.get(
        "document_path",
        "data/intent/intent_catalog.md",
    )
    path = Path(get_project_abs_path(document_path))
    if not path.exists():
        raise FileNotFoundError(f"Intent catalog not found: {path}")

    intents: dict[str, list[str]] = {}
    lexicons: dict[str, list[str]] = {}
    mappings: dict[str, dict[str, tuple[str, ...]]] = {}
    section_kind: str | None = None
    section_name: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        heading = SECTION_PATTERN.match(raw_line.strip())
        if heading:
            section_kind = heading.group("kind")
            section_name = heading.group("name")
            if section_kind == "intent":
                intents.setdefault(section_name, [])
            elif section_kind == "lexicon":
                lexicons.setdefault(section_name, [])
            else:
                mappings.setdefault(section_name, {})
            continue

        item = BULLET_PATTERN.match(raw_line.strip())
        if not item or not section_kind or not section_name:
            continue

        value = item.group(1).strip()
        if not value:
            continue
        if section_kind == "intent":
            intents[section_name].append(value)
        elif section_kind == "lexicon":
            lexicons[section_name].append(value)
        else:
            parts = tuple(part.strip() for part in value.split("|") if part.strip())
            if len(parts) < 2:
                raise ValueError(
                    f"Map entry must contain a key and at least one phrase: {value}"
                )
            mappings[section_name][parts[0]] = parts[1:]

    if not intents:
        raise ValueError(f"Intent catalog has no [intent] sections: {path}")

    return IntentCatalog(
        intents={name: tuple(values) for name, values in intents.items()},
        lexicons={name: tuple(values) for name, values in lexicons.items()},
        mappings=mappings,
    )


intent_catalog = load_intent_catalog()
