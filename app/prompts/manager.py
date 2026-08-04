from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from app.utils.config_handler import prompts_config
from app.utils.tool_path import get_project_abs_path


class PromptConfigError(ValueError):
    pass


class PromptNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class PromptDefinition:
    name: str
    path: str
    description: str = ""
    required_variables: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["required_variables"] = list(self.required_variables)
        return data


class PromptManager:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or prompts_config)
        self._definitions = self._parse_definitions(self._config)

    def list_definitions(self) -> list[PromptDefinition]:
        return list(self._definitions.values())

    def list_configs(self) -> dict[str, dict[str, Any]]:
        return {name: definition.to_dict() for name, definition in self._definitions.items()}

    def get_definition(self, name: str) -> PromptDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise PromptNotFoundError(f"Prompt not found: {name}") from exc

    def get_content(self, name: str, encoding: str = "utf-8") -> str:
        definition = self.get_definition(name)
        path = self._resolve_path(definition.path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found for '{name}': {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Prompt path is not a file for '{name}': {path}")
        return path.read_text(encoding=encoding)

    def get_prompt(self, name: str, encoding: str = "utf-8") -> dict[str, Any]:
        definition = self.get_definition(name)
        data = definition.to_dict()
        data["content"] = self.get_content(name, encoding=encoding)
        return data

    def render(self, name: str, variables: Mapping[str, Any] | None = None) -> str:
        values = dict(variables or {})
        definition = self.get_definition(name)
        missing = [key for key in definition.required_variables if key not in values]
        if missing:
            raise PromptConfigError(
                f"Missing required variables for prompt '{name}': {', '.join(missing)}"
            )

        content = self.get_content(name)
        for key, value in values.items():
            text = str(value)
            content = re.sub(r"{{\s*" + re.escape(key) + r"\s*}}", text, content)
            content = content.replace("{" + key + "}", text)
        return content

    def _parse_definitions(self, config: Mapping[str, Any]) -> dict[str, PromptDefinition]:
        prompts = config.get("prompts", {})
        if not isinstance(prompts, Mapping):
            raise PromptConfigError("prompts.yml must contain a 'prompts' object")

        definitions: dict[str, PromptDefinition] = {}
        for name, entry in prompts.items():
            if isinstance(entry, str):
                definitions[str(name)] = PromptDefinition(name=str(name), path=entry)
                continue

            if not isinstance(entry, Mapping):
                raise PromptConfigError(f"Prompt '{name}' config must be an object or path string")

            path = entry.get("path")
            if not isinstance(path, str) or not path.strip():
                raise PromptConfigError(f"Prompt '{name}' must define a non-empty path")

            required_variables = entry.get("required_variables", [])
            if required_variables is None:
                required_variables = []
            if not isinstance(required_variables, list) or not all(
                isinstance(item, str) for item in required_variables
            ):
                raise PromptConfigError(
                    f"Prompt '{name}' required_variables must be a list of strings"
                )

            description = entry.get("description", "")
            if description is None:
                description = ""
            if not isinstance(description, str):
                raise PromptConfigError(f"Prompt '{name}' description must be a string")

            definitions[str(name)] = PromptDefinition(
                name=str(name),
                path=path,
                description=description,
                required_variables=tuple(required_variables),
            )
        return definitions

    def _resolve_path(self, prompt_path: str) -> Path:
        path = Path(prompt_path)
        if path.is_absolute():
            return path
        return Path(get_project_abs_path(prompt_path))


prompt_manager = PromptManager()
