from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv
from app.utils.tool_path import get_abs_path, get_project_abs_path
load_dotenv(get_project_abs_path(".env"))
def load_yaml_config(config_path: str, encoding: str = "utf-8") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding=encoding) as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")

    return config


def load_rag_config(config_path: str | None = None, encoding: str = "utf-8") -> dict[str, Any]:
    return load_yaml_config(config_path or get_abs_path("config/rag.yml"), encoding)


def load_agent_config(config_path: str | None = None, encoding: str = "utf-8") -> dict[str, Any]:
    return load_yaml_config(config_path or get_abs_path("config/agent.yml"), encoding)


def load_chroma_config(config_path: str | None = None, encoding: str = "utf-8") -> dict[str, Any]:
    return load_yaml_config(config_path or get_abs_path("config/chroma.yml"), encoding)


def load_prompts_config(config_path: str | None = None, encoding: str = "utf-8") -> dict[str, Any]:
    return load_yaml_config(config_path or get_abs_path("config/prompts.yml"), encoding)


agent_config = load_agent_config()
rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
