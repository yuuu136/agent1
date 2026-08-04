from pathlib import Path

import pytest

from app.prompts import PromptConfigError, PromptManager


def test_list_configs_reads_prompt_metadata() -> None:
    manager = PromptManager(
        {
            "prompts": {
                "sample": {
                    "path": "prompts/sample.md",
                    "description": "sample prompt",
                    "required_variables": ["name"],
                }
            }
        }
    )

    configs = manager.list_configs()

    assert configs["sample"]["path"] == "prompts/sample.md"
    assert configs["sample"]["description"] == "sample prompt"
    assert configs["sample"]["required_variables"] == ["name"]


def test_render_validates_required_variables(tmp_path: Path) -> None:
    prompt_file = tmp_path / "sample.md"
    prompt_file.write_text("hello {{ name }} / {name}", encoding="utf-8")
    manager = PromptManager(
        {
            "prompts": {
                "sample": {
                    "path": str(prompt_file),
                    "required_variables": ["name"],
                }
            }
        }
    )

    assert manager.render("sample", {"name": "alice"}) == "hello alice / alice"

    with pytest.raises(PromptConfigError):
        manager.render("sample", {})


def test_project_prompt_config_can_load_rag_answer() -> None:
    manager = PromptManager()

    content = manager.get_content("rag_answer")

    assert content.strip()
