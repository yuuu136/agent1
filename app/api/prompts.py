from typing import Any

from fastapi import APIRouter, HTTPException

from app.prompts import PromptConfigError, PromptNotFoundError, prompt_manager


router = APIRouter(prefix="/agent/prompts", tags=["prompts"])


@router.get("")
def list_prompts() -> dict[str, Any]:
    return {
        "prompts": prompt_manager.list_configs(),
    }


@router.get("/{prompt_name}")
def get_prompt(prompt_name: str) -> dict[str, Any]:
    try:
        return prompt_manager.get_prompt(prompt_name)
    except PromptNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PromptConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
