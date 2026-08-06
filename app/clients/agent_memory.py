from __future__ import annotations

import logging
from typing import Any

import httpx

from app.utils.config_handler import agent_config


logger = logging.getLogger(__name__)


class AgentMemoryClient:
    """Persistence client for the authenticated Spring Boot agent-memory API."""

    def __init__(self) -> None:
        spring_config = agent_config.get("spring_boot", {})
        self.base_url = str(spring_config.get("base_url") or "").rstrip("/")
        self.timeout_seconds = int(spring_config.get("timeout_seconds", 15))
        endpoints = spring_config.get("endpoints", {})
        self.current_endpoint = str(
            endpoints.get(
                "agent_memory_current",
                "/api/user/agent/memory/current",
            )
        )
        self.turn_endpoint = str(
            endpoints.get(
                "agent_memory_turn",
                "/api/user/agent/memory/turn",
            )
        )

    def current(
        self,
        *,
        jwt: str | None,
        session_id: str,
        memory_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any] | None:
        if not jwt or not self.base_url:
            return None

        params: dict[str, Any] = {
            "sessionId": session_id,
            "limit": max(1, min(limit, 100)),
        }
        if memory_id:
            params["memoryId"] = memory_id

        try:
            response = httpx.get(
                f"{self.base_url}{self.current_endpoint}",
                params=params,
                headers={"Authorization": f"Bearer {jwt}"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("加载长期记忆失败，继续使用当前进程状态: %s", exc)
            return None

        if payload.get("code") != 1:
            logger.warning(
                "加载长期记忆失败，Spring Boot 返回: code=%s msg=%s",
                payload.get("code"),
                payload.get("msg") or payload.get("message"),
            )
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None

    def save_turn(
        self,
        *,
        jwt: str | None,
        session_id: str,
        memory_id: str | None,
        user_message: str,
        assistant_message: str,
        state_json: str,
        request_json: str,
        response_json: str,
        event: str | None = None,
        intent: str | None = None,
        action: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any] | None:
        if not jwt or not self.base_url:
            return None

        body = {
            "sessionId": session_id,
            "memoryId": memory_id,
            "userMessage": user_message,
            "assistantMessage": assistant_message,
            "event": event,
            "intent": intent,
            "action": action,
            "state": state,
            "stateJson": state_json,
            "requestJson": request_json,
            "responseJson": response_json,
        }

        try:
            response = httpx.post(
                f"{self.base_url}{self.turn_endpoint}",
                json=body,
                headers={"Authorization": f"Bearer {jwt}"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("保存长期记忆失败，本轮对话不受影响: %s", exc)
            return None

        if payload.get("code") != 1:
            logger.warning(
                "保存长期记忆失败，Spring Boot 返回: code=%s msg=%s",
                payload.get("code"),
                payload.get("msg") or payload.get("message"),
            )
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None


agent_memory_client = AgentMemoryClient()
