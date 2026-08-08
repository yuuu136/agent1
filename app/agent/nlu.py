"""NLU module — lightweight rule-based fast path for greetings/acks/cancels,
LLM-powered intent + slot extraction for everything else."""

import json
import os
import re
from typing import Any

from openai import OpenAI

from app.prompts import prompt_manager
from app.schemas.agent import AgentState, ChatRequest, NLUResult
from app.utils.config_handler import agent_config


def _normalize_short_text(text: str) -> str:
    return re.sub(r"[\s，。,.!?！？、:：;；]+", "", text.strip()).casefold()


def is_greeting_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        _normalize_short_text(value)
        for value in agent_config.get("nlu_fast_path", {}).get("greeting",
            ["你好", "您好", "嗨", "哈喽", "hi", "hello",
             "早上好", "早安", "上午好", "中午好", "下午好", "晚上好",
             "晚安", "再见", "拜拜", "开始", "start"])
    }


def is_ack_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        _normalize_short_text(value)
        for value in agent_config.get("nlu_fast_path", {}).get("ack",
            ["好", "好的", "好吧", "就好", "就行", "可以的",
             "行", "行吧", "可以", "嗯", "哦", "知道了",
             "谢谢", "谢谢你", "感谢", "多谢", "辛苦了",
             "没问题", "没事", "不用谢", "收到", "明白了", "我知道了", "很好"])
    }


CANCEL_TEXTS = {
    "取消", "取消订单", "不用了", "算了", "先不买", "不买了",
    "别买了", "不要了", "先不要了", "暂时不要了",
}


def is_cancel_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    if normalized in {_normalize_short_text(v) for v in CANCEL_TEXTS}:
        return True
    cancel_contains = ["先不支付", "暂时不支付", "不想支付", "取消支付",
                       "不付了", "不想要了", "不想买了", "不看了", "我不要了", "算了吧"]
    return any(phrase in normalized for phrase in cancel_contains)


def _build_llm_client() -> OpenAI:
    llm = agent_config.get("llm", {})
    api_key = os.getenv(llm.get("api_key_env", "DASHSCOPE_API_KEY"))
    if not api_key:
        raise RuntimeError(f"Missing API key: {llm.get('api_key_env')}")
    return OpenAI(
        api_key=api_key,
        base_url=llm.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=llm.get("timeout_seconds", 15),
    )


_NLU_PROMPT: str | None = None
_LLM_CLIENT: OpenAI | None = None


def _nlu_prompt() -> str:
    global _NLU_PROMPT
    if _NLU_PROMPT is None:
        _NLU_PROMPT = prompt_manager.get_content("nlu")
    return _NLU_PROMPT


def _llm_client() -> OpenAI:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        try:
            _LLM_CLIENT = _build_llm_client()
        except RuntimeError:
            _LLM_CLIENT = None
    return _LLM_CLIENT


class LLMNLU:
    """LLM-powered NLU with rule-based fast path for trivial intents."""

    def extract(self, request: ChatRequest, state: AgentState | None = None) -> NLUResult:
        text = request.text or ""
        payload = request.payload or {}

        # ── fast path: no LLM needed ──
        if request.event is None:
            if is_greeting_text(text):
                return NLUResult(intent="smalltalk", confidence=0.95, intent_source="rule")
            if is_ack_text(text):
                return NLUResult(intent="smalltalk", confidence=0.85, intent_source="rule")
            if is_cancel_text(text):
                return NLUResult(intent="cancel", confidence=0.95, intent_source="rule")

        # ── payload-driven intents: event-based ──
        if request.event:
            event_intent = {
                "select_movie": "select_or_modify",
                "select_cinema": "select_or_modify",
                "select_showtime": "select_showtime",
                "navigate": "select_showtime",
                "get_current_location": "location_query",
                "select_seats": "confirm_order",
                "select_snacks": "select_snacks",
                "select_coupon": "select_coupon",
                "confirm_order": "confirm_order",
                "pay_order": "pay_order",
                "get_order": "order_query",
                "refund_order": "refund_order",
                "get_refund_status": "refund_status_query",
            }.get(request.event)
            if event_intent:
                slots: dict[str, Any] = dict(payload.get("slots", {}) or {})
                for key in [
                    "showtimeId", "seatIds", "cinemaId", "movieId", "orderId",
                    "couponId", "snackIds", "snackId", "snackItems", "quantity",
                    "location", "ticketCount",
                ]:
                    if key in payload and payload[key] not in [None, ""]:
                        slots[key] = payload[key]
                return NLUResult(intent=event_intent, confidence=0.90,
                                 intent_source="rule", slots=slots)

        # ── slot-fill pre-check: when system is waiting for specific info ──
        if state and state.pending_action and request.event is None:
            slot_fill = self._try_fill_pending_slot(text, state)
            if slot_fill is not None:
                return slot_fill

        # ── LLM path ──
        return self._llm_extract(text, payload, state)

    @staticmethod
    def _try_fill_pending_slot(text: str, state: "AgentState") -> NLUResult | None:
        """When the system is actively waiting for a slot, try short-text matching
        before calling the LLM.  Only fires on short (<20 char) inputs without
        payload data — longer messages still go to the LLM."""
        if len(text) > 20:
            return None
        pending = state.pending_action

        # ── Waiting for time/date ──
        if pending == "ask_time":
            trimmed = text.strip()
            date_map = {
                "今天": "today", "今晚": "today", "today": "today",
                "明天": "tomorrow", "明晚": "tomorrow", "tomorrow": "tomorrow",
                "后天": "after_tomorrow",
            }
            date_val = date_map.get(trimmed)
            time_range_map = {
                "上午": "morning", "早上": "morning", "morning": "morning",
                "下午": "afternoon", "afternoon": "afternoon",
                "晚上": "evening", "evening": "evening", "傍晚": "evening",
            }
            time_val = time_range_map.get(trimmed)
            any_time = trimmed in ("随便", "都行", "不限", "无所谓", "什么时候都行", "都可以", "任意")
            time_match = re.match(r"^(\d{1,2})[:：点](\d{0,2})?$", trimmed)

            if any_time:
                return NLUResult(intent="book_ticket", confidence=0.92, intent_source="rule",
                                 slots={"timePreference": "any"})
            if date_val:
                return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                 slots={"date": date_val})
            if time_val:
                return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                 slots={"timeRange": time_val})
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                     slots={"timeRange": f"{hour:02d}:{minute:02d}"})

        # ── Waiting for ticket count ──
        if pending == "ask_ticket_count":
            trimmed = text.strip()
            if trimmed in ("随便", "都行", "不限", "都可以", "无所谓"):
                return NLUResult(intent="book_ticket", confidence=0.90, intent_source="rule",
                                 slots={"ticketCount": 2})
            num_match = re.match(r"^(\d+|[一二两三四五六七八九十])$", trimmed)
            if num_match:
                chinese_num = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
                val = num_match.group(1)
                count = int(val) if val.isdigit() else chinese_num.get(val, 2)
                return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                 slots={"ticketCount": count})

        # ── Waiting for movie/genre ──
        if pending == "ask_movie_or_genre":
            if text.strip() in ("随便", "都行", "不限", "都可以", "无所谓", "什么都行"):
                return NLUResult(intent="search_movies", confidence=0.88, intent_source="rule",
                                 slots={})

        return None

    def _llm_extract(self, text: str, payload: dict[str, Any],
                      state: AgentState | None = None) -> NLUResult:
        client = _llm_client()
        if client is None:
            return NLUResult(intent="smalltalk", confidence=0.30, intent_source="rule")

        nlu_settings = agent_config.get("nlu", {})
        llm_settings = agent_config.get("llm", {})

        user_message = f"用户输入：{text}"
        if payload:
            user_message += f"\n前端携带数据：{json.dumps(payload, ensure_ascii=False)}"

        context = self._build_context(state)
        if context:
            user_message += f"\n\n当前会话上下文：{context}"

        try:
            response = client.chat.completions.create(
                model=nlu_settings.get("model") or llm_settings.get("chat_model_name", "qwen-turbo"),
                temperature=nlu_settings.get("temperature", 0.5),
                max_tokens=nlu_settings.get("max_tokens", 400),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _nlu_prompt()},
                    {"role": "user", "content": user_message},
                ],
            )
            parsed = json.loads((response.choices[0].message.content or "").strip())
        except Exception:
            return NLUResult(intent="smalltalk", confidence=0.30, intent_source="rule")

        intent = str(parsed.get("intent") or "smalltalk")
        llm_slots: dict[str, Any] = parsed.get("slots") or {}
        llm_slots.update(payload.get("slots", {}) or {})
        confidence = float(parsed.get("confidence") or 0.80)

        # Normalise date aliases
        date_value = llm_slots.get("date")
        if isinstance(date_value, str):
            date_value = date_value.strip()
            if date_value in ("今天", "今晚"):  date_value = "today"
            if date_value in ("明天", "明晚"):  date_value = "tomorrow"
            if date_value in ("后天",):        date_value = "after_tomorrow"
            llm_slots["date"] = date_value

        return NLUResult(
            intent=intent,
            confidence=confidence,
            intent_source="llm",
            slots=llm_slots,
            is_modification=bool(parsed.get("is_modification")),
            reference_text=str(parsed.get("reference") or text),
        )

    @staticmethod
    def _build_context(state: AgentState | None) -> str:
        if state is None:
            return ""
        parts: list[str] = []
        if state.last_bot_message:
            # Truncate to avoid flooding the prompt
            bot_summary = state.last_bot_message[:300]
            parts.append(f"上一轮助手回复: {bot_summary}")
        if state.intent and state.intent != "smalltalk":
            parts.append(f"上一轮意图: {state.intent}")
        if state.state and state.state != "idle":
            parts.append(f"当前阶段: {state.state}")
        if state.pending_action:
            parts.append(f"待处理动作: {state.pending_action}")
        selected = state.selected or {}
        # 上一个动作搜出的候选列表（不要堆满 prompt）
        for key, label in [
            ("cinema_candidates", "已展示的影院"),
            ("movie_candidates", "已展示的影片"),
            ("showtime_candidates", "已展示的场次"),
        ]:
            items = selected.get(key) or []
            if isinstance(items, list) and items:
                names = [
                    str(item.get("cinemaName") or item.get("movieName") or item.get("name") or "")
                    for item in items[:6]
                    if isinstance(item, dict)
                ]
                names = [n for n in names if n]
                if names:
                    parts.append(f"{label}: {', '.join(names)}")
        if state.slots:
            # Only include meaningful slots (skip metadata)
            relevant = {
                k: v for k, v in state.slots.items()
                if k not in ("__clearSlots",) and v not in (None, "")
            }
            if relevant:
                parts.append(f"当前筛选条件: {json.dumps(relevant, ensure_ascii=False)}")
        return "；".join(parts)


nlu_engine = LLMNLU()
