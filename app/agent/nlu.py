"""NLU module — lightweight rule-based fast path for greetings/acks/cancels,
LLM-powered intent + slot extraction for everything else."""

import json
import logging
import os
import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from openai import OpenAI

from app.agent.genres import (
    GENRE_TERMS_PATTERN,
    canonical_genre_from_text,
    is_genre_phrase,
)
from app.agent.intent_rag import IntentMatch, IntentRAGUnavailable, get_intent_rag_retriever
from app.prompts import prompt_manager
from app.schemas.agent import AgentState, ChatRequest, NLUResult
from app.utils.config_handler import agent_config


logger = logging.getLogger(__name__)


SUPPORTED_INTENTS = {
    "book_ticket",
    "multi_movie_booking",
    "search_movies",
    "search_showtimes",
    "nearby_cinema",
    "location_query",
    "seat_query",
    "select_or_modify",
    "select_showtime",
    "price_query",
    "order_query",
    "refund_order",
    "refund_status_query",
    "pay_order",
    "confirm_order",
    "snack",
    "select_snacks",
    "skip_snacks",
    "skip_coupon",
    "cancel",
    "faq",
    "smalltalk",
}

INTENT_ALIASES = {
    "search_movie": "search_movies",
    "search_showtime": "search_showtimes",
    "refund_policy": "faq",
    "refund_status": "refund_status_query",
}


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
    cancel_contains = [
        "先不支付",
        "暂时不支付",
        "不想支付",
        "取消支付",
        "不付了",
        "不想要了",
        "不想买了",
        "先不买",
        "不买了",
        "别买了",
        "不要了",
        "不看了",
        "我不要了",
        "算了吧",
        "算了",
    ]
    return any(phrase in normalized for phrase in cancel_contains)


def is_skip_snacks_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return any(
        phrase in normalized
        for phrase in [
            "不要零食",
            "不需要零食",
            "不用零食",
            "不吃零食",
            "不加零食",
            "不买零食",
            "零食不要",
            "零食不要了",
            "不要爆米花",
            "不需要爆米花",
            "不加爆米花",
            "不买爆米花",
            "不要饮料",
            "不买饮料",
            "不要套餐",
            "不加套餐",
            "不需要小吃",
        ]
    )


def is_capability_question_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    if not any(marker in normalized for marker in ["电影票", "影票", "订票", "买票", "买电影", "订电影", "购票"]):
        return False
    return any(
        phrase in normalized
        for phrase in [
            "可以帮我买",
            "能帮我买",
            "能不能帮我买",
            "可不可以帮我买",
            "可以帮我订",
            "能帮我订",
            "能不能帮我订",
            "可不可以帮我订",
            "能否买",
            "能否订",
            "可否买",
            "可否订",
            "能不能买",
            "能不能订",
            "能不能购票",
            "可以购票",
            "能否购票",
            "会买电影票",
            "能买电影票",
            "可以买电影票",
            "能订电影票",
            "可以订电影票",
            "支持买电影票",
            "支持订电影票",
        ]
    )


def is_change_seat_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return any(word in normalized for word in ["座位", "选座", "位置", "座"]) and any(
        word in normalized
        for word in ["换", "改", "重新", "重选", "不要这个", "不要当前", "不想要这个"]
    )


def _is_pay_order_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        "支付",
        "付款",
        "去支付",
        "去付款",
        "我要支付",
        "我要付款",
        "现在支付",
        "现在付款",
        "确认支付",
        "确认付款",
    }


def _is_refund_order_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        "退款",
        "退票",
        "我要退款",
        "我要退票",
        "申请退款",
        "申请退票",
        "帮我退款",
        "帮我退票",
    }


def _is_refund_status_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        "退款进度",
        "退款状态",
        "退款到哪了",
        "退款成功了吗",
        "退款到账了吗",
        "退票进度",
        "退票状态",
    }


def _is_order_query_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        "订单",
        "我的订单",
        "订单详情",
        "订单信息",
        "订单状态",
        "查看订单",
        "查询订单",
        "看看订单",
    }


_AUTO_PURCHASE_TEXTS: set[str] = {
    "随便买一张",
    "随便订一张",
    "随机买一张",
    "随机订一张",
    "给我随便买一张",
    "给我随便订一张",
    "帮我随便买一张",
    "帮我随便订一张",
}

_AUTO_PURCHASE_PREFIXES = ("那就", "再", "还", "也", "那")

# Core pattern: optional prefix ("给我"/"帮我") + "随便"/"随机" + "买"/"订" + "一张"
_AUTO_PURCHASE_RE = re.compile(
    r"^(给我|帮我)?(随便|随机)(买|订)"
    r"(?P<count>\d+|[一二两俩三四五六七八九十]+)张"
    r"(?:电影票|影票|票)?$"
)


def _strip_auto_purchase_prefix(normalized: str) -> str:
    """Remove a single leading adverbial prefix so "再随便买一张" matches like "随便买一张"."""
    for prefix in _AUTO_PURCHASE_PREFIXES:
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return normalized[len(prefix):]
    return normalized


def _parse_auto_purchase_count(value: str) -> int | None:
    if value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    if value in {"两", "俩"}:
        return 2
    values = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if value in values:
        return values[value]
    if value.startswith("十") and len(value) == 2:
        tail = values.get(value[1])
        return 10 + tail if tail is not None else None
    if value.endswith("十") and len(value) == 2:
        head = values.get(value[0])
        return head * 10 if head is not None else None
    if "十" in value and len(value) == 3:
        head, tail = value.split("十", 1)
        head_value = values.get(head)
        tail_value = values.get(tail)
        if head_value is not None and tail_value is not None:
            return head_value * 10 + tail_value
    return None


def _auto_purchase_ticket_count(text: str) -> int | None:
    normalized = _normalize_short_text(text)
    if normalized in _AUTO_PURCHASE_TEXTS:
        return 1
    stripped = _strip_auto_purchase_prefix(normalized)
    if stripped in _AUTO_PURCHASE_TEXTS:
        return 1
    match = _AUTO_PURCHASE_RE.match(stripped)
    if not match:
        return None
    return _parse_auto_purchase_count(match.group("count"))


def is_single_ticket_auto_purchase_request(text: str) -> bool:
    """Match an explicit one-ticket autopurchase command without an LLM hop."""
    return _auto_purchase_ticket_count(text) == 1


def _auto_seat_preference(text: str) -> str:
    normalized = _normalize_short_text(text)
    # "再随便买一张" may not contain seat keywords but should still map to random.
    if any(word in normalized for word in ["随便", "随机", "任意"]):
        return "random"
    if any(word in normalized for word in ["前排", "靠前", "前面"]):
        return "front"
    if any(word in normalized for word in ["后排", "靠后", "后面"]):
        return "back"
    if any(word in normalized for word in ["最佳观影区", "黄金观影区", "最佳观影", "黄金位置", "好位置"]):
        return "best"
    return "middle"


def is_text_auto_seat_or_auto_purchase(text: str) -> bool:
    """True when the text signals auto (random) seat selection, either through
    explicit seat keywords or through an auto-purchase variant like "再随便买一张"."""
    if _auto_purchase_ticket_count(text) is not None:
        return True
    return is_auto_seat_request(text)


def is_direct_current_showtime_confirmation(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        "确认",
        "确认这场",
        "确认当前选择",
        "就这场",
        "选这场",
        "帮我直接买",
        "帮我直接买了",
        "直接买",
        "直接买了",
        "买这场",
        "订这场",
    }


def is_auto_seat_request(text: str) -> bool:
    normalized = _normalize_short_text(text)
    if not any(word in normalized for word in ["座位", "选座", "位置", "座", "观影区", "观影"]):
        return False
    return any(
        phrase in normalized
        for phrase in [
            "随便选",
            "你帮我选",
            "帮我选",
            "自动选",
            "推荐个",
            "推荐一个",
            "选个好位置",
            "选个位置",
            "选好位置",
            "最佳观影区",
            "黄金观影区",
            "最佳观影",
            "黄金位置",
            "好位置",
            "中间位置",
            "中间的",
            "靠中间",
            "前排",
            "靠前",
            "后排",
            "靠后",
            "都行",
            "无所谓",
            "随机",
            "随机座位",
            "座位随机",
        ]
    )


def _showtime_candidate_context(state: AgentState | None) -> dict[str, Any]:
    if state is None:
        return {}
    keys = [
        "showtimeId",
        "movieId",
        "movieName",
        "cinemaId",
        "cinemaName",
        "hallName",
        "hallType",
        "language",
        "date",
        "time",
        "startAt",
        "endAt",
        "price",
        "remainingSeats",
    ]
    if state.slots.get("showtimeId") not in [None, ""]:
        return {
            key: state.slots[key]
            for key in keys
            if state.slots.get(key) not in [None, ""]
        }

    candidates = state.selected.get("showtime_candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        return {}
    showtime = candidates[0]
    if not isinstance(showtime, dict) or showtime.get("showtimeId") in [None, ""]:
        return {}
    return {
        key: showtime[key]
        for key in keys
        if showtime.get(key) not in [None, ""]
    }


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
            auto_purchase_count = _auto_purchase_ticket_count(text)
            if auto_purchase_count is not None:
                slots = {
                    "ticketCount": auto_purchase_count,
                    "autoSelectShowtime": True,
                    "autoSelectSeats": True,
                    "seatPreference": "random",
                    "skipSnacks": True,
                    "timePreference": "earliest",
                }
                if (
                    state
                    and state.state == "selecting_seats"
                    and state.pending_action == "get_seats"
                    and state.slots.get("showtimeId") not in [None, ""]
                ):
                    return NLUResult(
                        intent="seat_query",
                        confidence=0.99,
                        intent_source="rule",
                        slots=slots,
                        reference_text=text,
                    )
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.99,
                    intent_source="rule",
                    slots=slots,
                    reference_text=text,
                )
            if (
                state
                and is_direct_current_showtime_confirmation(text)
                and _showtime_candidate_context(state)
                and not state.slots.get("orderId")
            ):
                slots = _showtime_candidate_context(state)
                if any(word in _normalize_short_text(text) for word in ["买", "订", "直接"]):
                    slots["autoSelectSeats"] = True
                    slots["seatPreference"] = state.slots.get("seatPreference", "middle")
                    if state.slots.get("seatType") not in [None, ""]:
                        slots["seatType"] = state.slots["seatType"]
                    slots["skipSnacks"] = True
                return NLUResult(
                    intent="select_showtime",
                    confidence=0.99,
                    intent_source="rule",
                    slots=slots,
                    reference_text=text,
                )
            recommendation_slots = self._extract_recommendation_slots(text)
            if (
                recommendation_slots
                and (
                    self._extract_recommendation_movie_limit(text) is not None
                    or recommendation_slots.get("genre") not in [None, ""]
                )
            ):
                return NLUResult(
                    intent="search_movies",
                    confidence=0.98,
                    intent_source="rule",
                    slots=recommendation_slots,
                    reference_text=text,
                )
            if self._is_deterministic_booking_text(text):
                slots = self._extract_booking_slots(text, payload)
                movie_names = self._extract_multi_movie_names(text)
                if len(movie_names) > 1:
                    slots["movieNames"] = movie_names
                    slots.pop("movieName", None)
                    return NLUResult(
                        intent="multi_movie_booking",
                        confidence=0.96,
                        intent_source="rule",
                        slots=slots,
                        reference_text=text,
                    )
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.96,
                    intent_source="rule",
                    slots=slots,
                    reference_text=text,
                )
            no_showtime_followup_slots = self._no_showtime_followup_slots(text, state)
            if no_showtime_followup_slots is not None:
                return NLUResult(
                    intent="search_showtimes",
                    confidence=0.99,
                    intent_source="rule",
                    slots=no_showtime_followup_slots,
                    reference_text=text,
                )
            hall_type_showtime_slots = self._extract_hall_type_showtime_slots(text)
            if hall_type_showtime_slots:
                return NLUResult(
                    intent="search_showtimes",
                    confidence=0.98,
                    intent_source="rule",
                    slots=hall_type_showtime_slots,
                    reference_text=text,
                )
            if self._is_all_seats_request(text):
                return NLUResult(
                    intent="seat_query",
                    confidence=0.98,
                    intent_source="rule",
                    slots=self._all_seats_request_slots(state),
                    reference_text=text,
                )
            if _is_pay_order_text(text):
                return NLUResult(
                    intent="pay_order",
                    confidence=0.98,
                    intent_source="rule",
                    reference_text=text,
                )
            if _is_refund_status_text(text):
                return NLUResult(
                    intent="refund_status_query",
                    confidence=0.98,
                    intent_source="rule",
                    reference_text=text,
                )
            if _is_refund_order_text(text):
                return NLUResult(
                    intent="refund_order",
                    confidence=0.98,
                    intent_source="rule",
                    reference_text=text,
                )
            if _is_order_query_text(text):
                return NLUResult(
                    intent="order_query",
                    confidence=0.98,
                    intent_source="rule",
                    reference_text=text,
                )
            if state and self._is_other_cinema_showtime_followup(text, state):
                return NLUResult(
                    intent="search_showtimes",
                    confidence=0.98,
                    intent_source="rule",
                    slots=self._other_cinema_showtime_slots(state),
                    reference_text=text,
                )
            seat_action_slots = self._extract_seat_action_slots(text, state)
            if seat_action_slots is not None:
                return NLUResult(
                    intent="seat_query",
                    confidence=0.98,
                    intent_source="rule",
                    slots=seat_action_slots,
                    reference_text=text,
                )
            if (
                state
                and state.state == "selecting_seats"
                and state.pending_action == "get_seats"
                and is_text_auto_seat_or_auto_purchase(text)
            ):
                return NLUResult(
                    intent="seat_query",
                    confidence=0.96,
                    intent_source="rule",
                    slots={
                        "autoSelectSeats": True,
                        "seatPreference": _auto_seat_preference(text),
                    },
                    reference_text=text,
                )
            if (
                state
                and state.state == "selecting_seats"
                and state.pending_action == "get_seats"
            ):
                seat_slots = self._extract_seat_preference_slots(text)
                if seat_slots:
                    return NLUResult(
                        intent="seat_query",
                        confidence=0.98,
                        intent_source="rule",
                        slots=seat_slots,
                        reference_text=text,
                    )
            if self._is_movie_knowledge_question(text):
                return NLUResult(
                    intent="smalltalk",
                    confidence=0.90,
                    intent_source="llm",
                    reference_text=text,
                )
            contextual_ack = self._try_contextual_ack(text, state)
            if contextual_ack is not None:
                return contextual_ack
            hybrid_rule = self._hybrid_rule_candidate(text, payload, state)
            if hybrid_rule is not None:
                return self._resolve_hybrid_intent(
                    text,
                    payload,
                    state,
                    rule_candidate=hybrid_rule,
                )
            # Seat selection is a stateful action. Resolve natural-language
            # preferences before generic RAG/LLM routing so follow-ups cannot
            # fall back to asking for movie/time again.
            if (
                state
                and state.state == "selecting_seats"
                and state.pending_action == "get_seats"
                and is_text_auto_seat_or_auto_purchase(text)
            ):
                return NLUResult(
                    intent="seat_query",
                    confidence=0.96,
                    intent_source="rule",
                    slots={
                        "autoSelectSeats": True,
                        "seatPreference": _auto_seat_preference(text),
                    },
                    reference_text=text,
                )
            if is_greeting_text(text):
                return NLUResult(intent="smalltalk", confidence=0.95, intent_source="rule")
            if is_ack_text(text):
                return NLUResult(intent="smalltalk", confidence=0.85, intent_source="rule")
            if is_capability_question_text(text):
                return NLUResult(
                    intent="smalltalk",
                    confidence=0.95,
                    intent_source="rule",
                    reference_text=text,
                )
            if is_cancel_text(text):
                return NLUResult(intent="cancel", confidence=0.95, intent_source="rule")
            if is_skip_snacks_text(text):
                return NLUResult(
                    intent="skip_snacks",
                    confidence=0.95,
                    intent_source="rule",
                    reference_text=text,
                )
            if is_change_seat_text(text):
                return NLUResult(
                    intent="seat_query",
                    confidence=0.95,
                    intent_source="rule",
                    slots=self._extract_seat_action_slots(text, state) or {},
                    reference_text=text,
                )
            if self._is_price_query_text(text):
                return NLUResult(intent="price_query", confidence=0.95, intent_source="rule")
            if self._is_nearby_cinema_text(text):
                slots = dict(payload.get("slots", {}) or {})
                if payload.get("location") not in [None, ""]:
                    slots["location"] = payload.get("location")
                if payload.get("city") not in [None, ""]:
                    slots["city"] = payload.get("city")
                hall_type = self._extract_hall_type(text)
                if hall_type:
                    slots["hallType"] = hall_type
                return NLUResult(
                    intent="nearby_cinema",
                    confidence=0.95,
                    intent_source="rule",
                    slots=slots,
                )
            if self._is_location_query_text(text):
                slots = {}
                if payload.get("location") not in [None, ""]:
                    slots["location"] = payload.get("location")
                return NLUResult(
                    intent="location_query",
                    confidence=0.95,
                    intent_source="rule",
                    slots=slots,
                    reference_text=text,
                )
            if self._is_movie_knowledge_question(text):
                return NLUResult(
                    intent="smalltalk",
                    confidence=0.90,
                    intent_source="llm",
                    reference_text=text,
                )
            actor = self._extract_actor_movie_query(text)
            if actor:
                return NLUResult(
                    intent="search_movies",
                    confidence=0.94,
                    intent_source="rule",
                    slots={"actor": actor},
                    reference_text=text,
                )
            recommendation_slots = self._extract_recommendation_slots(text)
            if recommendation_slots:
                return NLUResult(
                    intent="search_movies",
                    confidence=0.95,
                    intent_source="rule",
                    slots=recommendation_slots,
                    reference_text=text,
                )
            if self._is_showtime_search_text(text):
                slots = self._extract_booking_slots(text, payload)
                self._apply_nearby_showtime_preferences(text, slots)
                movie_name = self._extract_showtime_query_movie_name(
                    text,
                    str(slots.get("cinemaName") or ""),
                )
                if movie_name:
                    slots["movieName"] = movie_name
                elif self._is_invalid_showtime_movie_name(
                    slots.get("movieName")
                ):
                    slots.pop("movieName", None)
                if self._should_clear_stale_showtime_constraints(text):
                    slots["__clearSlots"] = [
                        "date",
                        "time",
                        "timeRange",
                        "timePreference",
                        "ticketCount",
                        "showtimeId",
                        "seatIds",
                        "seatPositions",
                        "orderId",
                        "lockId",
                        "couponId",
                        "snackIds",
                        "snackItems",
                        "snackRequests",
                        "price",
                        "amount",
                        "status",
                        "expiresAt",
                    ]
                return NLUResult(
                    intent="search_showtimes",
                    confidence=0.95,
                    intent_source="rule",
                    slots=slots,
                    reference_text=text,
                )
            # "我要看/想看 + movie_name" → search_showtimes
            # (placed after _is_showtime_search_text so "场次" etc. take priority)
            if self._is_movie_view_request(text):
                movie_name = self._extract_movie_name(text)
                if movie_name and not is_genre_phrase(movie_name):
                    slots = self._extract_booking_slots(text, payload)
                    slots["movieName"] = movie_name
                    self._apply_nearby_showtime_preferences(text, slots)
                    return NLUResult(
                        intent="search_showtimes",
                        confidence=0.92,
                        intent_source="rule",
                        slots=slots,
                        reference_text=text,
                    )
            if self._is_booking_request_text(text):
                slots = self._extract_booking_slots(text, payload)
                movie_names = self._extract_multi_movie_names(text)
                if len(movie_names) > 1:
                    slots["movieNames"] = movie_names
                    slots.pop("movieName", None)
                    return NLUResult(
                        intent="multi_movie_booking",
                        confidence=0.95,
                        intent_source="rule",
                        slots=slots,
                    )
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.92,
                    intent_source="rule",
                    slots=slots,
                )
            snack_requests = self._extract_snack_requests(text)
            if snack_requests:
                return NLUResult(
                    intent="snack",
                    confidence=0.92,
                    intent_source="rule",
                    slots={"snackRequests": snack_requests},
                    reference_text=text,
                )

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
                "skip_snacks": "skip_snacks",
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
                    "location", "ticketCount", "movieName", "cinemaName",
                    "hallName", "hallType", "language", "date", "time",
                    "startAt", "endAt", "endTime", "price", "remainingSeats",
                    "seatPreference", "seatType", "seatPositions", "snackRequests",
                ]:
                    if key in payload and payload[key] not in [None, ""]:
                        slots[key] = payload[key]
                if (
                    request.event == "select_seats"
                    and "ticketCount" not in slots
                    and not (state and state.slots.get("ticketCount"))
                    and isinstance(slots.get("seatIds"), list)
                    and slots["seatIds"]
                ):
                    slots["ticketCount"] = len(slots["seatIds"])
                return NLUResult(intent=event_intent, confidence=0.90,
                                 intent_source="rule", slots=slots)

        # ── slot-fill pre-check: when system is waiting for specific info ──
        if state and state.pending_action and request.event is None:
            slot_fill = self._try_fill_pending_slot(text, state)
            if slot_fill is not None:
                return slot_fill

        # ── Intent RAG + LLM arbitration ──
        return self._resolve_hybrid_intent(text, payload, state)

    def _hybrid_rule_candidate(
        self,
        text: str,
        payload: dict[str, Any],
        state: AgentState | None,
    ) -> NLUResult | None:
        """Return a candidate for ambiguous natural language, not a final decision."""
        if state and state.pending_action in {
            "ask_time",
            "ask_ticket_count",
            "ask_movie_or_genre",
        }:
            return None
        if is_capability_question_text(text):
            return None
        if is_ack_text(text):
            return NLUResult(
                intent="smalltalk",
                confidence=0.65,
                intent_source="rule",
                reference_text=text,
            )
        actor = self._extract_actor_movie_query(text)
        if actor:
            return NLUResult(
                intent="search_movies",
                confidence=0.80,
                intent_source="rule",
                slots={"actor": actor},
                reference_text=text,
            )
        recommendation_slots = self._extract_recommendation_slots(text)
        if recommendation_slots:
            return NLUResult(
                intent="search_movies",
                confidence=0.82,
                intent_source="rule",
                slots=recommendation_slots,
                reference_text=text,
            )
        if self._is_showtime_search_text(text):
            slots = self._extract_booking_slots(text, payload)
            self._apply_nearby_showtime_preferences(text, slots)
            movie_name = self._extract_showtime_query_movie_name(
                text,
                str(slots.get("cinemaName") or ""),
            )
            if movie_name:
                slots["movieName"] = movie_name
            elif self._is_invalid_showtime_movie_name(slots.get("movieName")):
                slots.pop("movieName", None)
            if self._should_clear_stale_showtime_constraints(text):
                slots["__clearSlots"] = [
                    "date",
                    "time",
                    "timeRange",
                    "timePreference",
                    "ticketCount",
                    "showtimeId",
                    "seatIds",
                    "seatPositions",
                    "orderId",
                    "lockId",
                    "couponId",
                    "snackIds",
                    "snackItems",
                    "snackRequests",
                    "price",
                    "amount",
                    "status",
                    "expiresAt",
                ]
            return NLUResult(
                intent="search_showtimes",
                confidence=0.84,
                intent_source="rule",
                slots=slots,
                reference_text=text,
            )
        # "我要看/想看 + movie_name" → search_showtimes (hybrid candidate)
        # (placed after _is_showtime_search_text so "场次" etc. take priority)
        if self._is_movie_view_request(text):
            movie_name = self._extract_movie_name(text)
            if movie_name and not is_genre_phrase(movie_name):
                slots = self._extract_booking_slots(text, payload)
                slots["movieName"] = movie_name
                self._apply_nearby_showtime_preferences(text, slots)
                return NLUResult(
                    intent="search_showtimes",
                    confidence=0.82,
                    intent_source="rule",
                    slots=slots,
                    reference_text=text,
                )
        if self._is_booking_request_text(text):
            slots = self._extract_booking_slots(text, payload)
            movie_names = self._extract_multi_movie_names(text)
            if len(movie_names) > 1:
                slots["movieNames"] = movie_names
                slots.pop("movieName", None)
                return NLUResult(
                    intent="multi_movie_booking",
                    confidence=0.86,
                    intent_source="rule",
                    slots=slots,
                    reference_text=text,
                )
            return NLUResult(
                intent="book_ticket",
                confidence=0.84,
                intent_source="rule",
                slots=slots,
                reference_text=text,
            )
        snack_requests = self._extract_snack_requests(text)
        if snack_requests:
            return NLUResult(
                intent="snack",
                confidence=0.82,
                intent_source="rule",
                slots={"snackRequests": snack_requests},
                reference_text=text,
            )
        return None

    def _resolve_hybrid_intent(
        self,
        text: str,
        payload: dict[str, Any],
        state: AgentState | None,
        rule_candidate: NLUResult | None = None,
    ) -> NLUResult:
        rag_candidates: list[IntentMatch] = []
        rag_match: IntentMatch | None = None
        rag_error: str | None = None
        if text.strip():
            try:
                retriever = get_intent_rag_retriever()
                if hasattr(retriever, "retrieve_candidates"):
                    rag_candidates = retriever.retrieve_candidates(text, limit=3)
                    if hasattr(retriever, "select_match"):
                        rag_match = retriever.select_match(rag_candidates)
                    else:
                        rag_match = rag_candidates[0] if rag_candidates else None
                else:
                    # Compatibility for lightweight test doubles and older
                    # retrievers during a rolling deployment.
                    rag_match = retriever.retrieve(text)
                    if rag_match is not None:
                        rag_candidates = [rag_match]
            except IntentRAGUnavailable as exc:
                rag_error = str(exc)
                logger.warning("Intent RAG unavailable for NLU: %s", exc)
            except Exception as exc:
                rag_error = str(exc)
                logger.exception("Intent RAG failed unexpectedly")

        return self._llm_extract(
            text,
            payload,
            state,
            rag_match,
            rag_error,
            rule_candidate=rule_candidate,
            rag_candidates=rag_candidates,
        )

    def _try_fill_pending_slot(self, text: str, state: "AgentState") -> NLUResult | None:
        """When the system is actively waiting for a slot, try short-text matching
        before calling the LLM.  Only fires on short (<20 char) inputs without
        payload data — longer messages still go to the LLM."""
        if len(text) > 20:
            return None
        pending = state.pending_action
        collecting_slot = (
            state.state.removeprefix("collecting_")
            if state.state.startswith("collecting_")
            else ""
        )

        # ── Waiting for time/date ──
        if pending == "ask_time" or collecting_slot in {"date", "timeRange"}:
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
            extracted_time = self._extract_time_range(trimmed)

            if any_time:
                return NLUResult(intent="book_ticket", confidence=0.92, intent_source="rule",
                                 slots={"timePreference": "any"})
            if date_val:
                return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                 slots={"date": date_val})
            if time_val:
                return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                 slots={"timeRange": time_val})
            if extracted_time:
                return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                 slots={"timeRange": extracted_time})
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                     slots={"timeRange": f"{hour:02d}:{minute:02d}"})

        # ── Waiting for ticket count ──
        if pending == "ask_ticket_count" or collecting_slot == "ticketCount":
            trimmed = text.strip()
            if trimmed in ("随便", "都行", "不限", "都可以", "无所谓"):
                return NLUResult(intent="book_ticket", confidence=0.90, intent_source="rule")
            count = self._extract_ticket_count(trimmed)
            if count is None:
                num_match = re.match(
                    r"^(\d+|[一二两三四五六七八九十])(?:张|个人|人)?$",
                    trimmed,
                )
                if num_match:
                    val = num_match.group(1)
                    count = int(val) if val.isdigit() else self._parse_number_token(val)
            if count is not None:
                return NLUResult(intent="book_ticket", confidence=0.95, intent_source="rule",
                                 slots={"ticketCount": count})

        if collecting_slot == "cinemaName":
            normalized = _normalize_short_text(text)
            if any(word in normalized for word in ["附近", "最近", "离我近", "就近"]):
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.93,
                    intent_source="rule",
                    slots={"nearbyFirst": True, "cinemaLimit": 5},
                    reference_text=text,
                )
            cinema_name = self._extract_cinema_name(text) or text.strip()
            if cinema_name and cinema_name not in {"随便", "都行", "不限", "都可以", "无所谓"}:
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.93,
                    intent_source="rule",
                    slots={"cinemaName": cinema_name},
                    reference_text=text,
                )

        if collecting_slot == "seatPreference":
            slots = self._extract_seat_preference_slots(text)
            seat_type = self._extract_seat_type(text)
            if seat_type:
                slots["seatType"] = seat_type
                slots.setdefault("autoSelectSeats", True)
                slots.setdefault("seatPreference", "middle")
            if not slots and text.strip() in ("随便", "都行", "不限", "都可以", "无所谓"):
                slots = {"autoSelectSeats": True, "seatPreference": "middle"}
            if slots:
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.93,
                    intent_source="rule",
                    slots=slots,
                    reference_text=text,
                )

        # ── Waiting for movie/genre ──
        if pending == "ask_movie_or_genre" or collecting_slot in {"movieName", "genre"}:
            if text.strip() in ("随便", "都行", "不限", "都可以", "无所谓", "什么都行"):
                return NLUResult(intent="search_movies", confidence=0.88, intent_source="rule",
                                 slots={})
            genre = self._extract_genre(text)
            if genre and collecting_slot == "genre":
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.94,
                    intent_source="rule",
                    slots={"genre": genre},
                    reference_text=text,
                )
            movie_name = self._extract_movie_name(text)
            if collecting_slot == "movieName" and movie_name:
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.94,
                    intent_source="rule",
                    slots={"movieName": movie_name},
                    reference_text=text,
                )

        return None

    def _try_contextual_ack(
        self,
        text: str,
        state: AgentState | None,
    ) -> NLUResult | None:
        if not state or not is_ack_text(text):
            return None
        last_message = _normalize_short_text(state.last_bot_message)
        if not last_message:
            return None
        if (
            "要不要看看现在有哪些电影正在上映" in last_message
            or "看看现在有哪些电影正在上映" in last_message
            or "现在有哪些电影正在上映" in last_message
        ):
            return NLUResult(
                intent="search_movies",
                confidence=0.96,
                intent_source="rule",
                slots={
                    "date": "today",
                    "recommendationCriteria": "hot",
                    "movieLimit": 3,
                    "__clearSlots": [
                        "genre",
                        "movieName",
                        "movieId",
                        "recommendationCriteria",
                        "showtimeId",
                        "seatIds",
                        "seatPositions",
                    ],
                },
                reference_text=text,
            )
        return None

    def _llm_extract(
        self,
        text: str,
        payload: dict[str, Any],
        state: AgentState | None = None,
        rag_match: IntentMatch | None = None,
        rag_error: str | None = None,
        rule_candidate: NLUResult | None = None,
        rag_candidates: list[IntentMatch] | None = None,
    ) -> NLUResult:
        client = _llm_client()
        if client is None:
            return self._fallback_nlu_result(
                text,
                payload,
                rule_candidate,
                rag_match,
                nlu_error="LLM client unavailable",
            )

        nlu_settings = agent_config.get("nlu", {})
        llm_settings = agent_config.get("llm", {})

        user_message = f"用户输入：{text}"
        if payload:
            user_message += f"\n前端携带数据：{json.dumps(payload, ensure_ascii=False)}"

        context = self._build_context(state)
        if context:
            user_message += f"\n\n当前会话上下文：{context}"
        if rule_candidate is not None:
            user_message += (
                "\n\n规则候选（仅供参考，不能直接决定最终意图）："
                f"intent={rule_candidate.intent}, "
                f"confidence={rule_candidate.confidence:.3f}, "
                f"slots={json.dumps(rule_candidate.slots, ensure_ascii=False)}"
            )
        if rag_candidates:
            candidates_text = [
                {
                    "intent": item.intent,
                    "score": round(item.score, 4),
                    "example": item.example,
                }
                for item in rag_candidates[:3]
            ]
            user_message += (
                "\n\n意图RAG候选（仅供参考，不能直接决定最终意图）："
                f"{json.dumps(candidates_text, ensure_ascii=False)}"
            )
        elif rag_error:
            user_message += f"\n\n意图RAG状态：检索失败，原因：{rag_error}"

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
        except Exception as exc:
            logger.warning("LLM NLU unavailable: %s", exc)
            return self._fallback_nlu_result(
                text,
                payload,
                rule_candidate,
                rag_match,
                nlu_error=str(exc),
            )

        intent = str(parsed.get("intent") or "smalltalk")
        intent = INTENT_ALIASES.get(intent, intent)
        if intent not in SUPPORTED_INTENTS:
            fallback = self._fallback_nlu_result(
                text,
                payload,
                rule_candidate,
                rag_match,
                nlu_error=f"Unsupported LLM intent: {intent}",
            )
            return fallback

        llm_slots: dict[str, Any] = {}
        if rag_match is not None and rule_candidate is None:
            llm_slots.update(
                self._slots_for_rag_intent(rag_match.intent, text, payload)
            )
        if rule_candidate is not None:
            llm_slots.update(rule_candidate.slots)
        llm_slots.update(parsed.get("slots") or {})
        llm_slots.update(payload.get("slots", {}) or {})
        confidence = float(parsed.get("confidence") or 0.80)

        # Keep deterministic lexical facts authoritative when the LLM omitted
        # or misread them, while still letting the LLM decide the intent.
        if rule_candidate is not None:
            for key in [
                "date",
                "timeRange",
                "ticketCount",
                "seatPositions",
                "seatType",
                "seatPreference",
                "autoSelectSeats",
                "cinemaName",
                "cinemaId",
                "location",
                "hallType",
                "maxPrice",
            ]:
                if key in rule_candidate.slots:
                    llm_slots[key] = rule_candidate.slots[key]
            payload_slots = payload.get("slots", {}) or {}
            if (
                "ticketCount" not in rule_candidate.slots
                and "ticketCount" not in payload_slots
                and self._extract_ticket_count(text) is None
            ):
                # A seat coordinate such as "3排1座" is not permission for
                # the model to infer a one-ticket order.
                llm_slots.pop("ticketCount", None)

        # Normalise date aliases
        date_value = llm_slots.get("date")
        if isinstance(date_value, str):
            date_value = date_value.strip()
            if date_value in ("今天", "今晚"):  date_value = "today"
            if date_value in ("明天", "明晚"):  date_value = "tomorrow"
            if date_value in ("后天",):        date_value = "after_tomorrow"
            llm_slots["date"] = date_value

        seat_positions = self._extract_seat_positions(text)
        if seat_positions:
            llm_slots["seatPositions"] = seat_positions
            if intent in {"smalltalk", "search_movies"}:
                intent = "book_ticket"
                confidence = max(confidence, 0.95)

        multi_movie_names = self._extract_multi_movie_names(text)
        if len(multi_movie_names) > 1:
            llm_slots["movieNames"] = multi_movie_names
            llm_slots.pop("movieName", None)
            if intent in {"smalltalk", "search_movies", "book_ticket"}:
                intent = "multi_movie_booking"
                confidence = max(confidence, 0.92)

        if (
            rule_candidate is not None
            and rule_candidate.intent == "search_movies"
            and (
                rule_candidate.slots.get("actor")
                or rule_candidate.slots.get("recommendationCriteria")
            )
        ):
            llm_slots.pop("movieName", None)
            if (
                rule_candidate.slots.get("actor")
                and self._extract_genre(text) is None
            ):
                llm_slots.pop("genre", None)

        if self._should_preserve_rule_intent(
            text,
            rule_candidate,
            intent,
        ):
            intent = rule_candidate.intent
            confidence = max(confidence, rule_candidate.confidence)

        # An uncertain smalltalk answer must not erase a strong booking
        # candidate produced from explicit purchase language.
        if (
            rule_candidate is not None
            and rule_candidate.intent not in {"smalltalk"}
            and intent == "smalltalk"
            and confidence < 0.65
        ):
            intent = rule_candidate.intent
            confidence = rule_candidate.confidence

        # Strip hallucinated movie names for intents that don't need one.
        _movie_name = llm_slots.get("movieName", "")
        if intent in {"select_showtime", "select_or_modify"} and isinstance(_movie_name, str):
            if any(noise in _movie_name for noise in ["场次", "场", "第", "选"]):
                llm_slots.pop("movieName", None)

        return NLUResult(
            intent=intent,
            confidence=confidence,
            intent_source="llm",
            rag_score=rag_match.score if rag_match else None,
            rag_example=rag_match.example if rag_match else None,
            slots=llm_slots,
            is_modification=bool(parsed.get("is_modification")),
            reference_text=str(parsed.get("reference") or text),
        )

    def _should_preserve_rule_intent(
        self,
        text: str,
        rule_candidate: NLUResult | None,
        llm_intent: str,
    ) -> bool:
        """Reject LLM routes that contradict an explicit business expression."""
        if rule_candidate is None or llm_intent == rule_candidate.intent:
            return False
        if (
            rule_candidate.intent == "search_showtimes"
            and not self._is_booking_request_text(text)
            and not any(word in _normalize_short_text(text) for word in ["座位", "选座"])
        ):
            return True
        return False

    def _fallback_nlu_result(
        self,
        text: str,
        payload: dict[str, Any],
        rule_candidate: NLUResult | None,
        rag_match: IntentMatch | None,
        nlu_error: str | None = None,
    ) -> NLUResult:
        if rule_candidate is not None:
            return rule_candidate
        if rag_match is not None:
            rag_result = self._result_from_rag_match(text, payload, rag_match)
            if rag_result is not None:
                return rag_result
        return NLUResult(
            intent="smalltalk",
            confidence=0.10,
            intent_source="llm",
            slots={"nluError": nlu_error} if nlu_error else {},
            reference_text=text,
        )

    def _result_from_rag_match(
        self,
        text: str,
        payload: dict[str, Any],
        match: IntentMatch | None,
    ) -> NLUResult | None:
        if match is None:
            return None
        slots = self._slots_for_rag_intent(match.intent, text, payload)
        confidence = min(0.96, max(0.70, match.score))
        return NLUResult(
            intent=match.intent,
            confidence=confidence,
            intent_source="rag",
            rag_score=match.score,
            rag_example=match.example,
            slots=slots,
            is_modification=match.intent == "select_or_modify",
            reference_text=text,
        )

    def _slots_for_rag_intent(
        self,
        intent: str,
        text: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if intent == "book_ticket":
            return self._extract_booking_slots(text, payload)
        if intent == "search_movies":
            slots = self._extract_recommendation_slots(text)
            actor = self._extract_actor_movie_query(text)
            if actor:
                slots["actor"] = actor
            movie_name = self._extract_movie_search_keyword(text)
            if movie_name and not slots.get("genre") and not slots.get("actor"):
                slots["movieName"] = movie_name
            return slots
        if intent == "nearby_cinema":
            slots = dict(payload.get("slots", {}) or {})
            if payload.get("location") not in [None, ""]:
                slots["location"] = payload.get("location")
            if payload.get("city") not in [None, ""]:
                slots["city"] = payload.get("city")
            return slots
        if intent == "location_query":
            slots = {}
            if payload.get("location") not in [None, ""]:
                slots["location"] = payload.get("location")
            return slots
        if intent == "snack":
            snack_requests = self._extract_snack_requests(text)
            return {"snackRequests": snack_requests} if snack_requests else {}
        return {}

    def _extract_booking_slots(
        self,
        text: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        slots = dict(payload.get("slots", {}) or {})
        if payload.get("location") not in [None, ""]:
            slots["location"] = payload.get("location")

        ticket_count = self._extract_ticket_count(text)
        if ticket_count is not None:
            slots["ticketCount"] = ticket_count

        date_value = self._extract_date(text)
        if date_value:
            slots["date"] = date_value

        time_range = self._extract_time_range(text)
        if time_range:
            slots["timeRange"] = time_range

        seat_positions = self._extract_seat_positions(text)
        if seat_positions:
            slots["seatPositions"] = seat_positions

        seat_type = self._extract_seat_type(text)
        if seat_type:
            slots["seatType"] = seat_type
            if seat_type == "couple" and ticket_count is None:
                ticket_count = 2
                slots["ticketCount"] = 2

        seat_preference_slots = self._extract_seat_preference_slots(text)
        if seat_preference_slots:
            slots.update(seat_preference_slots)
        elif seat_type:
            slots["autoSelectSeats"] = True
            slots.setdefault("seatPreference", "middle")

        genre = self._extract_genre(text)
        if genre:
            slots["genre"] = genre

        snack_requests = self._extract_snack_requests(text)
        if snack_requests:
            slots["snackRequests"] = snack_requests

        max_price = self._extract_max_price(text, ticket_count)
        if max_price is not None:
            slots["maxPrice"] = max_price

        cinema_name = self._extract_cinema_name(text)
        if cinema_name:
            slots["cinemaName"] = cinema_name

        hall_type = self._extract_hall_type(text)
        if hall_type:
            slots["hallType"] = hall_type

        movie_name = self._extract_movie_name(text, snack_requests, cinema_name)
        if movie_name:
            slots["movieName"] = movie_name

        if self._wants_auto_showtime(text):
            slots["autoSelectShowtime"] = True
            if not date_value and not time_range:
                slots.setdefault("timePreference", "earliest")
        if is_single_ticket_auto_purchase_request(text):
            slots["ticketCount"] = 1
            slots["autoSelectShowtime"] = True
            slots["autoSelectSeats"] = True
            slots["seatPreference"] = "random"
            slots["skipSnacks"] = True
        if is_text_auto_seat_or_auto_purchase(text):
            slots["autoSelectSeats"] = True
            slots["seatPreference"] = _auto_seat_preference(text)
        if self._wants_direct_payment(text):
            slots["skipSnacks"] = True

        return slots

    def _is_deterministic_booking_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if not any(word in normalized for word in ["买", "订", "预订", "电影票", "影票", "票"]):
            return False
        slots = self._extract_booking_slots(text, {})
        strong_slot_count = sum(
            1
            for key in [
                "movieName",
                "cinemaName",
                "date",
                "timeRange",
                "ticketCount",
                "hallType",
                "maxPrice",
            ]
            if slots.get(key) not in [None, ""]
        )
        return strong_slot_count >= 2

    @staticmethod
    def _wants_auto_showtime(text: str) -> bool:
        normalized = _normalize_short_text(text)
        return any(
            phrase in normalized
            for phrase in [
                "随便选一场",
                "随便选场",
                "场次随便",
                "随便买一张",
                "帮我选一场",
                "你帮我选一场",
                "自动选场",
                "场次随机",
                "随机一场",
                "随便一场",
                "场次都行",
                "场次都可以",
                "场次无所谓",
            ]
        )

    @staticmethod
    def _wants_direct_payment(text: str) -> bool:
        normalized = _normalize_short_text(text)
        return any(
            phrase in normalized
            for phrase in [
                "直接支付",
                "直接到支付",
                "直接去支付",
                "跳过零食",
                "不需要零食",
                "不要零食",
            ]
        )

    def _extract_seat_positions(self, text: str) -> list[dict[str, int]]:
        normalized = re.sub(r"\s+", "", text)
        pattern = re.compile(
            r"(?P<row>[0-9一二三四五六七八九十两]+)排"
            r"(?P<seat>[0-9一二三四五六七八九十两]+)"
            r"(?:座|号|位)?"
        )
        positions: list[dict[str, int]] = []
        for match in pattern.finditer(normalized):
            row_no = self._parse_number_token(match.group("row"))
            seat_no = self._parse_number_token(match.group("seat"))
            if row_no is None or seat_no is None:
                continue
            position = {"rowNo": row_no, "seatNo": seat_no}
            if position not in positions:
                positions.append(position)
        return positions

    @staticmethod
    def _extract_seat_type(text: str) -> str:
        normalized = _normalize_short_text(text)
        if any(word in normalized for word in ["情侣座", "情侣厅", "双人座"]):
            return "couple"
        if any(word in normalized for word in ["普通座", "普通票", "普通位", "普通座位"]):
            return "standard"
        return ""

    def _extract_ticket_count(self, text: str) -> int | None:
        normalized = re.sub(r"\s+", "", text)
        match = re.search(r"(?P<count>[0-9一二两三四五六七八九十]+)\s*(?:张|份|个)?(?:电影票|影票|票)", normalized)
        if not match:
            match = re.search(
                r"(?P<count>[0-9一二两三四五六七八九十]+)张"
                rf"(?=(?:的)?(?:电影|影片|片|{GENRE_TERMS_PATTERN}))",
                normalized,
            )
        if not match:
            match = re.search(r"(?:买|订|要|来)(?P<count>[0-9一二两三四五六七八九十]+)张", normalized)
        if not match:
            match = re.search(
                r"(?P<count>[0-9一二两三四五六七八九十]+)张(?:普通座|座位)",
                normalized,
            )
        if not match:
            match = re.search(
                r"(?P<count>[0-9一二两三四五六七八九十]+)(?:个|张)?"
                r"(?:情侣座|情侣厅|双人座|普通座|普通位|普通座位|座位)",
                normalized,
            )
        if not match:
            match = re.search(
                r"(?P<count>[0-9一二两三四五六七八九十]+)(?:个)?人",
                normalized,
            )
        if not match:
            return None
        count = self._parse_number_token(match.group("count"))
        return count if count and count > 0 else None

    def _extract_seat_action_slots(
        self,
        text: str,
        state: AgentState | None = None,
    ) -> dict[str, Any] | None:
        normalized = _normalize_short_text(text)
        has_seat_word = any(
            word in normalized
            for word in ["座位", "选座", "位置", "座", "观影区", "观影"]
        )
        has_seat_action = any(
            word in normalized
            for word in ["换", "改", "重新", "重选", "选", "坐", "安排", "帮我选", "给我选"]
        )
        if not has_seat_word or not has_seat_action:
            return None

        slots = self._extract_seat_preference_slots(text)
        seat_type = self._extract_seat_type(text)
        if seat_type:
            slots["seatType"] = seat_type
            slots.setdefault("autoSelectSeats", True)
            slots.setdefault("seatPreference", "middle")

        count = self._extract_ticket_count(text)
        if count is not None:
            slots["ticketCount"] = count
        elif seat_type == "couple":
            current_count = self._parse_number_token(
                str((state.slots or {}).get("ticketCount"))
            ) if state else None
            slots["ticketCount"] = current_count if current_count and current_count % 2 == 0 else 2

        if is_text_auto_seat_or_auto_purchase(text):
            slots["autoSelectSeats"] = True
            slots.setdefault("seatPreference", _auto_seat_preference(text))

        if not slots and is_change_seat_text(text):
            return {}
        return slots or None

    def _extract_max_price(self, text: str, ticket_count: int | None = None) -> float | None:
        normalized = re.sub(r"\s+", "", text)

        per_person = re.search(
            r"(?:预算)?(?:每人|每张|单张|一张|人均)(?:不超过|不超|以内|低于|小于|不高于|最多|最高)?"
            r"(?P<amount>\d+(?:\.\d+)?|[一二两三四五六七八九十百]+)(?:元|块|块钱)",
            normalized,
        )
        if not per_person:
            per_person = re.search(
                r"(?P<amount>\d+(?:\.\d+)?|[一二两三四五六七八九十百]+)(?:元|块|块钱)"
                r"(?:以内|以下|内|每人|每张|单张|一张|人均)",
                normalized,
            )
        if per_person:
            return self._parse_price_amount(per_person.group("amount"))

        total = re.search(
            r"(?:预算|一共|总共|总价|合计)(?:不超过|不超|以内|低于|小于|不高于|最多|最高)?"
            r"(?P<amount>\d+(?:\.\d+)?|[一二两三四五六七八九十百]+)(?:元|块|块钱)",
            normalized,
        )
        if total:
            amount = self._parse_price_amount(total.group("amount"))
            if amount is None:
                return None
            if ticket_count and ticket_count > 0:
                return round(amount / ticket_count, 2)
            return amount

        return None

    def _parse_price_amount(self, value: str) -> float | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            parsed = self._parse_number_token(text)
            return float(parsed) if parsed is not None else None

    def _extract_date(self, text: str) -> str | None:
        explicit_date = self._extract_explicit_month_day(text)
        if explicit_date:
            return explicit_date
        if "后天" in text:
            return "after_tomorrow"
        if "明天" in text or "明晚" in text:
            return "tomorrow"
        if "今天" in text or "今晚" in text:
            return "today"
        if "周末" in text:
            return "weekend"
        return None

    def _extract_explicit_month_day(self, text: str) -> str | None:
        normalized = re.sub(r"\s+", "", text)
        match = re.search(
            r"(?P<month>\d{1,2}|[一二两三四五六七八九十]{1,3})月"
            r"(?P<day>\d{1,2}|[一二两三四五六七八九十]{1,3})(?:号|日)?",
            normalized,
        )
        if not match:
            return None
        month = self._parse_number_token(match.group("month"))
        day = self._parse_number_token(match.group("day"))
        if month is None or day is None:
            return None
        now = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        try:
            parsed = date(now.year, month, day)
        except ValueError:
            return None
        if parsed < now:
            try:
                parsed = date(now.year + 1, month, day)
            except ValueError:
                return None
        return parsed.isoformat()

    def _extract_time_range(self, text: str) -> str | None:
        text_without_titles = re.sub(r"[《【][^》】]+[》】]", "", text)
        normalized = re.sub(r"\s+", "", text_without_titles)
        period_match = re.search(
            r"(上午|早上|中午|下午|晚上|今晚|明晚)"
            r"([0-9一二两三四五六七八九十]{1,3})(?:[:：点])"
            r"([0-9一二两三四五六七八九十]{0,2})?(半)?",
            normalized,
        )
        if not period_match:
            period_match = re.search(
                r"(?<![A-Za-z0-9\u4e00-\u9fff])"
                r"([0-9一二两三四五六七八九十]{1,3})(?:点)"
                r"([0-9一二两三四五六七八九十]{0,2})?(半)?",
                normalized,
            )
        if period_match:
            period = period_match.group(1) or "" if len(period_match.groups()) == 4 else ""
            hour_group_index = 2 if len(period_match.groups()) == 4 else 1
            minute_group_index = 3 if len(period_match.groups()) == 4 else 2
            half_group_index = 4 if len(period_match.groups()) == 4 else 3
            hour = self._parse_number_token(period_match.group(hour_group_index))
            minute_text = period_match.group(minute_group_index)
            minute = self._parse_number_token(minute_text) if minute_text else 0
            if period_match.group(half_group_index):
                minute = 30
            if hour is not None and minute is not None:
                if period in {"下午", "晚上", "今晚", "明晚"} and 1 <= hour < 12:
                    hour += 12
                if period == "中午" and hour < 11:
                    hour += 12
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return f"{hour:02d}:{minute:02d}"

        if any(value in normalized for value in ["今晚", "明晚", "晚上"]):
            return "evening"
        if "下午" in normalized:
            return "afternoon"
        if "上午" in normalized or "早上" in normalized:
            return "morning"
        return None

    def _extract_genre(self, text: str) -> str | None:
        return canonical_genre_from_text(text)

    def _should_clear_stale_showtime_constraints(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if not any(
            marker in normalized
            for marker in [
                "场次",
                "有哪些场",
                "有什么场",
                "放映时间",
                "几点能看",
                "什么时候能看",
                "什么时候有",
            ]
        ):
            return False
        if self._extract_date(text):
            return False
        if self._extract_time_range(text):
            return False
        if self._extract_ticket_count(text) is not None:
            return False
        return True

    def _extract_snack_requests(self, text: str) -> list[dict[str, Any]]:
        normalized = re.sub(r"\s+", "", text)
        snack_names = r"爆米花|可乐|薯条|热狗|饮料|套餐"
        connector = re.search(
            rf"(?:和|加|再加|再来|以及|外加)"
            rf"(?=[0-9一二两三四五六七八九十]*"
            rf"(?:桶|瓶|杯|份|个|盒)?(?:{snack_names}))",
            normalized,
        )
        if connector:
            tail = normalized[connector.end():]
        else:
            if not re.search(snack_names, normalized):
                return []
            tail = re.sub(
                r"^(?:我要|我想要|想要|要|买|来|给我|帮我买|加|再加|再来)+",
                "",
                normalized,
            )
        pattern = re.compile(
            rf"(?P<quantity>[0-9一二两三四五六七八九十]+)?"
            rf"(?P<unit>桶|瓶|杯|份|个|盒)?"
            rf"(?P<name>{snack_names})"
        )
        requests: list[dict[str, Any]] = []
        for match in pattern.finditer(tail):
            quantity = self._parse_number_token(match.group("quantity") or "一") or 1
            requests.append(
                {
                    "name": match.group("name"),
                    "quantity": max(1, quantity),
                    "unit": match.group("unit") or "份",
                }
            )
        return requests

    def _extract_movie_name(
        self,
        text: str,
        snack_requests: list[dict[str, Any]] | None = None,
        cinema_name: str | None = None,
    ) -> str:
        cleaned = re.sub(r"\s+", "", text)
        if cinema_name:
            cleaned = cleaned.replace(cinema_name, "")
        quoted = re.search(r"[《【](?P<name>[^》】]+)[》】]", cleaned)
        if quoted:
            return quoted.group("name").strip()
        if snack_requests:
            snack_names = r"爆米花|可乐|薯条|热狗|饮料|套餐"
            cleaned = re.sub(
                rf"(?:和|加|再加|再来|以及|外加)"
                rf"[0-9一二两三四五六七八九十]*(?:桶|瓶|杯|份|个|盒)?"
                rf"(?:{snack_names}).*$",
                "",
                cleaned,
            )
        cleaned = re.sub(
            r"[0-9一二两三四五六七八九十]+排[0-9一二两三四五六七八九十]+(?:座|号|位)?",
            "",
            cleaned,
        )
        cleaned = self._strip_booking_modifier_clauses(cleaned)
        marker_match = re.search(r"(?P<name>.+?)(?:电影票|影票)", cleaned)
        if marker_match:
            candidate = self._clean_movie_candidate(marker_match.group("name"))
            if cinema_name:
                candidate = self._clean_movie_candidate(candidate.replace(cinema_name, ""))
            if candidate:
                return candidate

        for pattern in [
            r"(?:买|订|预订|要|来)(?:[0-9一二两三四五六七八九十]+张)?(?P<name>[^，,。；;]+?)(?:的)?(?:电影票|影票|票|普通座|座位|IMAX厅|杜比厅|巨幕厅|数字厅|$)",
            r"(?:看|观看)(?P<name>[^，,。；;]+?)(?:的)?(?:场次|排片|电影票|影票|票)",
        ]:
            match = re.search(pattern, cleaned)
            if not match:
                continue
            candidate = self._clean_movie_candidate(match.group("name"))
            if candidate and candidate not in {"普通座", "座位", "票"}:
                return candidate

        cleaned = re.split(
            r"(?:，|,|。|；|;)?(?:\d{1,2}|[一二两三四五六七八九十]{1,3})月"
            r"(?:\d{1,2}|[一二两三四五六七八九十]{1,3})(?:号|日)?",
            cleaned,
            maxsplit=1,
        )[0]
        cleaned = re.split(
            r"(?:CGV|万达|奥斯卡|影院|影城|电影院|IMAX厅|MAX厅|数字厅)",
            cleaned,
            maxsplit=1,
        )[0]
        cleaned = re.sub(r"[0-9一二两三四五六七八九十]+张(?:电影票|影票|票)?", "", cleaned)
        cleaned = re.sub(r"(今天|今晚|明天|明晚|后天|周末)", "", cleaned)
        cleaned = re.sub(
            r"(上午|早上|中午|下午|晚上)?[0-9一二两三四五六七八九十]{1,3}[:：点][0-9一二两三四五六七八九十]{0,2}(?:半)?(?:左右|前后|以后|之后|之前|前)?的?",
            "",
            cleaned,
        )
        cleaned = self._clean_movie_candidate(cleaned)
        if cleaned in {"", "片", "部片", "一部片"} or is_genre_phrase(cleaned):
            return ""
        return cleaned

    @staticmethod
    def _strip_seat_preference_clause(text: str) -> str:
        preference = (
            r"最佳观影区|黄金观影区|最佳观影|黄金位置|好位置|"
            r"中间(?:的)?(?:座位|位置)?|靠中间(?:的)?(?:座位|位置)?|"
            r"居中(?:的)?(?:座位|位置)?|中部(?:的)?(?:座位|位置)?|"
            r"正中(?:的)?(?:座位|位置)?|中央(?:的)?(?:座位|位置)?|"
            r"前排(?:座位)?|靠前(?:的)?(?:座位|位置)?|"
            r"后排(?:座位)?|靠后(?:的)?(?:座位|位置)?"
        )
        return re.sub(
            rf"(?:[，,。；;、]|^)?(?:我想要|想要|我要|要|选|坐|安排)?(?:{preference})(?:的)?$",
            "",
            text,
        )

    @classmethod
    def _strip_booking_modifier_clauses(cls, text: str) -> str:
        cleaned = cls._strip_seat_preference_clause(text)
        hall = (
            r"IMAX厅|IMAX影厅|IMAX|MAX厅|MAX影厅|杜比厅|杜比影厅|杜比|"
            r"巨幕厅|巨幕影厅|巨幕|激光厅|激光影厅|激光|"
            r"4DX厅|4DX影厅|4DX|MX4D厅|MX4D影厅|MX4D|数字厅|数字影厅"
        )
        seat_type = r"情侣座|情侣厅|双人座|普通座|普通票|普通位|普通座位"
        modifier = rf"(?:{hall}|{seat_type})"
        previous = None
        while previous != cleaned:
            previous = cleaned
            cleaned = re.sub(
                rf"(?:[，,。；;、]|^)?(?:我想要|想要|我要|要|选|坐|安排)?(?:{modifier})(?:的)?$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = cls._strip_seat_preference_clause(cleaned)
        return cleaned

    def _extract_cinema_name(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        suffix = r"(?:电影院|影城|影院)"
        branch = r"(?:(?:[（(][^）)]{1,20}[）)])|(?:[A-Za-z0-9\u4e00-\u9fff·_-]{1,12}店))?"
        known_brands = (
            "CGV|万达|大地|奥斯卡|博纳|金逸|中影|横店|UME|卢米埃|"
            "保利|橙天嘉禾|星美|幸福蓝海"
        )
        patterns = (
            re.compile(
                rf"(?P<name>(?:[A-Za-z\u4e00-\u9fff]{{0,8}})?(?:{known_brands})"
                rf"[A-Za-z0-9\u4e00-\u9fff·_-]{{0,12}}{suffix}{branch})(?:的)?",
                re.IGNORECASE,
            ),
            re.compile(
                rf"(?P<name>[A-Za-z0-9\u4e00-\u9fff·_-]{{1,30}}{suffix}{branch})(?:的)?"
            ),
        )
        generic_terms = {
            "影院",
            "影城",
            "电影院",
            "附近影院",
            "附近影城",
            "附近电影院",
            "最近影院",
            "最近影城",
            "最近电影院",
            "最近的影院",
            "最近的影城",
            "最近的电影院",
            "当前影院",
            "这个影院",
            "那家影院",
            "其他影院",
            "其它影院",
            "别的影院",
            "别家影院",
            "其他影城",
            "其它影城",
            "别的影城",
            "其他电影院",
            "其它电影院",
            "别的电影院",
        }
        for pattern in patterns:
            for match in pattern.finditer(normalized):
                candidate = self._clean_cinema_candidate(match.group("name"))
                if not candidate or candidate in generic_terms:
                    continue
                if any(marker in candidate for marker in ["附近", "最近", "周边", "当前", "这个", "那家"]):
                    continue
                return candidate
        return ""

    @staticmethod
    def _extract_hall_type(text: str) -> str | None:
        normalized = re.sub(r"\s+", "", text).upper()
        for marker, hall_type in [
            ("IMAX", "IMAX"),
            ("MAX厅", "IMAX"),
            ("MAX影厅", "IMAX"),
            ("DOLBY", "杜比"),
            ("杜比", "杜比"),
            ("4DX", "4DX"),
            ("MX4D", "MX4D"),
            ("巨幕", "巨幕"),
        ]:
            if marker in normalized:
                return hall_type
        return None

    def _extract_hall_type_showtime_slots(self, text: str) -> dict[str, Any]:
        hall_type = self._extract_hall_type(text)
        if not hall_type:
            return {}
        if (
            self._extract_date(text)
            or self._extract_time_range(text)
            or self._extract_ticket_count(text) is not None
            or self._extract_max_price(text) is not None
            or self._extract_cinema_name(text)
        ):
            return {}

        normalized = _normalize_short_text(text)
        if not any(marker in normalized for marker in ["场次", "场", "放映", "影厅", "厅", "的"]):
            return {}
        if not any(
            marker in normalized
            for marker in ["选", "找", "查", "有", "哪", "推荐", "帮我", "个"]
        ):
            return {}

        return {
            "hallType": hall_type,
            "timePreference": "earliest",
            "showtimeLimit": 3,
        }

    def _clean_cinema_candidate(self, value: str) -> str:
        cleaned = re.sub(r"\s+", "", str(value or ""))
        cleaned = re.sub(r"^[0-9一二两三四五六七八九十]+张(?:电影票|影票|票)?", "", cleaned)
        cleaned = re.sub(
            r"(?:\d{1,2}|[一二两三四五六七八九十]{1,3})月"
            r"(?:\d{1,2}|[一二两三四五六七八九十]{1,3})(?:号|日)?",
            "",
            cleaned,
        )
        cleaned = re.sub(r"(今天|今晚|明天|明晚|后天|周末)", "", cleaned)
        cleaned = re.sub(r"(上午|早上|中午|下午|晚上)的?", "", cleaned)
        cleaned = re.sub(
            r"(上午|早上|中午|下午|晚上)?[0-9一二两三四五六七八九十]{1,3}[:：点][0-9一二两三四五六七八九十]{0,2}(?:半)?(?:左右|前后|以后|之后|之前|前)?的?",
            "",
            cleaned,
        )
        cleaned = re.sub(r"^(在|我看看|看看|给我|帮我|我要|我想|想要|请帮我|麻烦帮我|买|订|预订|来|看|想看|去看)+", "", cleaned)
        cleaned = re.sub(r"^[0-9一二两三四五六七八九十]+张(?:电影票|影票|票)?", "", cleaned)
        cleaned = re.sub(r"^(?:点|:|：)+", "", cleaned)
        brand_match = re.search(
            r"(CGV|万达|大地|奥斯卡|博纳|金逸|中影|横店|UME|卢米埃|保利|橙天嘉禾|星美|幸福蓝海)",
            cleaned,
            re.IGNORECASE,
        )
        if brand_match and brand_match.start() > 0:
            prefix = cleaned[:brand_match.start()]
            if len(prefix) > 2 and not prefix.endswith(("市", "区", "县")):
                cleaned = cleaned[brand_match.start():]
        cleaned = cleaned.strip("的《》<>【】[]，,。.!?！？：:")
        return cleaned

    def _extract_movie_search_keyword(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        quoted = re.search(r"[《【](?P<name>[^》】]+)[》】]", normalized)
        if quoted:
            return quoted.group("name").strip()
        if (
            "适合" in normalized
            and any(word in normalized for word in ["电影", "影片", "片"])
            and any(word in normalized for word in ["看", "观影"])
        ):
            return ""

        candidate = normalized
        candidate = re.sub(r"^(我想|想|我要|要|帮我|给我|请|麻烦)?(?:看|找|查|搜索|有没有|有无|有)?", "", candidate)
        candidate = re.sub(r"(?:电影|影片|片子|片)(?:吗|么|嘛|呀|啊)?$", "", candidate)
        candidate = re.sub(r"(?:有没有|有无|有|哪些|什么|推荐|最近|热映|上映|正在上映)", "", candidate)
        candidate = self._clean_movie_candidate(candidate)
        if candidate in {
            "",
            "电影",
            "影片",
            "片",
            "片子",
            "高分",
            "热门",
            "热映",
            "推荐",
            "评分",
            "评分高",
            "高评分",
            "评价",
            "好评",
            "比较火",
            "最火",
            "火爆",
            "票房",
            "票房最高",
            "最高票房",
        }:
            return ""
        return candidate

    def _clean_movie_candidate(self, value: str) -> str:
        cleaned = re.sub(r"\s+", "", str(value or ""))
        cleaned = re.sub(r"[0-9一二两三四五六七八九十]+张(?:电影票|影票|票)?", "", cleaned)
        cleaned = re.sub(
            r"(?:\d{1,2}|[一二两三四五六七八九十]{1,3})月"
            r"(?:\d{1,2}|[一二两三四五六七八九十]{1,3})(?:号|日)?",
            "",
            cleaned,
        )
        cleaned = re.sub(r"(今天|今晚|明天|明晚|后天|周末)", "", cleaned)
        cleaned = re.sub(
            r"(上午|早上|中午|下午|晚上)?[0-9一二两三四五六七八九十]{1,3}[:：点][0-9一二两三四五六七八九十]{0,2}(?:半)?(?:左右|前后|以后|之后|之前|前)?的?",
            "",
            cleaned,
        )
        cleaned = re.sub(r"^(上午|早上|中午|下午|晚上)+", "", cleaned)
        cleaned = re.sub(
            r"^(在|给我|帮我|我要|我想|想要|请帮我|麻烦帮我|买|订|预订|来|看|想看|去看)+",
            "",
            cleaned,
        )
        cleaned = re.sub(r"^(上午|早上|中午|下午|晚上)+", "", cleaned)
        cleaned = cleaned.strip("的《》<>【】[]()，,。.!?！？：:")
        cleaned = re.sub(r"[，,。；;、]?(?:在|于)$", "", cleaned)
        cleaned = cleaned.strip("的《》<>【】[]()，,。.!?！？：:")
        cleaned = re.sub(r"(电影票|影票|电影|影片|票)+$", "", cleaned)
        cleaned = cleaned.strip("的《》<>【】[]()，,。.!?！？：:")
        return cleaned

    def _extract_multi_movie_names(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", "", text)
        if not any(marker in normalized for marker in ["买", "订", "要", "票", "影票", "电影票"]):
            return []
        if not any(marker in normalized for marker in ["和", "及", "以及", "还有", "、"]):
            return []

        segments = re.split(r"和|及|以及|还有|、", normalized)
        names: list[str] = []
        for segment in segments:
            if self._is_seat_only_segment(segment):
                continue
            cleaned = self._strip_movie_segment(segment)
            cleaned = re.sub(
                r"[0-9一二两三四五六七八九十]+排[0-9一二两三四五六七八九十]+(?:座|号|位)?",
                "",
                cleaned,
            )
            if cleaned and not any(
                snack in cleaned
                for snack in ["爆米花", "可乐", "薯条", "热狗", "饮料", "套餐"]
            ):
                names.append(cleaned)

        deduped: list[str] = []
        for name in names:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _strip_movie_segment(self, segment: str) -> str:
        cleaned = str(segment or "")
        cinema_name = self._extract_cinema_name(cleaned)
        if cinema_name:
            cleaned = cleaned.replace(cinema_name, "")
        cleaned = re.sub(
            r"^(在|给我|帮我|我要|我想|想要|买|订|预订|来|各|一张|两张|三张|四张|五张|六张|七张|八张|九张|十张|今天|明天|今晚|明晚|后天|上午|下午|晚上|的)+",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(各?[0-9一二两三四五六七八九十]+张|电影票|影票|电影|票|的)+$",
            "",
            cleaned,
        )
        cleaned = cleaned.strip("《》<>【】[]()，,。.!?：:")
        return cleaned

    @staticmethod
    def _is_seat_only_segment(segment: str) -> bool:
        normalized = re.sub(r"\s+", "", str(segment or ""))
        return bool(
            re.fullmatch(
                r"[0-9一二两三四五六七八九十]+排[0-9一二两三四五六七八九十]+(?:座|号|位)?",
                normalized,
            )
        )

    def _parse_number_token(self, value: str) -> int | None:
        text = str(value or "").strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)

        digits = {
            "零": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        if text in digits:
            return digits[text]
        if len(text) == 2 and text[0] == "十" and text[1] in digits:
            return 10 + digits[text[1]]
        if len(text) == 2 and text[1] == "十" and text[0] in digits:
            return digits[text[0]] * 10
        if len(text) == 3 and text[1] == "十" and text[0] in digits and text[2] in digits:
            return digits[text[0]] * 10 + digits[text[2]]
        return None

    def _is_price_query_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        return any(
            phrase in normalized
            for phrase in ["多少钱", "什么价位", "票价", "价格", "多少钱一张", "多少钱啊"]
        )

    def _is_nearby_cinema_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if any(
            phrase in normalized
            for phrase in ["附近有什么影院", "附近影院", "周边影院", "附近影城", "nearbycinema", "附近有啥影院"]
        ):
            return True

        has_nearby_marker = any(
            phrase in normalized
            for phrase in ["附近", "周边", "离我近", "离我最近", "最近的"]
        )
        has_cinema_marker = any(
            phrase in normalized
            for phrase in ["影院", "影城", "电影院"]
        )
        has_showtime_marker = any(
            phrase in normalized
            for phrase in ["场次", "几点", "什么时候", "最早", "放映", "能看", "排片"]
        )
        return has_nearby_marker and has_cinema_marker and not has_showtime_marker

    @staticmethod
    def _is_location_query_text(text: str) -> bool:
        normalized = _normalize_short_text(text)
        return normalized in {
            "我在哪里",
            "我在哪",
            "我在哪儿",
            "我现在在哪里",
            "我现在在哪",
            "我的当前位置",
            "我的位置",
            "我的地理位置",
            "我的地理位置在哪里",
            "我的位置在哪里",
            "当前位置",
            "当前定位",
            "定位信息",
            "这里是哪里",
            "这里在哪",
            "这是哪里",
            "我的坐标",
            "当前坐标",
            "当前经纬度是多少",
        }

    @staticmethod
    def _is_movie_knowledge_question(text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        if not normalized:
            return False
        if any(
            marker in normalized
            for marker in [
                "退票",
                "退款",
                "改签",
                "优惠券",
                "取票",
                "支付",
                "付款",
                "订单",
                "锁座",
                "座位",
                "规则",
                "政策",
            ]
        ):
            return False
        if any(
            marker in normalized
            for marker in [
                "谁演的",
                "谁主演",
                "主演是谁",
                "演员是谁",
                "谁饰演",
                "扮演者",
                "导演是谁",
                "谁导演",
                "讲什么",
                "剧情",
                "结局",
                "彩蛋",
                "片长",
                "上映时间",
                "好看吗",
                "好看么",
                "好不好看",
                "值得看吗",
                "值得一看吗",
            ]
        ):
            return True
        return bool(re.search(r"(?:.+?)(?:是谁|是什么|什么意思)$", normalized))

    @staticmethod
    def _apply_nearby_showtime_preferences(
        text: str,
        slots: dict[str, Any],
    ) -> None:
        """Translate location-first showtime wording into deterministic filters."""
        normalized = _normalize_short_text(text)
        nearby_first = any(
            phrase in normalized
            for phrase in [
                "距离我最近",
                "离我最近",
                "最近的影院",
                "最近影院",
                "离我近",
                "就近影院",
            ]
        )
        if not nearby_first:
            return

        slots["nearbyFirst"] = True
        slots.setdefault("cinemaLimit", 5)

        wants_one_earliest = any(
            phrase in normalized
            for phrase in [
                "最早一场",
                "最早",
                "最近一场",
                "时间最近",
                "离现在最近",
                "最近几点",
                "最近能看",
                "找一个",
                "一场",
            ]
        )
        if wants_one_earliest:
            slots["timePreference"] = "earliest"
            slots["showtimeLimit"] = 1

    def _no_showtime_followup_slots(
        self,
        text: str,
        state: AgentState | None,
    ) -> dict[str, Any] | None:
        if state is None:
            return None
        if (
            state.slots.get("movieId") in [None, ""]
            and state.slots.get("movieName") in [None, ""]
            and state.slots.get("genre") in [None, ""]
        ):
            return None
        if not self._is_no_showtime_context(state):
            return None

        normalized = _normalize_short_text(text)
        wants_time_relax = any(
            phrase in normalized
            for phrase in [
                "有什么场次",
                "有哪些场次",
                "还有什么场次",
                "其他场次",
                "其它场次",
                "别的场次",
                "换个时间",
                "换时间",
                "其他时间",
                "其它时间",
                "别的时间",
                "还有什么时间",
                "还有哪些时间",
            ]
        )
        wants_type_relax = any(
            phrase in normalized
            for phrase in [
                "换个类型",
                "换类型",
                "其他类型",
                "其它类型",
                "别的类型",
                "换个厅",
                "换厅",
                "普通厅",
                "普通影厅",
                "普通场",
            ]
        )
        new_time = self._extract_time_range(text)
        new_hall_type = self._extract_hall_type(text)
        if not (wants_time_relax or wants_type_relax or new_time or new_hall_type):
            return None

        clear_slots = [
            "showtimeId",
            "seatIds",
            "seatPositions",
            "orderId",
            "lockId",
            "couponId",
            "snackIds",
            "snackItems",
            "snackRequests",
            "price",
            "amount",
            "status",
            "expiresAt",
        ]
        if wants_time_relax and not new_time:
            clear_slots.extend(["timeRange", "time", "startAt", "endAt", "timePreference"])
        if wants_type_relax and not new_hall_type:
            clear_slots.extend(["hallType", "notHallType"])

        slots: dict[str, Any] = {"__clearSlots": clear_slots}
        skip_keys = set(clear_slots) | {"__clearSlots"}
        for key in [
            "movieId",
            "movieName",
            "genre",
            "date",
            "timeRange",
            "ticketCount",
            "cinemaId",
            "cinemaName",
            "hallType",
            "notHallType",
            "maxPrice",
            "pricePreference",
            "seatPreference",
            "seatType",
            "autoSelectShowtime",
            "autoSelectSeats",
            "skipSnacks",
            "location",
            "nearbyFirst",
        ]:
            if key not in skip_keys and state.slots.get(key) not in [None, ""]:
                slots[key] = state.slots[key]

        date_value = self._extract_date(text)
        if date_value:
            slots["date"] = date_value
        if new_time:
            slots["timeRange"] = new_time
            slots.pop("timePreference", None)
        ticket_count = self._extract_ticket_count(text)
        if ticket_count is not None:
            slots["ticketCount"] = ticket_count
        cinema_name = self._extract_cinema_name(text)
        if cinema_name:
            slots["cinemaName"] = cinema_name
        if new_hall_type:
            slots["hallType"] = new_hall_type
        slots.update(self._extract_seat_preference_slots(text))
        return slots

    @staticmethod
    def _is_no_showtime_context(state: AgentState) -> bool:
        last_bot_message = _normalize_short_text(state.last_bot_message or "")
        if any(
            phrase in last_bot_message
            for phrase in [
                "没有找到合适的场次",
                "没找到合适的场次",
                "暂无合适场次",
                "没有可选场次",
                "暂无可选场次",
            ]
        ):
            return True
        if state.pending_action != "search_showtimes":
            return False
        candidates = state.selected.get("showtime_candidates")
        return isinstance(candidates, list) and not candidates

    @staticmethod
    def _is_other_cinema_showtime_followup(text: str, state: AgentState) -> bool:
        if state.slots.get("movieId") in [None, ""] and state.slots.get("movieName") in [None, ""]:
            return False
        normalized = _normalize_short_text(text)
        has_other_cinema = any(
            phrase in normalized
            for phrase in [
                "其他影院",
                "其它影院",
                "别的影院",
                "别家影院",
                "其他电影院",
                "其它电影院",
                "别的电影院",
                "其他影城",
                "其它影城",
                "别的影城",
                "换一家影院",
                "换个影院",
                "换家影院",
            ]
        )
        if not has_other_cinema:
            return False
        return any(
            marker in normalized
            for marker in ["场次", "有场", "有吗", "有没有", "还有吗", "呢", "换"]
        )

    @staticmethod
    def _other_cinema_showtime_slots(state: AgentState) -> dict[str, Any]:
        slots: dict[str, Any] = {
            "__clearSlots": [
                "cinemaId",
                "cinemaName",
                "showtimeId",
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "snackItems",
                "snackRequests",
                "price",
                "amount",
                "status",
                "expiresAt",
            ],
        }
        for key in [
            "movieId",
            "movieName",
            "date",
            "timeRange",
            "ticketCount",
            "hallType",
            "maxPrice",
            "seatPreference",
            "seatType",
            "autoSelectSeats",
            "location",
            "nearbyFirst",
        ]:
            if state.slots.get(key) not in [None, ""]:
                slots[key] = state.slots[key]
        return slots

    @staticmethod
    def _is_all_seats_request(text: str) -> bool:
        normalized = _normalize_short_text(text)
        if any(
            phrase in normalized
            for phrase in [
                "包场",
                "全包",
                "全场",
                "整场",
                "全部买下",
                "全买了",
                "都买了",
                "剩下都要",
                "剩余都要",
                "余座都要",
                "所有座位",
                "全部座位",
                "座位全要",
            ]
        ):
            return True
        return bool(
            re.search(
                r"(?:剩下|剩余|所有|全部|全场|整场).{0,4}(?:座位|票|座).{0,4}(?:都要|全要|买)",
                normalized,
            )
        )

    @staticmethod
    def _all_seats_request_slots(state: AgentState | None) -> dict[str, Any]:
        slots = {
            **_showtime_candidate_context(state),
            "autoSelectSeats": True,
            "seatPreference": "all",
            "__clearSlots": [
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "snackItems",
                "snackRequests",
                "price",
                "amount",
                "status",
                "expiresAt",
            ],
        }
        return slots

    def _extract_seat_preference_slots(self, text: str) -> dict[str, Any]:
        positions = self._extract_seat_positions(text)
        if positions:
            return {"seatPositions": positions}

        normalized = _normalize_short_text(text)
        if any(phrase in normalized for phrase in ["前排", "靠前", "前面"]):
            return {
                "autoSelectSeats": True,
                "seatPreference": "front",
            }
        if any(phrase in normalized for phrase in ["后排", "靠后", "后面"]):
            return {
                "autoSelectSeats": True,
                "seatPreference": "back",
            }
        if any(
            phrase in normalized
            for phrase in [
                "最佳观影区",
                "黄金观影区",
                "最佳观影",
                "黄金位置",
                "好位置",
            ]
        ):
            return {
                "autoSelectSeats": True,
                "seatPreference": "best",
            }
        if any(
            phrase in normalized
            for phrase in ["中间", "靠中间", "居中", "中部", "正中", "中央"]
        ):
            return {
                "autoSelectSeats": True,
                "seatPreference": "middle",
            }
        return {}

    @staticmethod
    def _is_showtime_search_text(text: str) -> bool:
        normalized = _normalize_short_text(text)
        nearby_earliest = (
            any(
                phrase in normalized
                for phrase in [
                    "距离我最近",
                    "离我最近",
                    "最近的影院",
                    "最近影院",
                    "就近影院",
                ]
            )
            and any(
                phrase in normalized
                for phrase in ["最早", "最近一场", "一场", "几点", "什么时候"]
            )
        )
        if nearby_earliest:
            return True
        if not any(
            marker in normalized
            for marker in [
                "场次",
                "放映时间",
                "什么时候有",
                "什么时候能看",
                "几点有",
                "几点能看",
                "一场",
            ]
        ):
            return False
        return any(
            marker in normalized
            for marker in ["找", "查", "看", "有", "最近", "距离", "几点", "什么时候"]
        )

    @staticmethod
    def _is_movie_view_request(text: str) -> bool:
        """Check if text expresses 'want to watch a specific movie'.

        Patterns: 我要看XXX / 我想看XXX / 想看XXX / 要看XXX / 去看XXX
        where XXX is a specific movie name (not a genre, not empty).
        """
        normalized = _normalize_short_text(text)
        return bool(re.search(r"(?:我要看|我想看|想看|要看|去看|去看看)\S", normalized))

    @staticmethod
    def _extract_actor_movie_query(text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        match = re.search(
            r"(?:想看|看|找|查)(?P<actor>[\u4e00-\u9fff·]{2,8}?)"
            r"(?:(?:最近|近期)?(?:正在|当前)?上映(?:的)?|"
            r"(?:主演|参演|出演|演的)(?:的)?)(?:电影|影片)",
            normalized,
        )
        if not match:
            match = re.search(
                r"^(?P<actor>[\u4e00-\u9fff·]{2,8}?)"
                r"(?:最近|近期)?(?:有)?(?:正在|当前)?上映(?:的)?(?:电影|影片)(?:吗|么|嘛)?$",
                normalized,
            )
        if not match:
            return ""
        actor = match.group("actor").strip("的《》【】")
        return actor if actor not in {"电影", "影片", "最近"} else ""

    def _extract_recommendation_slots(self, text: str) -> dict[str, Any]:
        normalized = _normalize_short_text(text)
        has_movie_word = any(word in normalized for word in ["电影", "影片", "片"])
        is_couple = any(word in normalized for word in ["情侣", "约会", "恋人", "对象"])
        is_family = any(
            word in normalized
            for word in ["亲子", "带孩子", "小朋友", "儿童", "全家", "一家人"]
        )
        has_recommendation_word = any(
            word in normalized
            for word in ["推荐", "高分", "好看", "热映", "热门", "火", "火爆", "票房"]
        )
        genre = self._extract_genre(text)
        is_genre_browse = bool(genre) and any(
            word in normalized
            for word in ["想看", "看看", "电影", "影片", "片", "类型", "推荐"]
        ) and not any(
            word in normalized
            for word in ["买", "订", "购", "张", "票"]
        )
        is_suitability_request = (
            has_movie_word
            and "适合" in normalized
            and any(word in normalized for word in ["看", "观影"])
        )
        if not (
            has_movie_word
            or is_couple
            or is_family
            or has_recommendation_word
            or is_suitability_request
            or is_genre_browse
        ):
            return {}
        is_general_browse = any(
            phrase in normalized
            for phrase in [
                "有什么电影",
                "有哪些电影",
                "什么电影可以看",
                "电影可以看",
                "电影能看",
                "\u6211\u60f3\u770b\u7535\u5f71",
                "\u60f3\u770b\u7535\u5f71",
                "\u60f3\u770b\u4e00\u90e8\u7535\u5f71",
                "\u7ed9\u6211\u63a8\u8350\u7535\u5f71",
                "正在上映",
                "热映电影",
            ]
        )
        if not (
            is_general_browse
            or is_couple
            or is_family
            or has_recommendation_word
            or is_suitability_request
            or is_genre_browse
        ):
            return {}
        slots: dict[str, Any] = {
            "movieLimit": self._extract_recommendation_movie_limit(text) or 3,
            "__clearSlots": [
                "movieId",
                "movieName",
                "genre",
                "date",
                "time",
                "timeRange",
                "timePreference",
                "ticketCount",
                "cinemaId",
                "cinemaName",
                "hallType",
                "maxPrice",
                "pricePreference",
                "recommendationCriteria",
                "showtimeId",
                "seatIds",
                "seatPositions",
                "seatPreference",
                "seatType",
                "orderId",
                "lockId",
                "snackIds",
                "snackItems",
                "snackRequests",
                "autoSelectShowtime",
                "autoSelectSeats",
                "skipSnacks",
            ],
        }
        date_value = self._extract_date(text)
        if date_value:
            slots["date"] = date_value
        if genre:
            slots["genre"] = genre
        if is_genre_browse:
            pass
        elif is_couple:
            slots["recommendationCriteria"] = "couple"
        elif is_family:
            slots["recommendationCriteria"] = "family"
        elif any(word in normalized for word in ["高分", "评分高", "口碑"]):
            slots["recommendationCriteria"] = "high_rating"
        elif any(word in normalized for word in ["热映", "热门", "火"]):
            slots["recommendationCriteria"] = "hot"
        elif is_general_browse:
            slots["recommendationCriteria"] = "hot"
        else:
            slots["recommendationCriteria"] = "high_rating"
        if any(word in normalized for word in ["还能看", "可看", "今天还可观看", "今天能看", "今天有什么电影", "今天有哪些电影"]):
            slots.setdefault("date", "today")
        return slots

    def _extract_recommendation_movie_limit(self, text: str) -> int | None:
        """Extract an explicit movie count from recommendation wording."""
        normalized = _normalize_short_text(text)
        match = re.search(
            r"(?P<count>[0-9一二两三四五六七八九十]+)部(?:电影|影片|片)?",
            normalized,
        )
        if not match:
            return None
        count = self._parse_number_token(match.group("count"))
        if count is None or count <= 0:
            return None
        return min(count, 10)

    def _extract_showtime_query_movie_name(
        self,
        text: str,
        cinema_name: str | None = None,
    ) -> str:
        normalized = re.sub(r"\s+", "", text)
        if cinema_name:
            normalized = normalized.replace(cinema_name, "")
        quoted = re.search(r"[《【](?P<name>[^》】]+)[》】]", normalized)
        if quoted:
            return quoted.group("name").strip()

        watch_context = re.search(
            r".*(?:看|观看)(?P<name>[^，,。；;]+?)(?:的)?(?:场次|排片)",
            normalized,
        )
        if watch_context:
            candidate = watch_context.group("name").strip("的《》<>【】[]()，,。.!?！？：: ")
            if candidate:
                return candidate

        direct = re.search(
            r"(?P<name>.+?)"
            r"(?:今天|今晚|明天|明晚|后天|周末)?"
            r"(?:上午|早上|中午|下午|晚上)?"
            r"[0-9一二两三四五六七八九十]{0,3}(?:[:：点])?"
            r"[0-9一二两三四五六七八九十]{0,2}(?:半)?"
            r"(?:有|有没有|有哪些|有什么)?"
            r"(?:场次|放映时间)(?:吗|么|嘛|啊)?$",
            normalized,
        )
        nearby_wording = any(
            phrase in normalized
            for phrase in [
                "距离我最近",
                "离我最近",
                "最近的影院",
                "最近影院",
                "附近最近",
                "就近影院",
            ]
        )
        if direct and not nearby_wording:
            candidate = direct.group("name")
        else:
            candidate = ""

        nearby_context = re.search(
            r"(?:距离(?:我)?|离我|就近).{0,8}?(?:电影院|影院)"
            r"(?:里|中|，|,|的)?(?P<name>.+?)"
            r"(?:电影|影片)?(?:的)?(?:最近)?(?:最早)?(?:一场)?"
            r"(?:场次|放映时间|什么时候(?:有|能看)?|几点(?:有|能看)?)$",
            normalized,
        )
        if not nearby_context:
            nearby_context = re.search(
                r"(?:附近|周边)?(?:最近的?|就近)?(?:电影院|影院|影城)"
                r"(?:里|中|，|,|的)?(?P<name>.+?)"
                r"(?:电影|影片)?(?:的)?(?:最近)?(?:最早)?(?:一场)?"
                r"(?:场次|放映时间|什么时候(?:有|能看)?|几点(?:有|能看)?)$",
                normalized,
            )
        if not candidate and nearby_context:
            candidate = nearby_context.group("name")
        elif not candidate:
            nearby_earliest = re.search(
                r"(?:最早一场|最近一场|最早)(?:的)?(?P<name>[^，。！？,.!?]+)$",
                normalized,
            )
            if nearby_earliest:
                candidate = nearby_earliest.group("name")
            else:
                trailing = re.search(
                    r"(?:场次|放映时间)(?:的)?(?P<name>[^，。！？,.!?]+)$",
                    normalized,
                )
                candidate = trailing.group("name") if trailing else ""
            if not candidate:
                one_showtime = re.search(
                    r"一场(?:的)?(?P<name>[^，。！？,.!?]+)$",
                    normalized,
                )
                if one_showtime:
                    candidate = one_showtime.group("name")
                else:
                    leading = re.search(
                        r"(?P<name>.+?)(?:电影|影片)?(?:有哪些|有什么|有|的)?"
                        r"(?:场次|放映时间|什么时候(?:有|能看)?|几点(?:有|能看)?)",
                        normalized,
                    )
                    candidate = leading.group("name") if leading else ""

        candidate = re.sub(
            r"^(?:那|然后|接着|继续)?"
            r"(?:给我|帮我|我要|我想|想要|请|麻烦)?"
            r"(?:找|查|看|要看|想看)"
            r"(?:一个|一下|下)?(?:距离我)?(?:最近)?(?:的)?",
            "",
            candidate,
        )
        candidate = re.sub(
            r"^(?:(?:今天|今晚|明天|明晚|后天|周末|最早|最近|一场|的|里|中)+)",
            "",
            candidate,
        )
        candidate = re.sub(
            r"^(?:\d{1,2}|[一二两三四五六七八九十]{1,3})月"
            r"(?:\d{1,2}|[一二两三四五六七八九十]{1,3})(?:号|日)?(?:的)?",
            "",
            candidate,
        )
        candidate = re.sub(
            r"(?:的)?(?:场次|放映时间|什么时候(?:有|能看)?|几点(?:有|能看)?)+$",
            "",
            candidate,
        )
        candidate = re.sub(r"(?:有哪些|有什么|有)$", "", candidate)
        candidate = re.sub(r"(?:最近|最早|一场|的)+$", "", candidate)
        candidate = self._clean_movie_candidate(candidate)
        if cinema_name:
            candidate = self._clean_movie_candidate(candidate.replace(cinema_name, ""))
        candidate = candidate.strip("的《》【】,，。！？!? ")
        if self._is_invalid_showtime_movie_name(candidate):
            return ""
        return candidate

    @staticmethod
    def _is_invalid_showtime_movie_name(candidate: Any) -> bool:
        normalized = re.sub(r"\s+", "", str(candidate or ""))
        invalid_candidates = {
            "",
            "电影",
            "影片",
            "场次",
            "最近",
            "附近",
            "离我最近",
            "离我近",
            "距离我最近",
            "最近的影院",
            "最近影院",
            "就近影院",
        }
        return normalized in invalid_candidates

    def _is_booking_request_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if self._wants_auto_showtime(text) and (
            is_text_auto_seat_or_auto_purchase(text) or self._wants_direct_payment(text)
        ):
            return True
        if any(
            phrase in normalized
            for phrase in [
                "bookmovietickets",
                "bookticket",
                "buyticket",
                "订票",
                "买票",
                "买电影票",
                "帮我订",
                "帮我买",
                "给我买",
                "给我订",
                "我要订票",
                "我要买票",
            ]
        ):
            return True
        if re.search(r"(?:给我|帮我|我要|我想|想要|请|麻烦)?(?:买|订|预订|来|要).{0,50}(?:电影票|影票)", normalized):
            return True

        has_date = any(marker in normalized for marker in ["今天", "今晚", "明天", "明晚", "后天", "周末"])
        has_ticket_count = bool(
            re.search(r"[0-9一二两三四五六七八九十]+张", normalized)
        )
        has_genre = bool(canonical_genre_from_text(normalized))
        return has_date and has_ticket_count and has_genre

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
