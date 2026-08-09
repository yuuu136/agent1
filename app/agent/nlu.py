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

from app.agent.intent_rag import IntentMatch, IntentRAGUnavailable, get_intent_rag_retriever
from app.prompts import prompt_manager
from app.schemas.agent import AgentState, ChatRequest, NLUResult
from app.utils.config_handler import agent_config


logger = logging.getLogger(__name__)


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
            if is_skip_snacks_text(text):
                return NLUResult(
                    intent="skip_snacks",
                    confidence=0.95,
                    intent_source="rule",
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
                return NLUResult(
                    intent="nearby_cinema",
                    confidence=0.95,
                    intent_source="rule",
                    slots=slots,
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
                movie_name = self._extract_showtime_query_movie_name(text)
                if movie_name:
                    slots["movieName"] = movie_name
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
                    "seatPreference", "seatPositions", "snackRequests",
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

        # ── Intent RAG path: semantic examples participate before the LLM ──
        rag_match: IntentMatch | None = None
        rag_error: str | None = None
        if request.event is None and text.strip():
            try:
                rag_match = get_intent_rag_retriever().retrieve(text)
            except IntentRAGUnavailable as exc:
                rag_error = str(exc)
                logger.warning("Intent RAG unavailable for NLU: %s", exc)
            except Exception as exc:
                rag_error = str(exc)
                logger.exception("Intent RAG failed unexpectedly")

            rag_result = self._result_from_rag_match(text, payload, rag_match)
            if rag_result is not None:
                return rag_result

        # ── LLM path ──
        return self._llm_extract(text, payload, state, rag_match, rag_error)

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
                return NLUResult(intent="book_ticket", confidence=0.90, intent_source="rule")
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

    def _llm_extract(
        self,
        text: str,
        payload: dict[str, Any],
        state: AgentState | None = None,
        rag_match: IntentMatch | None = None,
        rag_error: str | None = None,
    ) -> NLUResult:
        client = _llm_client()
        if client is None:
            return NLUResult(intent="smalltalk", confidence=0.10, intent_source="llm")

        nlu_settings = agent_config.get("nlu", {})
        llm_settings = agent_config.get("llm", {})

        user_message = f"用户输入：{text}"
        if payload:
            user_message += f"\n前端携带数据：{json.dumps(payload, ensure_ascii=False)}"

        context = self._build_context(state)
        if context:
            user_message += f"\n\n当前会话上下文：{context}"
        if rag_match is not None:
            user_message += (
                "\n\n意图RAG候选："
                f"intent={rag_match.intent}, score={rag_match.score:.3f}, "
                f"example={rag_match.example}"
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
            if rag_match is not None:
                rag_result = self._result_from_rag_match(text, payload, rag_match)
                if rag_result is not None:
                    return rag_result
            if self._is_booking_request_text(text):
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.80,
                    intent_source="rule",
                    slots=self._extract_booking_slots(text, payload),
                    reference_text=text,
                )
            return NLUResult(
                intent="smalltalk",
                confidence=0.10,
                intent_source="llm",
                slots={"nluError": str(exc)},
                reference_text=text,
            )

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

        seat_positions = self._extract_seat_positions(text)
        if seat_positions:
            llm_slots["seatPositions"] = seat_positions
            if intent in {"smalltalk", "search_movies"}:
                intent = "book_ticket"
                confidence = max(confidence, 0.95)

        multi_movie_names = self._extract_multi_movie_names(text)
        if len(multi_movie_names) > 1:
            llm_slots["movieNames"] = multi_movie_names
            if intent in {"smalltalk", "search_movies", "book_ticket"}:
                intent = "multi_movie_booking"
                confidence = max(confidence, 0.92)

        return NLUResult(
            intent=intent,
            confidence=confidence,
            intent_source="llm",
            slots=llm_slots,
            is_modification=bool(parsed.get("is_modification")),
            reference_text=str(parsed.get("reference") or text),
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

        genre = self._extract_genre(text)
        if genre:
            slots["genre"] = genre

        snack_requests = self._extract_snack_requests(text)
        if snack_requests:
            slots["snackRequests"] = snack_requests

        cinema_name = self._extract_cinema_name(text)
        if cinema_name:
            slots["cinemaName"] = cinema_name

        movie_name = self._extract_movie_name(text, snack_requests, cinema_name)
        if movie_name:
            slots["movieName"] = movie_name

        return slots

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

    def _extract_ticket_count(self, text: str) -> int | None:
        normalized = re.sub(r"\s+", "", text)
        match = re.search(r"(?P<count>[0-9一二两三四五六七八九十]+)\s*(?:张|份|个)?(?:电影票|影票|票)", normalized)
        if not match:
            match = re.search(
                r"(?P<count>[0-9一二两三四五六七八九十]+)张"
                r"(?=(?:的)?(?:电影|影片|片|喜剧|爱情|动作|科幻|动画|悬疑|恐怖))",
                normalized,
            )
        if not match:
            match = re.search(r"(?:买|订|要|来)(?P<count>[0-9一二两三四五六七八九十]+)张", normalized)
        if not match:
            return None
        count = self._parse_number_token(match.group("count"))
        return count if count and count > 0 else None

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
        normalized = re.sub(r"\s+", "", text)
        period_match = re.search(r"(上午|早上|中午|下午|晚上|今晚|明晚)?([0-9一二两三四五六七八九十]{1,3})(?:[:：点])([0-9一二两三四五六七八九十]{0,2})?(半)?", normalized)
        if period_match:
            period = period_match.group(1) or ""
            hour = self._parse_number_token(period_match.group(2))
            minute_text = period_match.group(3)
            minute = self._parse_number_token(minute_text) if minute_text else 0
            if period_match.group(4):
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
        for genre in ["喜剧", "爱情", "动作", "科幻", "动画", "悬疑", "恐怖"]:
            if genre in text:
                return genre
        return None

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
        marker_match = re.search(r"(?P<name>.+?)(?:电影票|影票)", cleaned)
        if marker_match:
            candidate = self._clean_movie_candidate(marker_match.group("name"))
            if cinema_name:
                candidate = self._clean_movie_candidate(candidate.replace(cinema_name, ""))
            if candidate:
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
        if cleaned in {"", "片", "部片", "一部片", "喜剧", "爱情", "动作", "科幻", "动画", "悬疑", "恐怖"}:
            return ""
        return cleaned

    def _extract_cinema_name(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        suffix = r"(?:电影院|影城|影院)"
        branch = r"(?:[（(][^）)]{1,20}[）)])?"
        known_brands = (
            "CGV|万达|大地|奥斯卡|博纳|金逸|中影|横店|UME|卢米埃|"
            "保利|橙天嘉禾|星美|幸福蓝海"
        )
        patterns = (
            re.compile(
                rf"(?P<name>(?:{known_brands})"
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
        cleaned = re.sub(r"^(给我|帮我|我要|我想|想要|请帮我|麻烦帮我|买|订|预订|来|看|想看|去看)+", "", cleaned)
        cleaned = cleaned.strip("的《》<>【】[]，,。.!?！？：:")
        return cleaned

    def _extract_movie_search_keyword(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        quoted = re.search(r"[《【](?P<name>[^》】]+)[》】]", normalized)
        if quoted:
            return quoted.group("name").strip()

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
        return any(
            phrase in normalized
            for phrase in ["附近有什么影院", "附近影院", "周边影院", "附近影城", "nearbycinema", "附近有啥影院"]
        )

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
                "找一个",
                "一场",
            ]
        )
        if wants_one_earliest:
            slots["timePreference"] = "earliest"
            slots["showtimeLimit"] = 1

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
    def _extract_actor_movie_query(text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        match = re.search(
            r"(?:想看|看|找|查)(?P<actor>[\u4e00-\u9fff·]{2,8}?)"
            r"(?:(?:最近|近期)?(?:正在|当前)?上映(?:的)?|"
            r"(?:主演|参演|出演|演的)(?:的)?)(?:电影|影片)",
            normalized,
        )
        if not match:
            return ""
        actor = match.group("actor").strip("的《》【】")
        return actor if actor not in {"电影", "影片", "最近"} else ""

    def _extract_recommendation_slots(self, text: str) -> dict[str, Any]:
        normalized = _normalize_short_text(text)
        if not any(word in normalized for word in ["推荐", "高分", "好看", "热映", "热门"]):
            return {}
        if not any(word in normalized for word in ["电影", "影片", "片"]):
            return {}
        slots: dict[str, Any] = {
            "movieLimit": 3,
            "__clearSlots": [
                "movieId",
                "movieName",
                "genre",
                "showtimeId",
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "snackIds",
                "snackItems",
                "snackRequests",
            ],
        }
        date_value = self._extract_date(text)
        if date_value:
            slots["date"] = date_value
        if any(word in normalized for word in ["高分", "评分高", "口碑"]):
            slots["recommendationCriteria"] = "high_rating"
        elif any(word in normalized for word in ["热映", "热门", "火"]):
            slots["recommendationCriteria"] = "hot"
        else:
            slots["recommendationCriteria"] = "high_rating"
        if any(word in normalized for word in ["还能看", "可看", "今天还可观看", "今天能看"]):
            slots.setdefault("date", "today")
        return slots

    @staticmethod
    def _extract_showtime_query_movie_name(text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        quoted = re.search(r"[《【](?P<name>[^》】]+)[》】]", normalized)
        if quoted:
            return quoted.group("name").strip()

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
        if nearby_context:
            candidate = nearby_context.group("name")
        else:
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
        candidate = re.sub(r"(?:最近|最早|一场|的)+$", "", candidate)
        candidate = candidate.strip("的《》【】,，。！？!? ")
        if candidate in {"", "电影", "影片", "场次", "最近", "附近"}:
            return ""
        return candidate

    def _is_booking_request_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
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
        has_genre = any(
            genre in normalized
            for genre in ["喜剧", "爱情", "动作", "科幻", "动画", "悬疑", "恐怖"]
        )
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
