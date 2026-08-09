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
            actor = self._extract_actor_movie_query(text)
            if actor:
                return NLUResult(
                    intent="search_movies",
                    confidence=0.94,
                    intent_source="rule",
                    slots={"actor": actor},
                    reference_text=text,
                )
            if self._is_showtime_search_text(text):
                slots = self._extract_booking_slots(text, payload)
                self._apply_nearby_showtime_preferences(text, slots)
                movie_name = self._extract_showtime_query_movie_name(text)
                if movie_name:
                    slots["movieName"] = movie_name
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
            if self._is_booking_request_text(text):
                return NLUResult(
                    intent="book_ticket",
                    confidence=0.80,
                    intent_source="rule",
                    slots=self._extract_booking_slots(text, payload),
                    reference_text=text,
                )
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

        movie_name = self._extract_movie_name(text, snack_requests)
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
        if "后天" in text:
            return "after_tomorrow"
        if "明天" in text or "明晚" in text:
            return "tomorrow"
        if "今天" in text or "今晚" in text:
            return "today"
        if "周末" in text:
            return "weekend"
        return None

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

    def _extract_snack_requests(self, text: str) -> list[dict[str, Any]]:
        normalized = re.sub(r"\s+", "", text)
        snack_names = r"爆米花|可乐|薯条|热狗|饮料|套餐"
        connector = re.search(
            rf"(?:和|加|再加|再来|以及|外加)"
            rf"(?=[0-9一二两三四五六七八九十]*"
            rf"(?:桶|瓶|杯|份|个|盒)?(?:{snack_names}))",
            normalized,
        )
        if not connector:
            return []

        tail = normalized[connector.end():]
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
    ) -> str:
        cleaned = re.sub(r"\s+", "", text)
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
        cleaned = re.sub(r"[0-9一二两三四五六七八九十]+张(?:电影票|影票|票)?", "", cleaned)
        cleaned = re.sub(r"(今天|今晚|明天|明晚|后天|周末)", "", cleaned)
        cleaned = re.sub(
            r"(上午|早上|中午|下午|晚上)?[0-9一二两三四五六七八九十]{1,3}[:：点][0-9一二两三四五六七八九十]{0,2}(?:半)?(?:左右|前后|以后|之后|之前|前)?的?",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"^(给我|帮我|我要|我想|想要|请帮我|麻烦帮我|买|订|预订|来|看|想看|去看)+",
            "",
            cleaned,
        )
        cleaned = cleaned.strip("的《》<>【】[]()，,。.!?！？：:")
        cleaned = re.sub(r"(电影票|影票|电影|影片|票)+$", "", cleaned)
        cleaned = cleaned.strip("的《》<>【】[]()，,。.!?！？：:")
        if cleaned in {"", "片", "部片", "一部片", "喜剧", "爱情", "动作", "科幻", "动画", "悬疑", "恐怖"}:
            return ""
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
            cleaned = self._strip_movie_segment(segment)
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
        cleaned = re.sub(
            r"^(给我|帮我|我要|我想|想要|买|订|预订|来|一张|两张|三张|四张|五张|六张|七张|八张|九张|十张|今天|明天|今晚|明晚|后天|上午|下午|晚上|的)+",
            "",
            cleaned,
        )
        cleaned = re.sub(r"(电影票|影票|电影|票|的)+$", "", cleaned)
        cleaned = cleaned.strip("《》<>【】[]()，,。.!?：:")
        return cleaned

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
            r"^(?:给我|帮我|我要|我想|请|麻烦)?(?:找|查|看)"
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
