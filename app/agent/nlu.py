import re
from typing import Any

from app.schemas.agent import ChatRequest, NLUResult


CHINESE_NUMBERS = {
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

CHINESE_HOURS = {
    **CHINESE_NUMBERS,
    "十一": 11,
    "十二": 12,
}

ACK_TEXTS = {
    "好",
    "好的",
    "好吧",
    "行",
    "行吧",
    "可以",
    "嗯",
    "哦",
    "知道了",
}
NON_MOVIE_TEXTS = ACK_TEXTS | {
    "选择影院",
    "选影院",
    "选择电影",
    "选电影",
    "选择这场",
    "确认座位",
    "确认支付",
    "取消",
}
MOVIE_SEARCH_TEXTS = {"最近热映", "正在上映", "有什么电影", "推荐电影"}
PRICE_QUERY_MARKERS = (
    "多少钱",
    "多少元",
    "票价",
    "价格",
    "价位",
    "什么价格",
    "什么价",
    "单价",
    "费用",
    "贵不贵",
)


class RuleBasedNLU:
    def extract(self, request: ChatRequest) -> NLUResult:
        text = request.text or ""
        payload = request.payload or {}
        intent = self._detect_intent(text, request.event, payload)
        slots = self._extract_slots(text, payload, request.event, intent)
        if (
            intent == "smalltalk"
            and slots.get("movieName")
            and self._looks_like_movie_title(text)
        ):
            intent = "book_ticket"

        return NLUResult(
            intent=intent,
            confidence=0.72 if intent != "smalltalk" else 0.45,
            slots=slots,
            is_modification=self._is_modification(text),
            reference_text=self._extract_reference(text, payload),
        )

    def _detect_intent(
        self,
        text: str,
        event: str | None,
        payload: dict[str, Any],
    ) -> str:
        if event:
            event_intent = {
                "select_movie": "select_or_modify",
                "select_cinema": "select_or_modify",
                "select_showtime": "select_showtime",
                "navigate": "select_showtime",
                "select_seats": "confirm_order",
                "select_snacks": "select_snacks",
                "select_coupon": "select_coupon",
                "confirm_order": "confirm_order",
                "pay_order": "pay_order",
            }.get(event)
            if event_intent:
                return event_intent

        lowered = text.lower()
        if text.strip() in ACK_TEXTS:
            return "smalltalk"
        if text.strip() in {"取消", "不用了", "算了", "先不买", "不买了"}:
            return "cancel"
        if any(word in text for word in ["退票", "改签", "规则", "政策", "怎么处理", "FAQ"]):
            return "faq"
        if any(marker in text for marker in PRICE_QUERY_MARKERS):
            return "price_query"
        if any(word in text for word in ["热映", "正在上映", "有什么电影", "推荐电影"]):
            return "search_movies"
        if any(word in text for word in ["附近", "最近", "周边", "离我近", "高德", "地图"]):
            return "nearby_cinema"
        if any(word in lowered for word in ["nearby", "around me", "map", "amap"]):
            return "nearby_cinema"
        payload_slots = payload.get("slots", {})
        if any(
            key in payload or key in payload_slots
            for key in ["movieName", "movieId", "genre", "showtimeId", "cinemaId"]
        ):
            return "book_ticket"
        if any(word in text for word in ["零食", "爆米花", "饮料", "套餐", "小吃"]):
            return "snack"
        if any(word in text for word in ["优惠", "优惠券", "券", "折扣", "便宜"]):
            return "coupon" if "券" in text or "优惠" in text else "select_or_modify"
        if any(word in text for word in ["座位", "选座", "靠中", "中间", "前排", "后排"]):
            return "select_or_modify" if payload else "seat_query"
        if any(word in text for word in ["支付", "付款", "出票"]):
            return "pay_order"
        if any(word in text for word in ["订单", "确认", "就这个", "就这场", "可以"]):
            return "confirm_order"
        if any(word in text for word in ["买票", "订票", "购票", "影票", "电影票", "电影", "场次", "影院", "看", "买"]):
            return "book_ticket"
        if any(word in lowered for word in ["book", "ticket", "movie", "showtime", "cinema"]):
            return "book_ticket"
        if "rag" in lowered:
            return "faq"
        return "smalltalk"

    def _extract_slots(
        self,
        text: str,
        payload: dict[str, Any],
        event: str | None = None,
        intent: str = "smalltalk",
    ) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        slots.update(payload.get("slots", {}))

        ticket_count = self._extract_ticket_count(text)
        if ticket_count:
            slots["ticketCount"] = ticket_count

        time_range = self._extract_time(text)
        if time_range:
            slots["timeRange"] = time_range

        date = self._extract_date(text)
        if date:
            slots["date"] = date

        genre = self._extract_genre(text)
        if genre:
            slots["genre"] = genre

        seat_preference = self._extract_seat_preference(text)
        if seat_preference:
            slots["seatPreference"] = seat_preference

        hall_type = self._extract_hall_type(text)
        if hall_type:
            slots["hallType"] = hall_type

        movie_name = payload.get("movieName") or payload.get("movie_name")
        if movie_name:
            slots["movieName"] = movie_name
        elif (
            not genre
            and text.strip() not in MOVIE_SEARCH_TEXTS
            and self._should_extract_movie_name(event, intent, text)
        ):
            extracted_movie = self._extract_movie_name(text)
            if extracted_movie:
                slots["movieName"] = extracted_movie

        for key in [
            "showtimeId",
            "seatIds",
            "cinemaId",
            "movieId",
            "orderId",
            "couponId",
            "snackIds",
            "snackId",
            "location",
            "city",
            "cinemaName",
            "phone",
            "phoneNumber",
            "hallType",
            "price",
            "basePrice",
            "unitPrice",
            "date",
            "time",
            "hallName",
            "startAt",
            "endAt",
        ]:
            if key in payload:
                slots[key] = payload[key]

        clear_slots = self._clear_slots_for_event(event)
        if clear_slots:
            slots["__clearSlots"] = clear_slots

        if event == "select_snacks" and slots.get("snackId"):
            slots["snackIds"] = [slots["snackId"]]
        if event == "select_coupon" and slots.get("couponId"):
            slots["couponId"] = slots["couponId"]

        return slots

    def _should_extract_movie_name(self, event: str | None, intent: str, text: str) -> bool:
        return (
            event in {None, "", "text"}
            and intent
            not in {
                "nearby_cinema",
                "snack",
                "coupon",
                "pay_order",
                "faq",
                "price_query",
            }
            and not self._is_modification(text)
        )

    def _looks_like_movie_title(self, text: str) -> bool:
        normalized = text.strip()
        if normalized in ACK_TEXTS or normalized in NON_MOVIE_TEXTS:
            return False
        if any(
            marker in normalized
            for marker in (
                "附近",
                "影院",
                "座位",
                "多少钱",
                "价位",
                "价格",
                "优惠",
                "规则",
            )
        ):
            return False
        return bool(
            re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9·\s]{2,40}", normalized)
        )

    def _clear_slots_for_event(self, event: str | None) -> list[str]:
        if event == "select_cinema":
            return [
                "showtimeId",
                "seatIds",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
            ]
        if event == "select_movie":
            return [
                "showtimeId",
                "seatIds",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
            ]
        if event == "select_showtime":
            return ["seatIds", "orderId", "lockId", "couponId", "snackIds", "price"]
        return []

    def _extract_ticket_count(self, text: str) -> int | None:
        match = re.search(r"(\d+)\s*[张人位]", text)
        if match:
            return int(match.group(1))
        for char, value in CHINESE_NUMBERS.items():
            if f"{char}张" in text or f"{char}个人" in text or f"{char}位" in text:
                return value
        return None

    def _extract_time(self, text: str) -> str | None:
        if "晚一点" in text or "早一点" in text:
            return None
        match = re.search(r"(\d{1,2})(?:点|:)(\d{1,2})?", text)
        if match:
            hour = int(match.group(1))
            if ("下午" in text or "今晚" in text or "晚上" in text or "晚" in text) and hour < 12:
                hour += 12
            minute = match.group(2) or "00"
            return f"{hour:02d}:{int(minute):02d}"
        chinese_match = re.search(r"([一二两三四五六七八九十]{1,2})点(半)?", text)
        if chinese_match:
            hour = CHINESE_HOURS.get(chinese_match.group(1))
            if hour is not None:
                if ("今晚" in text or "晚上" in text or "晚" in text) and hour < 12:
                    hour += 12
                minute = "30" if chinese_match.group(2) else "00"
                return f"{hour:02d}:{minute}"
        if "今晚" in text or "晚上" in text:
            return "evening"
        if "下午" in text:
            return "afternoon"
        if "上午" in text:
            return "morning"
        return None

    def _extract_date(self, text: str) -> str | None:
        if "今天" in text or "今晚" in text:
            return "today"
        if "明天" in text or "明晚" in text:
            return "tomorrow"
        if "周末" in text:
            return "weekend"
        return None

    def _extract_genre(self, text: str) -> str | None:
        for genre in ["喜剧", "爱情", "动作", "科幻", "动画", "悬疑", "恐怖"]:
            if genre in text:
                return genre
        return None

    def _extract_hall_type(self, text: str) -> str | None:
        upper_text = text.upper()
        if "IMAX" in upper_text:
            return "IMAX"
        if "杜比" in text:
            return "杜比"
        if "巨幕" in text:
            return "巨幕"
        if "激光" in text:
            return "激光"
        return None

    def _extract_movie_name(self, text: str) -> str | None:
        if text.strip() in NON_MOVIE_TEXTS:
            return None
        if self._extract_ordinal_text(text) is not None:
            return None
        value = re.sub(r"^(?:给我|帮我|我想|想|我要|请|麻烦你?)", "", text.strip())
        value = re.sub(r"(?:买|订|预订|购买|来|看|选|找)", "", value)
        value = re.sub(r"\d+\s*[张人位]", "", value)
        value = re.sub(r"[一二两三四五六七八九十]\s*[张人位]", "", value)
        value = re.sub(r"(?:今天|今晚|明天|明晚|晚上|下午|上午)", "", value)
        value = re.sub(r"(?:\d{1,2}|[一二两三四五六七八九十]{1,2})(?:点|:)\d{0,2}(?:半)?", "", value)
        value = re.sub(r"(?:IMAX|imax|杜比|巨幕|激光|场次|场|影厅|厅)", "", value)
        value = re.sub(r"^(?:的|要|一场|一部)+", "", value)
        value = re.sub(r"(?:电影票|影票|电影|票)$", "", value)
        value = value.strip(" ，。,.!?！？的")
        if value in {"喜剧", "爱情", "动作", "科幻", "动画", "悬疑", "恐怖"}:
            return None
        if 2 <= len(value) <= 20:
            return value

        match = re.search(r"(?P<name>[\u4e00-\u9fa5A-Za-z0-9·]{2,20})(?:电影票|影票|电影|票)", text)
        if match:
            return match.group("name").strip("的")
        return None

    def _extract_ordinal_text(self, text: str) -> int | None:
        match = re.search(
            r"第?\s*(\d+|[一二两三四五六七八九十])\s*(?:个|家|场|项)",
            text,
        )
        if not match:
            return None
        value = match.group(1)
        return int(value) if value.isdigit() else CHINESE_NUMBERS.get(value)

    def _extract_seat_preference(self, text: str) -> str | None:
        if any(word in text for word in ["中间", "靠中", "居中"]):
            return "middle"
        if "前排" in text:
            return "front"
        if "后排" in text:
            return "back"
        if "便宜座" in text or "便宜位置" in text:
            return "cheap"
        return None

    def _is_modification(self, text: str) -> bool:
        return any(word in text for word in ["换", "改", "更", "不要这个", "便宜点", "晚一点"])

    def _extract_reference(self, text: str, payload: dict[str, Any]) -> str:
        return str(payload.get("reference") or text)


nlu_engine = RuleBasedNLU()
