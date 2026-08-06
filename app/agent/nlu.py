import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

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
    "俩": 2,
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
    "就好",
    "就行",
    "可以的",
    "行",
    "行吧",
    "可以",
    "嗯",
    "哦",
    "知道了",
    "谢谢",
    "谢谢你",
    "感谢",
    "多谢",
    "辛苦了",
    "没问题",
    "没事",
    "不用谢",
    "收到",
    "明白了",
    "我知道了",
    "很好",
}
GREETING_TEXTS = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "hi",
    "hello",
    "早上好",
    "早安",
    "上午好",
    "中午好",
    "下午好",
    "晚上好",
    "晚安",
    "再见",
    "拜拜",
    "拜拜了",
    "开始",
    "start",
}
CANCEL_TEXTS = {
    "取消",
    "取消订单",
    "不用了",
    "算了",
    "先不买",
    "不买了",
    "别买了",
    "不要了",
    "先不要了",
    "暂时不要了",
}
SHOWTIME_QUERY_TEXTS = {
    "有什么场次",
    "有哪些场次",
    "有些什么场次",
    "场次有哪些",
    "查场次",
    "查看场次",
    "看看场次",
    "查一下场次",
    "有什么时间",
    "有哪些时间",
}
NON_MOVIE_TEXTS = GREETING_TEXTS | ACK_TEXTS | SHOWTIME_QUERY_TEXTS | {
    "选择影院",
    "选影院",
    "选择电影",
    "选电影",
    "选择这场",
    "这个",
    "这场",
    "这家",
    "就这个",
    "就这场",
    "就这家",
    "确认",
    "确认一下",
    "确认订单",
    "确认座位",
    "确认支付",
    "取消",
    "取消订单",
    "不用了",
    "算了",
    "先不买",
    "不买了",
    "别买了",
    "都可以",
    "都行",
    "随便",
    "不限",
    "时间不限",
    "什么时候都可以",
    "哪个时间都可以",
    "无所谓",
    "换一场",
    "换个场次",
    "换到下一场",
    "下一场",
    "再来一场",
    "下一个",
    "换时间",
    "换个时间",
    "早一点",
    "晚一点",
    "便宜点",
    "换便宜点",
    "不要这个",
    "不要这场",
    "重新选座",
    "重新选择座位",
    "换个位置",
    "换座位",
    "更换座位",
}
MOVIE_SEARCH_TEXTS = {
    "最近热映",
    "正在上映",
    "有什么电影",
    "有啥电影",
    "有哪些电影",
    "有些什么电影",
    "有什么影片",
    "有啥影片",
    "有哪些影片",
    "有些什么影片",
    "推荐电影",
    "推荐影片",
    "看看电影",
    "查电影",
}
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
ORDER_QUERY_PHRASES = (
    "查看订单",
    "查订单",
    "查询订单",
    "看看订单",
    "我的订单",
    "订单详情",
    "订单记录",
    "历史订单",
    "付款了吗",
    "支付了吗",
    "付钱了吗",
    "支付状态",
    "付款状态",
    "支付结果",
    "付款结果",
)
LOCATION_QUERY_PHRASES = (
    "我的地理位置",
    "我现在的具体位置",
    "我的具体位置",
    "我现在的位置",
    "我的当前位置",
    "当前位置",
    "当前定位",
    "定位信息",
    "我的位置",
    "现在在哪里",
    "现在在哪",
    "现在在哪儿",
    "当前位置在哪里",
    "当前位置在哪",
    "当前位置在哪儿",
    "这里是哪里",
    "这里在哪",
    "这里在哪儿",
    "这是哪里",
    "这是哪儿",
    "我在哪里",
    "我在哪",
    "我在哪儿",
    "我的坐标",
    "当前坐标",
    "经纬度",
)
PRICE_PREFERENCE_MARKERS = (
    "便宜",
    "低价",
    "价格低",
    "价低",
    "实惠",
    "省钱",
    "最低价",
    "最省",
)
TIME_PREFERENCE_TEXTS = (
    "早一点",
    "早些",
    "早点",
    "早一些",
    "早一点儿",
    "更早",
    "晚一点",
    "晚些",
    "晚点",
    "晚一些",
    "晚一点儿",
    "更晚",
)
GENERIC_BOOKING_TEXTS = {
    "book movie tickets",
    "buy movie tickets",
    "buy tickets",
    "book tickets",
    "movie tickets",
    "tickets",
}
IMPLICIT_SINGLE_TICKET_PATTERN = re.compile(
    r"(?:购买|预订|买|订|来)\s*张"
)


def _normalize_short_text(text: str) -> str:
    return re.sub(r"[\s，。,.!?！？、:：;；]+", "", text.strip()).casefold()


def is_greeting_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    greeting_values = {_normalize_short_text(value) for value in GREETING_TEXTS}
    if normalized in greeting_values:
        return True
    return normalized in {
        "你好呀",
        "您好呀",
        "你好啊",
        "您好啊",
        "嗨呀",
        "哈喽呀",
    }


def is_ack_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    ack_values = {_normalize_short_text(value) for value in ACK_TEXTS}
    return normalized in ack_values


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
            intent = (
                "book_ticket"
                if self._has_explicit_booking_cue(text)
                else "search_movies"
            )

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
                "get_current_location": "location_query",
                "select_seats": "confirm_order",
                "select_snacks": "select_snacks",
                "select_coupon": "select_coupon",
                "confirm_order": "confirm_order",
                "pay_order": "pay_order",
            }.get(event)
            if event_intent:
                return event_intent

        lowered = text.lower()
        if is_greeting_text(text):
            return "smalltalk"
        if is_ack_text(text):
            return "smalltalk"
        if self._is_cancel_text(text):
            return "cancel"
        if self._is_negative_snack_request(text):
            return "skip_snacks"
        if self._is_negative_coupon_request(text):
            return "skip_coupon"
        if self._is_order_query_text(text):
            return "order_query"
        if any(word in text for word in ["退票", "改签", "规则", "政策", "怎么处理", "FAQ"]):
            return "faq"
        if self._is_price_preference_text(text):
            return "select_or_modify"
        if any(marker in text for marker in PRICE_QUERY_MARKERS):
            return "price_query"
        if self._is_movie_keyword_query(text):
            return "search_movies"
        if self._is_movie_search_text(text):
            return "search_movies"
        if any(word in text for word in ["附近", "最近", "周边", "离我近", "高德", "地图"]):
            return "nearby_cinema"
        if any(word in lowered for word in ["nearby", "around me", "map", "amap"]):
            return "nearby_cinema"
        if self._is_location_query_text(text):
            return "location_query"
        if self._is_seat_only_request(text):
            return "seat_query"
        payload_slots = payload.get("slots", {})
        if self._is_plain_hall_type_request(text):
            return "select_or_modify"
        if any(
            key in payload or key in payload_slots
            for key in [
                "movieName",
                "movieId",
                "genre",
                "showtimeId",
                "cinemaId",
                "ticketCount",
            ]
        ):
            return "book_ticket"
        if any(word in text for word in ["零食", "爆米花", "饮料", "套餐", "小吃"]):
            return "snack"
        if any(word in text for word in ["优惠", "优惠券", "券", "折扣", "便宜"]):
            return "coupon" if "券" in text or "优惠" in text else "select_or_modify"
        if self._has_booking_slot_text(text):
            return "book_ticket"
        if any(
            word in text
            for word in [
                "座位",
                "选座",
                "靠中",
                "中间",
                "前排",
                "后排",
                "位置",
                "坐席",
            ]
        ):
            if any(word in text for word in ["买", "订", "购票", "影票", "电影票", "看电影"]):
                return "book_ticket"
            return "seat_query"
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
        seat_positions = self._extract_seat_positions(text)
        if seat_positions:
            slots["seatPositions"] = seat_positions

        hall_type = self._extract_hall_type(text)
        if hall_type:
            if self._is_negative_hall_type_request(text):
                self._add_clear_slots(slots, "hallType")
            else:
                slots["hallType"] = hall_type
        elif self._is_plain_hall_type_request(text):
            self._add_clear_slots(slots, "hallType")
        if self._is_price_preference_text(text):
            slots["pricePreference"] = "lower"
        if self._is_time_preference_text(text):
            slots["timePreference"] = "later" if "晚" in text else "earlier"

        movie_name = payload.get("movieName") or payload.get("movie_name")
        if movie_name:
            slots["movieName"] = movie_name
        elif intent == "search_movies":
            movie_keyword = self._extract_movie_search_keyword(text)
            if movie_keyword:
                slots["movieName"] = movie_keyword
        elif (
            not genre
            and not self._is_movie_search_text(text)
            and self._should_extract_movie_name(event, intent, text)
            and not self._is_seat_only_request(text)
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
            "seatPositions",
            "ticketCount",
        ]:
            if key in payload:
                slots[key] = payload[key]

        clear_slots = self._clear_slots_for_event(event)
        if clear_slots:
            self._add_clear_slots(slots, *clear_slots)

        if event == "select_snacks" and slots.get("snackId"):
            slots["snackIds"] = [slots["snackId"]]
        if event == "select_coupon" and slots.get("couponId"):
            slots["couponId"] = slots["couponId"]

        return slots

    def _should_extract_movie_name(self, event: str | None, intent: str, text: str) -> bool:
        if text.strip().casefold() in GENERIC_BOOKING_TEXTS:
            return False
        if self._is_ticket_count_only_request(text):
            return False
        if self._is_negative_hall_type_request(text) or self._is_plain_hall_type_request(text):
            return False
        if self._is_seat_only_request(text):
            return False
        return (
            event in {None, "", "text"}
            and intent in {"smalltalk", "book_ticket", "select_or_modify"}
            and not self._is_modification(text)
        )

    def _looks_like_movie_title(self, text: str) -> bool:
        normalized = self._normalize_movie_title(text)
        if self._is_non_movie_text(normalized):
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
            re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9·\s_:-]{2,40}", normalized)
        )

    def _clear_slots_for_event(self, event: str | None) -> list[str]:
        if event == "select_cinema":
            return [
                "cinemaId",
                "cinemaName",
                "showtimeId",
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
            ]
        if event == "select_movie":
            return [
                "movieId",
                "movieName",
                "genre",
                "showtimeId",
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
                "changeShowtime",
            ]
        if event == "select_showtime":
            return [
                "showtimeId",
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
                "changeShowtime",
            ]
        return []

    def _extract_ticket_count(self, text: str) -> int | None:
        match = re.search(r"(\d+)\s*[张人位票]", text)
        if match:
            return int(match.group(1))
        chinese_match = re.search(
            r"([一二两三四五六七八九十俩]{1,2})\s*[张人位票]",
            text,
        )
        if chinese_match:
            return self._parse_chinese_number(chinese_match.group(1))
        if IMPLICIT_SINGLE_TICKET_PATTERN.search(text):
            return 1
        return None

    def _parse_chinese_number(self, value: str) -> int | None:
        if value in CHINESE_NUMBERS:
            return CHINESE_NUMBERS[value]
        if len(value) == 2 and value[0] == "十":
            unit = CHINESE_NUMBERS.get(value[1])
            return 10 + unit if unit is not None else None
        if len(value) == 2 and value[1] == "十":
            tens = CHINESE_NUMBERS.get(value[0])
            return tens * 10 if tens is not None else None
        return None

    def _extract_time(self, text: str) -> str | None:
        if (
            self._is_price_preference_text(text)
            or self._is_time_preference_text(text)
        ):
            return None
        match = re.search(r"(\d{1,2})(?:点|:)(\d{1,2})?", text)
        if match:
            hour = int(match.group(1))
            if ("下午" in text or "今晚" in text or "晚上" in text or "晚" in text) and hour < 12:
                hour += 12
            minute = int(match.group(2) or 0)
            if hour > 23 or minute > 59:
                return None
            return f"{hour:02d}:{minute:02d}"
        chinese_match = re.search(r"([一二两三四五六七八九十]{1,2})点(半)?", text)
        if chinese_match:
            hour = CHINESE_HOURS.get(chinese_match.group(1))
            if hour is not None:
                if (
                    "下午" in text
                    or "今晚" in text
                    or "晚上" in text
                    or "晚" in text
                ) and hour < 12:
                    hour += 12
                if hour > 23:
                    return None
                minute = "30" if chinese_match.group(2) else "00"
                return f"{hour:02d}:{minute}"
        if "今晚" in text or "明晚" in text or "晚上" in text:
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

        match = re.search(
            r"(?<!\d)(?:(\d{4})\s*[年./-]\s*)?"
            r"(\d{1,2})\s*(?:[./-]\s*(\d{1,2})|月\s*(\d{1,2})\s*[日号]?)"
            r"\s*(?:日|号)?",
            text,
        )
        if not match:
            return None

        year_text, month_text, numeric_day, chinese_day = match.groups()
        month = int(month_text)
        day = int(numeric_day or chinese_day)
        year = int(year_text) if year_text else datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).year
        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            return None
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
        if self._is_non_movie_text(text):
            return None
        if self._is_ticket_count_only_request(text):
            return None
        if self._is_ticket_count_change_request(text):
            return None
        if self._is_negative_hall_type_request(text) or self._is_plain_hall_type_request(text):
            return None
        if self._is_seat_only_request(text):
            return None
        if self._extract_ordinal_text(text) is not None:
            return None
        value = text.strip()
        value = re.sub(
            r"^(?:你好|您好|嗨|哈喽|hi|hello)[，,\s]+"
            r"(?=(?:给我|帮我|我想|想|我要|请|麻烦|买|订|预订|购买|"
            r"看看|查|选|找))",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"^(?:给我|帮我|我想|想|我要|请|麻烦你?)", "", value)
        # "买张/订张" is colloquial shorthand for buying one ticket. Remove
        # the classifier before extracting the movie title, otherwise it leaks
        # into queries such as "张功夫女足".
        value = re.sub(
            r"^(?:购买|预订|买|订|来)\s*张",
            "",
            value,
        )
        value = re.sub(
            r"^(?:选择|看看|看一下|查一下|查询|查看|查|买|订|预订|购买|来|看|选|找)+",
            "",
            value,
        )
        value = re.sub(r"\d+\s*[张人位票]", "", value)
        value = re.sub(r"[一二两三四五六七八九十]\s*[张人位票]", "", value)
        value = re.sub(
            r"(?:(?:\d{4})\s*[年./-]\s*)?"
            r"\d{1,2}\s*(?:[./-]\s*\d{1,2}|月\s*\d{1,2}\s*[日号]?)"
            r"\s*(?:日|号)?",
            "",
            value,
        )
        value = re.sub(r"(?:今天|今晚|明天|明晚|晚上|下午|上午)", "", value)
        value = re.sub(r"(?:\d{1,2}|[一二两三四五六七八九十]{1,2})(?:点|:)\d{0,2}(?:半)?", "", value)
        value = re.sub(r"(?:IMAX|imax|杜比|巨幕|激光|场次|场|影厅|厅)", "", value)
        value = re.sub(
            r"(?:第?\s*)?(?:\d+|[一二两三四五六七八九十]{1,3})\s*排"
            r"(?:第?\s*)?(?:\d+|[一二两三四五六七八九十]{1,3})\s*座?",
            "",
            value,
        )
        value = re.sub(r"^(?:的|要|一场|一部)+", "", value)
        value = re.split(r"(?:电影票|影票|电影|票)", value, maxsplit=1)[0]
        value = value.strip(" ，。,.!?！？的")
        value = value.strip("《》「」『』“”\"' ")
        value = value.strip(" ，。,.!?！？的")
        if (
            value in {"喜剧", "爱情", "动作", "科幻", "动画", "悬疑", "恐怖"}
            or self._is_non_movie_text(value)
        ):
            return None
        if 2 <= len(value) <= 20:
            return value

        match = re.search(r"(?P<name>[\u4e00-\u9fa5A-Za-z0-9·]{2,20})(?:电影票|影票|电影|票)", text)
        if match:
            candidate = match.group("name").strip("的《》「」『』“”\"' ")
            if self._is_ticket_count_text(candidate):
                return None
            return candidate
        return None

    def _normalize_movie_title(self, text: str) -> str:
        return text.strip().strip("《》「」『』“”\"' ").strip()

    def _is_non_movie_text(self, text: str) -> bool:
        normalized = re.sub(r"[\s，。,.!?！？、:：;；]+", "", text.strip()).casefold()
        if not normalized:
            return True

        known_phrases = {
            re.sub(r"[\s，。,.!?！？、:：;；]+", "", phrase).casefold()
            for phrase in NON_MOVIE_TEXTS
        }
        if normalized in known_phrases:
            return True

        remaining = normalized
        ordered_phrases = sorted(known_phrases, key=len, reverse=True)
        while remaining:
            phrase = next(
                (item for item in ordered_phrases if remaining.startswith(item)),
                None,
            )
            if phrase is None:
                return False
            remaining = remaining[len(phrase):]
        return not remaining

    def _extract_ordinal_text(self, text: str) -> int | None:
        if "下一个" in text:
            return None
        match = re.search(
            r"第?\s*(\d+|[一二两三四五六七八九十])\s*(?:个|家|场|项|部)",
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

    def _extract_seat_positions(self, text: str) -> list[dict[str, int]]:
        positions: list[dict[str, int]] = []
        pattern = re.compile(
            r"(?:第?\s*)?(?P<row>\d+|[一二两三四五六七八九十]{1,3})\s*排"
            r"(?:第?\s*)?(?P<seat>\d+|[一二两三四五六七八九十]{1,3})\s*座?"
        )
        for match in pattern.finditer(text):
            row_no = self._parse_number_text(match.group("row"))
            seat_no = self._parse_number_text(match.group("seat"))
            if row_no is None or seat_no is None:
                continue
            item = {"rowNo": row_no, "seatNo": seat_no}
            if item not in positions:
                positions.append(item)
        return positions

    def _parse_number_text(self, value: str) -> int | None:
        if value.isdigit():
            return int(value)
        if value in CHINESE_HOURS:
            return CHINESE_HOURS[value]
        return self._parse_chinese_number(value)

    def _is_seat_only_request(self, text: str) -> bool:
        if not self._extract_seat_positions(text):
            return False

        seat_pattern = re.compile(
            r"(?:第?\s*)?(?:\d+|[一二两三四五六七八九十]{1,3})\s*排"
            r"(?:第?\s*)?(?:\d+|[一二两三四五六七八九十]{1,3})\s*座?"
        )
        remainder = seat_pattern.sub("", text)
        remainder = re.sub(
            r"(?:分别是|分别为|分别要|分别选|还是选|我就要|我想选|我想要|我选|"
            r"还是|那就|那么|就|换成|换为|改成|改为|重新选|再选|选座|座位|位置|"
            r"给我|帮我|我想|我要|请|要|换|改|选)",
            "",
            remainder,
        )
        remainder = re.sub(r"[和及与跟、，,。\s]+", "", remainder)
        return not remainder or self._extract_movie_name(remainder) is None

    def _is_ticket_count_text(self, value: str) -> bool:
        normalized = re.sub(r"\s+", "", value)
        return bool(
            re.fullmatch(
                r"(?:\d+|[一二两三四五六七八九十俩]{1,3})(?:张|人|位|票)",
            normalized,
        )
        )

    def _is_ticket_count_only_request(self, text: str) -> bool:
        normalized = re.sub(r"[\s，。,.!?！？]+", "", text)
        normalized = re.sub(
            r"^(?:给我|帮我|我想|我要|请|麻烦你?)?"
            r"(?:买|订|预订|购买|来|再加|加|多加|增加|添|改成|改为|换成|换为)?",
            "",
            normalized,
        )
        return bool(
            re.fullmatch(
            r"(?:\d+|[一二两三四五六七八九十俩]{1,3})"
                r"(?:张|人|位|票)(?:电影票|影票|票)?",
                normalized,
            )
        )

    def _is_modification(self, text: str) -> bool:
        return (
            self._is_price_preference_text(text)
            or self._is_time_preference_text(text)
            or "不要" in text
            or self._is_negative_hall_type_request(text)
            or self._is_plain_hall_type_request(text)
            or any(
                word in text
                for word in [
                    "换",
                    "改",
                    "更",
                    "不要这个",
                    "不要这场",
                    "不想要这场",
                    "便宜点",
                    "换便宜点",
                    "晚一点",
                    "早一点",
                ]
            )
        )

    def _is_cancel_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        cancel_values = {_normalize_short_text(value) for value in CANCEL_TEXTS}
        if normalized in cancel_values:
            return True
        return any(
            phrase in normalized
            for phrase in [
                "先不支付",
                "暂时不支付",
                "不想支付",
                "不要支付",
                "取消支付",
                "取消付款",
                "不支付",
                "先不付",
                "暂时不付",
                "不想付",
                "不想付款",
                "不付款了",
                "不付钱了",
                "先不付款",
                "暂时不付款",
                "不付了",
                "暂时不要了",
                "先不要了",
                "不想要了",
                "不想买了",
                "不用买了",
                "不想看了",
                "不看了",
                "我不要了",
                "我不想要了",
                "算了吧",
            ]
        )

    def _is_order_query_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if any(phrase in normalized for phrase in ORDER_QUERY_PHRASES):
            return True
        return bool(
            re.search(r"(?:查|查看|查询|看看).{0,3}订单", normalized)
        )

    def _is_movie_search_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        movie_search_values = {
            _normalize_short_text(value) for value in MOVIE_SEARCH_TEXTS
        }
        return any(value in normalized for value in movie_search_values)

    def _is_movie_keyword_query(self, text: str) -> bool:
        return self._extract_movie_search_keyword(text) is not None

    def _has_explicit_booking_cue(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if any(
            marker in normalized
            for marker in [
                "买",
                "订",
                "购票",
                "购买",
                "预订",
                "影票",
                "电影票",
                "张",
                "票",
                "座",
                "今晚",
                "明晚",
                "今天",
                "明天",
                "上午",
                "下午",
                "晚上",
                "几点",
            ]
        ):
            return True
        return bool(
            re.search(
                r"(?:\d+|[一二两三四五六七八九十俩]{1,3})(?:点|:|张|人|位|票)",
                normalized,
            )
        )

    def _extract_movie_search_keyword(self, text: str) -> str | None:
        normalized = _normalize_short_text(text)
        if not normalized:
            return None
        if any(
            marker in normalized
            for marker in [
                "买",
                "订",
                "购票",
                "购买",
                "预订",
                "影票",
                "电影票",
                "几张",
            ]
        ):
            return None
        match = re.fullmatch(
            r"(?:有没有|有无|查一下|查询|查看|看看|找一下|找)?"
            r"(?P<keyword>[\u4e00-\u9fffA-Za-z0-9·]{2,20})"
            r"(?:电影|影片)",
            normalized,
        )
        if not match:
            return None
        keyword = match.group("keyword").strip()
        if keyword in {"什么", "啥", "哪些", "一些", "些", "推荐", "热映"}:
            return None
        if keyword.startswith(("什么", "啥", "哪些", "有什么", "有啥", "有哪些", "有些")):
            return None
        return keyword

    def _is_location_query_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if any(
            phrase in normalized
            for phrase in ["换个位置", "换位置", "换座位", "选座", "座位"]
        ):
            return False
        if any(phrase in normalized for phrase in LOCATION_QUERY_PHRASES):
            return True
        return bool(
            re.search(
                r"(?:我|当前|现在|这里|这儿|此处).{0,4}"
                r"(?:位置|定位|坐标|经纬度|哪里|哪儿|在哪)",
                normalized,
            )
        )

    def _is_ticket_count_change_request(self, text: str) -> bool:
        normalized = re.sub(r"[\s，。,.!?！？]+", "", text)
        return bool(
            re.fullmatch(
                r"(?:再加|加|多加|增加|添|改成|改为|换成|换为)"
                r"(?:\d+|[一二两三四五六七八九十俩]{1,3})"
                r"(?:张|人|位|票)(?:电影票|影票|票)?",
                normalized,
            )
        )

    def _is_negative_snack_request(self, text: str) -> bool:
        return any(
            phrase in text
            for phrase in [
                "不要零食",
                "不需要零食",
                "不用零食",
                "不吃零食",
                "不加零食",
                "不买零食",
                "不买零食了",
                "零食不要",
                "零食不要了",
                "不要爆米花",
                "不需要爆米花",
                "不加爆米花",
                "不加爆米花了",
                "不买爆米花",
                "不要饮料",
                "不买饮料",
                "不要套餐",
                "不加套餐",
                "不需要小吃",
            ]
        )

    def _is_negative_coupon_request(self, text: str) -> bool:
        return any(
            phrase in text
            for phrase in [
                "不用券",
                "不用优惠券",
                "不使用优惠券",
                "不要优惠券",
                "优惠券不要",
                "优惠券不要了",
                "不需要优惠券",
                "不想用券",
                "不使用券",
                "不用优惠",
                "不使用优惠",
                "不要优惠",
            ]
        )

    def _is_price_preference_text(self, text: str) -> bool:
        return any(marker in text for marker in PRICE_PREFERENCE_MARKERS)

    def _is_time_preference_text(self, text: str) -> bool:
        return any(marker in text for marker in TIME_PREFERENCE_TEXTS)

    def _is_negative_hall_type_request(self, text: str) -> bool:
        if not self._extract_hall_type(text):
            return False
        normalized = re.sub(r"\s+", "", text)
        return any(
            marker in normalized
            for marker in [
                "不要",
                "不想要",
                "不需要",
                "不用",
                "别要",
                "不要看",
                "不看",
            ]
        )

    def _is_plain_hall_type_request(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        return any(
            phrase in normalized
            for phrase in [
                "普通厅",
                "普通场",
                "普通版",
                "普通2D",
                "2D就行",
                "换普通",
                "改普通",
            ]
        )

    def _add_clear_slots(self, slots: dict[str, Any], *keys: str) -> None:
        current = slots.get("__clearSlots")
        clear_slots = list(current) if isinstance(current, list) else []
        for key in keys:
            if key not in clear_slots:
                clear_slots.append(key)
        slots["__clearSlots"] = clear_slots

    def _has_booking_slot_text(self, text: str) -> bool:
        return any(
            value is not None
            for value in [
                self._extract_ticket_count(text),
                self._extract_time(text),
                self._extract_date(text),
                self._extract_genre(text),
                self._extract_hall_type(text),
            ]
        )

    def _extract_reference(self, text: str, payload: dict[str, Any]) -> str:
        return str(payload.get("reference") or text)


nlu_engine = RuleBasedNLU()
