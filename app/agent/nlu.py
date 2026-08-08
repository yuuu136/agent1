import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.agent.intent_catalog import intent_catalog
from app.agent.intent_rag import intent_rag_retriever
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

IMPLICIT_SINGLE_TICKET_PATTERN = re.compile(
    r"(?:购买|预订|买|订|来)\s*张"
)


def _normalize_short_text(text: str) -> str:
    return re.sub(r"[\s，。,.!?！？、:：;；]+", "", text.strip()).casefold()


def _contains_catalog_term(text: str, lexicon: str) -> bool:
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in intent_catalog.terms(lexicon))


def is_greeting_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        _normalize_short_text(value)
        for value in intent_catalog.terms("greeting")
    }


def is_ack_text(text: str) -> bool:
    normalized = _normalize_short_text(text)
    return normalized in {
        _normalize_short_text(value)
        for value in intent_catalog.terms("ack")
    }


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
                "get_order": "order_query",
                "refund_order": "refund_order",
                "get_refund_status": "refund_status_query",
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
        if self._is_refund_status_query_text(text):
            return "refund_status_query"
        if self._is_refund_request_text(text):
            return "refund_order"
        if _contains_catalog_term(text, "faq"):
            return "faq"
        if self._is_price_preference_text(text):
            return "select_or_modify"
        if self._is_time_preference_text(text):
            return "select_or_modify"
        if _contains_catalog_term(text, "price_query"):
            return "price_query"
        if self._is_movie_recommendation_text(text):
            return "search_movies"
        if self._is_movie_keyword_query(text):
            return "search_movies"
        if self._is_movie_search_text(text):
            return "search_movies"
        if self._is_showtime_query_text(text):
            return "book_ticket"
        if self._is_explicit_showtime_booking_text(text):
            return "book_ticket"
        if _contains_catalog_term(text, "nearby_cinema"):
            if "最近" in text and ("上映" in text or "新片" in text
                                  or "电影" in text or "影片" in text
                                  or self._extract_movie_name(text)
                                  or self._extract_genre(text)):
                pass  # temporal "最近", not spatial, keep going to movie checks
            else:
                return "nearby_cinema"
        if _contains_catalog_term(lowered, "nearby_cinema_english"):
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
        if _contains_catalog_term(text, "coupon"):
            return "coupon" if "券" in text or "优惠" in text else "select_or_modify"
        if _contains_catalog_term(text, "seat"):
            if _contains_catalog_term(text, "booking_with_seat"):
                return "book_ticket"
            return "seat_query"
        intent_match = (
            intent_rag_retriever.retrieve(text)
            if self._should_use_intent_rag(text)
            else None
        )
        if intent_match:
            return intent_match.intent
        if self._is_movie_browsing(text):
            return "search_movies"
        if self._has_booking_slot_text(text):
            return "book_ticket"
        if self._extract_snack_requests(text):
            return "snack"
        if _contains_catalog_term(text, "payment"):
            return "pay_order"
        if _contains_catalog_term(text, "confirm"):
            return "confirm_order"
        if _contains_catalog_term(text, "booking"):
            return "book_ticket"
        if _contains_catalog_term(lowered, "booking_english"):
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
        snack_requests = self._extract_snack_requests(text)
        if snack_requests and not self._is_negative_snack_request(text):
            slots["snackRequests"] = snack_requests

        recommendation_criteria = self._extract_movie_recommendation_criteria(text)
        if recommendation_criteria and not self._has_explicit_booking_cue(text):
            slots["recommendationCriteria"] = recommendation_criteria
            self._add_clear_slots(
                slots,
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
                "snackItems",
                "snackRequests",
                "price",
                "amount",
                "status",
                "expiresAt",
            )

        movie_name = payload.get("movieName") or payload.get("movie_name")
        if movie_name:
            slots["movieName"] = movie_name
        elif intent == "search_movies":
            if not recommendation_criteria and not genre:
                movie_keyword = self._extract_movie_search_keyword(text)
                if movie_keyword:
                    slots["movieName"] = movie_keyword
                elif not self._is_movie_search_text(text):
                    extracted_movie = self._extract_movie_name(text)
                    if extracted_movie:
                        slots["movieName"] = extracted_movie
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
            "snackItems",
            "snackRequests",
            "quantity",
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
            quantity = self._positive_int(slots.get("quantity")) or 1
            slots["snackItems"] = [
                {"snackId": slots["snackId"], "quantity": quantity}
            ]
        if event == "select_coupon" and slots.get("couponId"):
            slots["couponId"] = slots["couponId"]

        return slots

    def _should_extract_movie_name(self, event: str | None, intent: str, text: str) -> bool:
        if text.strip().casefold() in {
            value.casefold()
            for value in intent_catalog.terms("generic_booking")
        }:
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
            for marker in intent_catalog.terms("movie_title_exclusion")
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

    def _extract_snack_requests(self, text: str) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        normalized = text.casefold()
        for canonical, aliases in intent_catalog.mapping("snack_alias").items():
            matched_alias = next(
                (alias for alias in aliases if alias.casefold() in normalized),
                None,
            )
            if not matched_alias:
                continue
            quantity, unit = self._extract_quantity_near_text(text, matched_alias)
            item = {"name": canonical, "quantity": quantity or 1}
            if unit:
                item["unit"] = unit
            if item not in requests:
                requests.append(item)
        return requests

    def _extract_quantity_near_text(self, text: str, keyword: str) -> tuple[int | None, str | None]:
        number_pattern = r"(?P<count>\d+|[一二两三四五六七八九十俩]{1,3})"
        unit_pattern = r"(?P<unit>瓶|杯|份|桶|个|套|包|盒)?"
        escaped = re.escape(keyword)
        before = re.search(
            rf"{number_pattern}\s*{unit_pattern}\s*{escaped}",
            text,
            re.IGNORECASE,
        )
        if before:
            return self._parse_number_text(before.group("count")), before.group("unit")
        after = re.search(
            rf"{escaped}\s*{number_pattern}\s*{unit_pattern}",
            text,
            re.IGNORECASE,
        )
        if after:
            return self._parse_number_text(after.group("count")), after.group("unit")
        return None, None

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
        if _contains_catalog_term(text, "time_evening"):
            return "evening"
        if _contains_catalog_term(text, "time_afternoon"):
            return "afternoon"
        if _contains_catalog_term(text, "time_morning"):
            return "morning"
        return None

    def _extract_date(self, text: str) -> str | None:
        if _contains_catalog_term(text, "date_today"):
            return "today"
        if _contains_catalog_term(text, "date_tomorrow"):
            return "tomorrow"
        if _contains_catalog_term(text, "date_weekend"):
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
        for genre, aliases in intent_catalog.mapping("genre").items():
            if any(alias in text for alias in aliases):
                return genre
        return None

    def _extract_hall_type(self, text: str) -> str | None:
        upper_text = text.upper()
        for hall_type, aliases in intent_catalog.mapping("hall_type").items():
            if any(alias.upper() in upper_text for alias in aliases):
                return hall_type
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
        value = re.sub(r"(?:最近|今天|今晚|明天|明晚|晚上|下午|上午)", "", value)
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
        value = re.sub(r"^(?:这个|那个)\s*", "", value)
        value = re.sub(r"\s*(?:这个|那个)$", "", value)
        if any(keyword in value for keyword in ("什么", "啥", "哪些", "怎么", "怎样", "如何", "哪部", "哪个", "哪家", "哪场")):
            return None
        if (
            value in intent_catalog.mapping("genre")
            or value in intent_catalog.terms("movie_title_generic")
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
            for phrase in (
                *intent_catalog.terms("greeting"),
                *intent_catalog.terms("ack"),
                *intent_catalog.terms("cancel"),
                *intent_catalog.terms("showtime_query"),
                *intent_catalog.terms("non_movie"),
            )
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
        for preference, aliases in intent_catalog.mapping("seat_preference").items():
            if any(alias in text for alias in aliases):
                return preference
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
            or _contains_catalog_term(text, "modification")
            or self._is_negative_hall_type_request(text)
            or self._is_plain_hall_type_request(text)
        )

    def _is_cancel_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        cancel_values = {
            _normalize_short_text(value)
            for value in intent_catalog.terms("cancel")
        }
        if normalized in cancel_values:
            return True
        return _contains_catalog_term(normalized, "cancel_contains")

    def _is_order_query_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if "退票" in normalized or "退款" in normalized:
            return False
        if _contains_catalog_term(normalized, "order_query"):
            return True
        return bool(
            re.search(r"(?:查|查看|查询|看看).{0,3}订单", normalized)
        )

    def _is_refund_request_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if not _contains_catalog_term(normalized, "refund_request"):
            return False
        if _contains_catalog_term(normalized, "refund_faq"):
            return False
        return True

    def _is_refund_status_query_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        return _contains_catalog_term(normalized, "refund_status")

    def _positive_int(self, value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _is_movie_search_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        movie_search_values = {
            _normalize_short_text(value)
            for value in intent_catalog.terms("movie_search")
        }
        return any(value in normalized for value in movie_search_values)

    def _is_showtime_query_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        showtime_values = {
            _normalize_short_text(value)
            for value in intent_catalog.terms("showtime_query")
        }
        return any(value in normalized for value in showtime_values)

    def _is_explicit_showtime_booking_text(self, text: str) -> bool:
        if "场次" not in text or self._is_modification(text):
            return False
        return _contains_catalog_term(text, "explicit_showtime_booking")

    def _is_movie_keyword_query(self, text: str) -> bool:
        return self._extract_movie_search_keyword(text) is not None

    def _is_movie_browsing(self, text: str) -> bool:
        """用户想浏览电影但没有指定具体电影名时返回 True。"""
        if self._extract_movie_name(text):
            return False
        if self._extract_ticket_count(text) or self._extract_date(text) or self._extract_time(text):
            return False
        genre = self._extract_genre(text)
        if genre:
            return True
        if "电影" not in text and "片" not in text:
            return False
        if any(phrase in text for phrase in [
            "看电影", "看看电影", "看看有什么电影", "有什么电影可以看",
            "想看电影", "想看啥电影", "想看什么电影",
            "想看什么片", "有什么片", "最近有什么片",
        ]):
            return True
        return False

    def _is_movie_recommendation_text(self, text: str) -> bool:
        return bool(
            self._extract_movie_recommendation_criteria(text)
            and not self._has_explicit_booking_cue(text)
        )

    def _extract_movie_recommendation_criteria(self, text: str) -> str | None:
        normalized = _normalize_short_text(text)
        if not normalized:
            return None

        for criteria, markers in intent_catalog.mapping(
            "recommendation_criteria"
        ).items():
            if any(marker in normalized for marker in markers):
                return criteria

        if "推荐" in normalized and _contains_catalog_term(
            normalized,
            "recommendation_general",
        ):
            return "general"
        return None

    def _should_use_intent_rag(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if not normalized:
            return False
        if not _contains_catalog_term(normalized, "rag_markers"):
            return False
        if self._looks_like_movie_title(text) and not _contains_catalog_term(
            normalized,
            "rag_title_markers",
        ):
            return False
        return True

    def _has_explicit_booking_cue(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if _contains_catalog_term(normalized, "booking_cues"):
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
        if _contains_catalog_term(normalized, "movie_keyword_excluded"):
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
        generic_keywords = set(intent_catalog.terms("movie_keyword_generic"))
        if keyword in generic_keywords:
            return None
        if keyword in intent_catalog.mapping("genre"):
            return None
        if keyword.startswith(
            intent_catalog.terms("movie_keyword_prefix_excluded")
        ):
            return None
        if "有什么" in keyword or "有啥" in keyword or "什么" in keyword:
            return None
        return keyword

    def _is_location_query_text(self, text: str) -> bool:
        normalized = _normalize_short_text(text)
        if _contains_catalog_term(normalized, "location_seat_exclusion"):
            return False
        if _contains_catalog_term(normalized, "location_query"):
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
        return _contains_catalog_term(text, "snack_negative")

    def _is_negative_coupon_request(self, text: str) -> bool:
        return _contains_catalog_term(text, "coupon_negative")

    def _is_price_preference_text(self, text: str) -> bool:
        return _contains_catalog_term(text, "price_preference")

    def _is_time_preference_text(self, text: str) -> bool:
        return _contains_catalog_term(text, "time_preference")

    def _is_negative_hall_type_request(self, text: str) -> bool:
        if not self._extract_hall_type(text):
            return False
        normalized = re.sub(r"\s+", "", text)
        return _contains_catalog_term(normalized, "negative_hall")

    def _is_plain_hall_type_request(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        return _contains_catalog_term(normalized, "plain_hall")

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
