import re
from typing import Any

from app.agent.nlu import SHOWTIME_QUERY_TEXTS
from app.schemas.agent import AgentState, NLUResult


CHINESE_ORDINALS = {
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
ANY_TIME_TEXTS = {
    "都可以",
    "都行",
    "随便",
    "不限",
    "时间不限",
    "什么时候都可以",
    "哪个时间都可以",
    "无所谓",
}


class ReferenceResolver:
    def resolve(self, state: AgentState, nlu: NLUResult) -> NLUResult:
        slots = dict(nlu.slots)
        text = nlu.reference_text or state.last_user_text

        if nlu.is_modification:
            if "便宜" in text:
                slots["pricePreference"] = "lower"
                self._clear_downstream_selection(slots)
            if "晚" in text:
                slots["timePreference"] = "later"
                self._clear_downstream_selection(slots)
            if "早" in text:
                slots["timePreference"] = "earlier"
                self._clear_downstream_selection(slots)
            if self._is_change_cinema(text):
                slots["changeCinema"] = True
                slots["__clearSlots"] = [
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
            if "还是老位置" in text:
                slots["seatPreference"] = state.slots.get("seatPreference", "middle")

        if self._is_change_showtime(text):
            slots["changeShowtime"] = True
            self._clear_downstream_selection(slots)

        if nlu.intent == "seat_query" and slots.get("seatPositions"):
            # A seat replacement keeps the selected showtime, but invalidates
            # any previous seat lock or order draft.
            self._add_clear_slots(
                slots,
                "seatIds",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
            )

        ordinal = self._extract_ordinal(text)
        if ordinal is not None:
            self._apply_candidate_ordinal(state, slots, ordinal, text)

        if any(word in text for word in ["这个", "这场", "这家"]) and state.selected:
            self._apply_current_selection(state, slots, text)

        if state.pending_action == "ask_time":
            self._normalize_followup_time(text, slots)

        if slots.get("timeRange"):
            self._add_clear_slots(slots, "timePreference")
            if state.slots.get("showtimeId"):
                self._clear_downstream_selection(slots)
        if slots.get("date") and state.slots.get("showtimeId"):
            self._clear_downstream_selection(slots)
        if slots.get("timePreference") and state.slots.get("showtimeId"):
            self._clear_downstream_selection(slots)

        intent = nlu.intent
        if state.pending_action == "ask_time" and self._is_any_time(text):
            slots["timePreference"] = "any"
            self._add_clear_slots(slots, "timeRange")
            slots.pop("timeRange", None)
            intent = "book_ticket"
        elif slots.get("changeShowtime"):
            intent = "book_ticket"
        elif self._should_continue_booking(state, nlu, slots):
            intent = "book_ticket"

        return nlu.model_copy(update={"intent": intent, "slots": slots})

    def _clear_downstream_selection(self, slots: dict[str, Any]) -> None:
        slots["__clearSlots"] = [
            "showtimeId",
            "seatIds",
            "seatPositions",
            "orderId",
            "lockId",
                "couponId",
                "snackIds",
                "price",
                "amount",
                "status",
                "expiresAt",
            ]

    def _is_change_cinema(self, text: str) -> bool:
        return any(
            phrase in text
            for phrase in [
                "换影院",
                "换个影院",
                "换一家",
                "换一家影院",
                "换别的影院",
                "重新选影院",
                "重选影院",
                "换个电影院",
                "换一家电影院",
            ]
        )

    def _is_change_showtime(self, text: str) -> bool:
        if any(
            phrase in text
            for phrase in [
                "换一场",
                "换个场次",
                "换时间",
                "换个时间",
                "重新选场次",
                "重选场次",
                "不要这场",
                "不想要这场",
                "下一场",
                "再来一场",
            ]
        ):
            return True
        if (
            any(word in text for word in ["换", "改", "更换", "重新", "重选"])
            and any(hall in text.upper() for hall in ["IMAX", "杜比", "巨幕", "激光"])
        ):
            return True
        if (
            any(word in text for word in ["换", "改", "更换", "重新", "重选"])
            and "场" in text
        ):
            return True
        return bool(
            re.search(
                r"(?:选|挑|找)(?:择)?(?:个|一个|一场)?"
                r"[^，。！？,.!?]{0,12}场次",
                text,
            )
        )

    def _add_clear_slots(self, slots: dict[str, Any], *keys: str) -> None:
        current = slots.get("__clearSlots")
        clear_slots = list(current) if isinstance(current, list) else []
        for key in keys:
            if key not in clear_slots:
                clear_slots.append(key)
        slots["__clearSlots"] = clear_slots

    def _is_any_time(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text.strip(" ，。,.!?！？"))
        if normalized in ANY_TIME_TEXTS | SHOWTIME_QUERY_TEXTS:
            return True
        return bool(
            re.fullmatch(
                r"(?:今天|明天|周末)(?:就)?(?:行|可以|好|都行|都可以|随便|无所谓|不限)",
                normalized,
            )
        )

    def _should_continue_booking(
        self,
        state: AgentState,
        nlu: NLUResult,
        slots: dict[str, Any],
    ) -> bool:
        if nlu.intent != "smalltalk":
            return False
        if not slots:
            return False
        if state.pending_action in {
            "ask_movie_or_genre",
            "ask_time",
            "ask_ticket_count",
            "search_movies",
            "search_showtimes",
            "get_seats",
        }:
            return True
        return state.state in {
            "collecting_movie",
            "collecting_time",
            "collecting_ticket_count",
            "selecting_movie",
            "selecting_showtime",
            "selecting_seats",
        }

    def _normalize_followup_time(self, text: str, slots: dict[str, Any]) -> None:
        time_range = slots.get("timeRange")
        if not isinstance(time_range, str):
            return
        if any(marker in text for marker in ["上午", "早上", "凌晨", "下午", "晚上", "今晚", "晚"]):
            return
        match = re.match(r"^(\d{2}):(\d{2})$", time_range)
        if not match:
            return
        hour = int(match.group(1))
        if 1 <= hour <= 11:
            slots["timeRange"] = f"{hour + 12:02d}:{match.group(2)}"

    def _extract_ordinal(self, text: str) -> int | None:
        if "下一个" in text:
            return None
        match = re.search(
            r"第?\s*(\d+|[一二两三四五六七八九十])\s*(?:个|家|场|项|部)",
            text,
        )
        if not match:
            return None
        value = match.group(1)
        return int(value) if value.isdigit() else CHINESE_ORDINALS.get(value)

    def _apply_candidate_ordinal(
        self,
        state: AgentState,
        slots: dict[str, Any],
        ordinal: int,
        text: str,
    ) -> None:
        index = ordinal - 1
        candidates = (
            ("movie_candidates", "movieId"),
            ("showtime_candidates", "showtimeId"),
            ("cinema_candidates", "cinemaId"),
            ("snack_candidates", "snackIds"),
            ("coupon_candidates", "couponId"),
        )
        preferred_key = self._candidate_key_for_text(text) or self._candidate_key_for_state(state)
        if preferred_key:
            candidates = (
                (preferred_key, dict(candidates)[preferred_key]),
                *tuple(item for item in candidates if item[0] != preferred_key),
            )
        for selected_key, slot_key in candidates:
            items = state.selected.get(selected_key) or []
            if not isinstance(items, list) or not 0 <= index < len(items):
                continue
            item = items[index]
            if not isinstance(item, dict):
                continue
            if slot_key == "snackIds":
                slots[slot_key] = [item.get("snackId")]
            else:
                slots[slot_key] = item.get(slot_key)
            self._clear_for_candidate_selection(slots, slot_key)
            for key in (
                "movieName",
                "cinemaName",
                "date",
                "time",
                "price",
                "hallName",
                "startAt",
                "location",
                "address",
            ):
                if item.get(key) is not None:
                    slots[key] = item[key]
            return

    def _apply_current_selection(
        self,
        state: AgentState,
        slots: dict[str, Any],
        text: str,
    ) -> None:
        candidates = (
            ("movie_candidates", "movieId"),
            ("showtime_candidates", "showtimeId"),
            ("cinema_candidates", "cinemaId"),
            ("snack_candidates", "snackIds"),
            ("coupon_candidates", "couponId"),
        )
        preferred_key = self._candidate_key_for_text(text) or self._candidate_key_for_state(state)
        if preferred_key:
            candidates = (
                (preferred_key, dict(candidates)[preferred_key]),
                *tuple(item for item in candidates if item[0] != preferred_key),
            )
        for selected_key, slot_key in candidates:
            items = state.selected.get(selected_key) or []
            if not items:
                continue
            item = items[0] if isinstance(items, list) else items
            if not isinstance(item, dict):
                continue
            if slot_key == "snackIds":
                slots[slot_key] = [item.get("snackId")]
            else:
                slots[slot_key] = item.get(slot_key)
            self._clear_for_candidate_selection(slots, slot_key)
            for key in (
                "movieName",
                "cinemaName",
                "date",
                "time",
                "price",
                "hallName",
                "startAt",
            ):
                if item.get(key) is not None:
                    slots[key] = item[key]
            return

    def _clear_for_candidate_selection(
        self,
        slots: dict[str, Any],
        slot_key: str,
    ) -> None:
        if slot_key == "movieId":
            self._add_clear_slots(
                slots,
                "genre",
                "showtimeId",
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
            )
        elif slot_key == "cinemaId":
            self._add_clear_slots(
                slots,
                "showtimeId",
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
            )
        elif slot_key == "showtimeId":
            self._add_clear_slots(
                slots,
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "price",
            )

    def _candidate_key_for_text(self, text: str) -> str | None:
        if any(word in text for word in ["影院", "这家", "电影院"]):
            return "cinema_candidates"
        if any(word in text for word in ["场次", "这场", "场"]):
            return "showtime_candidates"
        if any(word in text for word in ["电影", "影片", "这部", "部"]):
            return "movie_candidates"
        if any(word in text for word in ["零食", "爆米花", "饮料", "套餐"]):
            return "snack_candidates"
        if any(word in text for word in ["优惠", "优惠券", "券"]):
            return "coupon_candidates"
        return None

    def _candidate_key_for_state(self, state: AgentState) -> str | None:
        if state.state == "selecting_movie" or state.pending_action == "search_movies":
            return "movie_candidates"
        if state.state == "selecting_cinema" or state.pending_action == "search_nearby_cinemas":
            return "cinema_candidates"
        if state.state == "selecting_showtime" or state.pending_action == "search_showtimes":
            return "showtime_candidates"
        if state.state == "selecting_snacks":
            return "snack_candidates"
        if state.state == "selecting_coupon":
            return "coupon_candidates"
        return None


reference_resolver = ReferenceResolver()
