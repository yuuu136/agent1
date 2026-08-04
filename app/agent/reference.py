import re
from typing import Any

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
                    "orderId",
                    "lockId",
                    "couponId",
                    "snackIds",
                    "price",
                ]
            if "还是老位置" in text:
                slots["seatPreference"] = state.slots.get("seatPreference", "middle")

        ordinal = self._extract_ordinal(text)
        if ordinal is not None:
            self._apply_candidate_ordinal(state, slots, ordinal)

        if any(word in text for word in ["这个", "这场", "这家"]) and state.selected:
            self._apply_current_selection(state, slots)

        if state.pending_action == "ask_time":
            self._normalize_followup_time(text, slots)

        intent = nlu.intent
        if self._should_continue_booking(state, nlu, slots):
            intent = "book_ticket"

        return nlu.model_copy(update={"intent": intent, "slots": slots})

    def _clear_downstream_selection(self, slots: dict[str, Any]) -> None:
        slots["__clearSlots"] = [
            "showtimeId",
            "seatIds",
            "orderId",
            "lockId",
            "couponId",
            "snackIds",
            "price",
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
        match = re.search(
            r"第?\s*(\d+|[一二两三四五六七八九十])\s*(?:个|家|场|项)",
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
    ) -> None:
        index = ordinal - 1
        candidates = (
            ("showtime_candidates", "showtimeId"),
            ("cinema_candidates", "cinemaId"),
            ("snack_candidates", "snackIds"),
            ("coupon_candidates", "couponId"),
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

    def _apply_current_selection(self, state: AgentState, slots: dict[str, Any]) -> None:
        candidates = (
            ("showtime_candidates", "showtimeId"),
            ("cinema_candidates", "cinemaId"),
            ("snack_candidates", "snackIds"),
            ("coupon_candidates", "couponId"),
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


reference_resolver = ReferenceResolver()
