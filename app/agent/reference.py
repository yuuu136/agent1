import re
from typing import Any

from app.agent.intent_catalog import intent_catalog
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
ANY_TIME_TEXTS = frozenset(intent_catalog.terms("any_time"))
SHOWTIME_QUERY_TEXTS = frozenset(intent_catalog.terms("showtime_query"))


class ReferenceResolver:
    def resolve(self, state: AgentState, nlu: NLUResult) -> NLUResult:
        slots = dict(nlu.slots)
        text = nlu.reference_text or state.last_user_text

        if (
            nlu.intent == "search_movies"
            and not slots.get("movieName")
            and not slots.get("movieId")
            and (
                slots.get("recommendationCriteria")
                or slots.get("movieLimit") not in [None, ""]
            )
        ):
            self._clear_movie_browse_context(state)

        if self._is_explicit_new_movie_request(state, nlu, slots, text):
            self._clear_previous_movie_context(slots)

        # LLM 已经利用上下文解析出了完整 slot，不再用硬编码覆盖。
        # 但“换场”是强上下文操作，必须优先保留当前影片并排除原场次。
        if (
            nlu.intent_source == "llm"
            and not self._is_change_showtime(text)
            and self._extract_ordinal(text) is None
            and not any(word in text for word in ["这个", "这场", "这家"])
            and (
            slots.get("cinemaName")
            or slots.get("movieName")
            or slots.get("movieId")
            or slots.get("showtimeId")
            )
        ):
            return nlu.model_copy(update={"slots": slots})

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
                self._add_clear_slots(
                    slots,
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
                )
            if "还是老位置" in text:
                slots["seatPreference"] = state.slots.get("seatPreference", "middle")

        if self._is_negative_hall_type_request(text):
            slots.pop("hallType", None)
            self._add_clear_slots(slots, "hallType")
            if self._has_downstream_selection(state):
                self._clear_downstream_selection(slots)

        if self._is_plain_hall_type_request(text):
            slots.pop("hallType", None)
            self._add_clear_slots(slots, "hallType")
            if self._has_downstream_selection(state):
                self._clear_downstream_selection(slots)

        if "ticketCount" in slots:
            self._normalize_ticket_count_change(state, slots, text)

        if self._is_change_showtime(text):
            slots["changeShowtime"] = True
            if state.slots.get("showtimeId") not in [None, ""]:
                slots["excludeShowtimeId"] = state.slots.get("showtimeId")
            self._clear_downstream_selection(slots)

        if nlu.intent == "seat_query" and (
            slots.get("seatPositions") or self._is_change_seat_request(text)
        ):
            # Seat replacement keeps the selected showtime, but invalidates
            # any previous seat lock or order draft.
            self._add_clear_slots(
                slots,
                "seatIds",
                "seatPositions",
                "orderId",
                "lockId",
                "couponId",
                "snackIds",
                "snackItems",
                "price",
            )

        ordinal = self._extract_ordinal(text)
        selected_slot: str | None = None
        if ordinal is not None:
            selected_slot = self._apply_candidate_ordinal(
                state,
                slots,
                ordinal,
                text,
            )

        if any(word in text for word in ["这个", "这场", "这家"]) and state.selected:
            self._apply_current_selection(state, slots, text)

        snack_request_selected = self._apply_snack_request_selection(
            state,
            slots,
        )

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
        if selected_slot == "showtimeId":
            intent = "select_showtime"
        elif selected_slot in {"movieId", "cinemaId"}:
            intent = "select_or_modify"
        elif snack_request_selected:
            intent = "select_snacks"
        elif state.pending_action == "ask_time" and self._is_any_time(text):
            slots["timePreference"] = "any"
            self._add_clear_slots(slots, "timeRange")
            slots.pop("timeRange", None)
            intent = "book_ticket"
        elif slots.get("changeShowtime"):
            intent = "book_ticket"
        elif self._should_continue_booking(state, nlu, slots):
            intent = "book_ticket"

        return nlu.model_copy(update={"intent": intent, "slots": slots})

    def _apply_snack_request_selection(
        self,
        state: AgentState,
        slots: dict[str, Any],
    ) -> bool:
        requests = slots.get("snackRequests")
        candidates = state.selected.get("snack_candidates") or []
        if not isinstance(requests, list) or not isinstance(candidates, list):
            return False
        if not requests or not candidates:
            return False

        snack_items: list[dict[str, int]] = []
        snack_ids: list[Any] = []
        for request in requests:
            if not isinstance(request, dict):
                continue
            requested_name = str(request.get("name") or "").strip()
            match = self._match_named_candidate(candidates, requested_name)
            if not match:
                continue
            snack_id = match.get("snackId")
            if snack_id in [None, ""]:
                continue
            quantity = self._positive_int(request.get("quantity")) or 1
            snack_ids.append(snack_id)
            snack_items.append({"snackId": snack_id, "quantity": quantity})

        if not snack_items:
            return False
        slots["snackIds"] = snack_ids
        slots["snackItems"] = snack_items
        return True

    def _match_named_candidate(
        self,
        candidates: list[Any],
        requested_name: str,
    ) -> dict[str, Any] | None:
        normalized_request = self._normalize_match_text(requested_name)
        if not normalized_request:
            return None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_name = self._normalize_match_text(
                candidate.get("name") or candidate.get("snackName")
            )
            if not candidate_name:
                continue
            if (
                normalized_request in candidate_name
                or candidate_name in normalized_request
            ):
                return candidate
        return None

    @staticmethod
    def _normalize_match_text(value: Any) -> str:
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).casefold()

    def _clear_downstream_selection(self, slots: dict[str, Any]) -> None:
        self._add_clear_slots(
            slots,
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

    @staticmethod
    def _clear_movie_browse_context(state: AgentState) -> None:
        for key in [
            "movie_candidates",
            "showtime_candidates",
            "cinema_candidates",
            "seat_map",
            "snack_candidates",
            "coupon_candidates",
            "order",
            "ticket",
            "calendar",
            "notification",
        ]:
            state.selected.pop(key, None)

    def _is_explicit_new_movie_request(
        self,
        state: AgentState,
        nlu: NLUResult,
        slots: dict[str, Any],
        text: str,
    ) -> bool:
        if nlu.intent not in {"book_ticket", "search_movies", "search_showtimes"}:
            return False
        if slots.get("movieId") not in [None, ""]:
            return False
        current_name = str(state.slots.get("movieName") or "").strip()
        requested_name = str(slots.get("movieName") or "").strip()
        if not current_name or not requested_name or current_name == requested_name:
            return False
        return requested_name.replace(" ", "") in text.replace(" ", "")

    def _clear_previous_movie_context(self, slots: dict[str, Any]) -> None:
        self._add_clear_slots(
            slots,
            "movieId",
            "genre",
            "date",
            "timeRange",
            "timePreference",
            "pricePreference",
            "ticketCount",
            "cinemaId",
            "cinemaName",
            "hallType",
            "showtimeId",
            "seatIds",
            "seatPositions",
        )

    def _normalize_ticket_count_change(
        self,
        state: AgentState,
        slots: dict[str, Any],
        text: str,
    ) -> None:
        old_count = self._positive_int(state.slots.get("ticketCount"))
        new_count = self._positive_int(slots.get("ticketCount"))
        if new_count is None:
            return
        if self._is_increment_ticket_count(text) and old_count is not None:
            new_count = old_count + new_count
            slots["ticketCount"] = new_count
        elif self._is_decrement_ticket_count(text) and old_count is not None:
            new_count = max(1, old_count - new_count)
            slots["ticketCount"] = new_count
        if (
            old_count is not None
            and new_count != old_count
            and self._has_downstream_selection(state)
        ):
            self._clear_downstream_selection(slots)

    def _is_increment_ticket_count(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        return bool(
            re.search(
                r"(?:再加|多加|增加|加)(?:一|两|俩|[1-9]\d*)张"
                r"(?:电影票|影票|票)?",
                normalized,
            )
        )

    def _is_decrement_ticket_count(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        return any(marker in normalized for marker in ["减", "少一", "少两", "少俩"])

    def _has_downstream_selection(self, state: AgentState) -> bool:
        downstream_slots = {
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
        }
        if any(key in state.slots for key in downstream_slots):
            return True
        return any(
            key in state.selected
            for key in [
                "showtime_candidates",
                "seat_map",
                "order",
                "ticket",
                "coupon_candidates",
                "snack_candidates",
            ]
        )

    def _is_negative_hall_type_request(self, text: str) -> bool:
        upper_text = text.upper()
        has_hall_type = any(hall in upper_text for hall in ["IMAX"]) or any(
            hall in text for hall in ["杜比", "巨幕", "激光"]
        )
        if not has_hall_type:
            return False
        normalized = re.sub(r"\s+", "", text)
        return any(
            marker in normalized
            for marker in ["不要", "不想要", "不需要", "不用", "别要", "不看"]
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

    def _is_change_seat_request(self, text: str) -> bool:
        return any(word in text for word in ["座位", "选座", "位置", "座"]) and any(
            word in text
            for word in ["换", "改", "重新", "重选", "不要这个", "不要当前", "不想要这个"]
        )

    def _positive_int(self, value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

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
        return False

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
    ) -> str | None:
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
            return slot_key
        return None

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
                "snackItems",
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
                "snackItems",
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
                "snackItems",
                "snackRequests",
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
