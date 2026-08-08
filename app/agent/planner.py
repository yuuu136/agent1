"""Task Planner — thin routing layer.  Reads NLU + DST, decides next action."""

from typing import Any

from app.agent.tracker import tracker
from app.schemas.agent import AgentPlan, AgentState, NLUResult


class TaskPlanner:
    def plan(self, state: AgentState, nlu: NLUResult) -> AgentPlan:
        slots = {**state.slots, **nlu.slots}
        order_context = state.selected.get("order")
        if (
            isinstance(order_context, dict)
            and order_context.get("orderId") not in [None, ""]
            and slots.get("orderId") in [None, ""]
        ):
            slots["orderId"] = order_context.get("orderId")

        # ── Low confidence: let LLM handle freely ──
        if nlu.intent_source == "llm" and nlu.confidence < 0.55:
            return AgentPlan(
                action="general_answer",
                reason=f"LLM 意图置信度低({nlu.confidence:.0%})",
                params={"query": state.last_user_text},
                state="answering",
            )

        # ── Direct-route intents (no DST needed) ──
        if nlu.intent in ("faq", "smalltalk"):
            if nlu.intent == "smalltalk" and nlu.intent_source == "llm":
                return AgentPlan(
                    action="general_answer",
                    reason="用户在与智能体对话",
                    params={"query": state.last_user_text},
                    state="answering",
                )
            return AgentPlan(
                action="smalltalk",
                reason="问候或闲聊",
                params={"query": state.last_user_text},
                state="idle",
            )

        if nlu.intent == "cancel":
            return AgentPlan(action="cancel", reason="用户取消当前流程", state="idle")

        if nlu.intent in ("skip_snacks",):
            return AgentPlan(
                action="confirm_selection",
                reason="用户跳过零食",
                params={"skipSnacks": True},
                state=self._resume_after_optional_skip(slots, state.state),
            )

        if nlu.intent == "skip_coupon":
            return AgentPlan(
                action="confirm_selection",
                reason="用户跳过优惠券",
                params={"skipCoupon": True},
                state=self._resume_after_optional_skip(slots, state.state),
            )

        if nlu.intent == "price_query":
            return AgentPlan(
                action="answer_price",
                reason="用户询问票价",
                params=self._pick(slots, ["showtimeId", "price", "ticketCount",
                    "movieName", "cinemaName", "date", "time", "hallName", "startAt"]),
                state=state.state or "selecting_showtime",
            )

        if nlu.intent in ("search_movies", "search_showtimes"):
            return AgentPlan(
                action="search_movies",
                reason="用户请求查看影片",
                params=self._pick(slots, ["movieName", "genre", "date",
                    "cinemaId", "cinemaName", "hallType", "notHallType",
                    "maxPrice", "recommendationCriteria", "movieLimit"]),
                state="selecting_movie",
            )

        if nlu.intent == "nearby_cinema":
            return AgentPlan(
                action="search_nearby_cinemas",
                reason="用户询问附近影院",
                params=self._pick(slots, ["location", "cinemaLimit"]),
                state="selecting_cinema",
            )

        if nlu.intent == "snack":
            return AgentPlan(
                action="recommend_snacks",
                reason="用户有零食需求",
                params=self._pick(slots, ["orderId", "ticketCount", "cinemaId", "cinemaName", "snackIds"]),
                state="selecting_snacks",
            )

        if nlu.intent == "select_snacks":
            snack_params = self._pick(
                slots,
                [
                    "orderId",
                    "snackIds",
                    "snackId",
                    "snackItems",
                    "snackRequests",
                    "quantity",
                    "ticketCount",
                    "cinemaId",
                    "cinemaName",
                ],
            )
            snack_id = snack_params.pop("snackId", None)
            quantity = snack_params.pop("quantity", None)
            if not snack_params.get("snackIds") and snack_id not in [None, ""]:
                snack_params["snackIds"] = [snack_id]
            if (
                snack_params.get("snackIds")
                and not snack_params.get("snackItems")
                and quantity not in [None, ""]
            ):
                snack_params["snackItems"] = [
                    {
                        "snackId": snack_params["snackIds"][0],
                        "quantity": quantity,
                    }
                ]
            if snack_params.get("snackIds") or snack_params.get("snackItems"):
                return AgentPlan(
                    action="confirm_selection",
                    reason="用户选择零食",
                    params=snack_params,
                    state="selecting_snacks",
                )
            return AgentPlan(
                action="recommend_snacks",
                reason="继续选择零食",
                params=snack_params,
                state="selecting_snacks",
            )

        if nlu.intent == "order_query":
            if slots.get("orderId") and "我的订单" not in nlu.reference_text:
                return AgentPlan(action="get_order", params=self._pick(slots, ["orderId"]), state="answering")
            return AgentPlan(action="list_orders", params={}, state="answering")

        if nlu.intent == "refund_status_query":
            if slots.get("orderId"):
                return AgentPlan(action="get_refund_status", params=self._pick(slots, ["orderId"]), state="answering")
            return AgentPlan(action="list_orders", params={}, state="answering")

        if nlu.intent == "refund_order":
            if slots.get("orderId"):
                return AgentPlan(action="refund_order", params=self._pick(slots, ["orderId"]), state="refunding")
            return AgentPlan(action="list_orders", params={}, state="answering")

        if nlu.intent == "location_query":
            return AgentPlan(
                action="get_current_location",
                params=self._pick(slots, ["location"]),
                state="answering",
            )

        if nlu.intent == "pay_order":
            if not slots.get("orderId"):
                if not slots.get("lockId"):
                    return AgentPlan(action="confirm_selection", reason="需要先选场次座位", params=slots, state="confirming")
                return AgentPlan(action="create_order", reason="先创建订单",
                    params=self._pick(slots, ["showtimeId", "seatIds", "snackIds"]), state="creating_order")
            return AgentPlan(action="pay_order", params=self._pick(slots, ["orderId"]), state="paying")

        if nlu.intent == "confirm_order":
            if slots.get("orderId"):
                return AgentPlan(action="pay_order", params=self._pick(slots, ["orderId"]), state="paying")
            if slots.get("showtimeId") and slots.get("seatIds"):
                return AgentPlan(action="lock_seats", params=self._pick(slots,
                    ["showtimeId", "seatIds", "ticketCount", "movieId", "movieName",
                     "cinemaId", "cinemaName", "hallName", "hallType", "language",
                     "date", "time", "startAt", "endAt"]), state="locking_seats")
            if slots.get("showtimeId"):
                return AgentPlan(action="get_seats",
                    params=self._pick(slots, ["showtimeId", "ticketCount", "seatPreference"]),
                    state="selecting_seats")
            return AgentPlan(action="confirm_selection", params=slots, state="confirming")

        if slots.get("changeShowtime"):
            showtime_params = self._pick(
                slots,
                [
                    "movieName",
                    "genre",
                    "date",
                    "timeRange",
                    "ticketCount",
                    "cinemaId",
                    "cinemaName",
                    "hallType",
                    "pricePreference",
                    "timePreference",
                    "seatPositions",
                    "excludeShowtimeId",
                ],
            )
            if slots.get("orderId") or state.selected.get("order"):
                return AgentPlan(
                    action="cancel_order",
                    reason="先释放旧的锁座订单再换场",
                    params={
                        "orderId": slots.get("orderId"),
                        "resumeAction": "search_showtimes",
                        "resumeParams": showtime_params,
                    },
                    state="selecting_showtime",
                )
            return AgentPlan(
                action="search_showtimes",
                params=showtime_params,
                state="selecting_showtime",
            )

        if nlu.intent in ("select_showtime", "seat_query") or (
            slots.get("showtimeId") and "showtimeId" in nlu.slots
        ):
            # User wants to (re)select seats. Cancel any pending order first.
            if (slots.get("orderId") or state.selected.get("order")):
                seat_params = self._pick(
                    slots,
                    [
                        "showtimeId",
                        "ticketCount",
                        "seatPreference",
                        "seatPositions",
                    ],
                )
                return AgentPlan(
                    action="cancel_order",
                    reason="先释放旧的锁座订单再重选",
                    params={
                        "orderId": slots.get("orderId"),
                        "resumeAction": (
                            "get_seats"
                            if seat_params.get("showtimeId")
                            else "search_showtimes"
                        ),
                        "resumeParams": (
                            seat_params
                            if seat_params.get("showtimeId")
                            else self._pick(
                                slots,
                                [
                                    "movieId",
                                    "movieName",
                                    "genre",
                                    "date",
                                    "timeRange",
                                    "ticketCount",
                                    "cinemaId",
                                    "cinemaName",
                                    "hallType",
                                    "pricePreference",
                                    "timePreference",
                                ],
                            )
                        ),
                    },
                    state="selecting_seats",
                )
            if not slots.get("showtimeId"):
                return AgentPlan(action="search_showtimes",
                    params=self._pick(slots, ["movieId", "movieName", "genre", "date",
                        "timeRange", "ticketCount", "cinemaId", "cinemaName",
                        "hallType", "pricePreference", "timePreference",
                        "seatPreference", "seatPositions"]),
                    state="selecting_showtime")
            return AgentPlan(action="get_seats",
                params=self._pick(slots, ["showtimeId", "ticketCount", "seatPreference", "seatPositions"]),
                state="selecting_seats")

        # ── Booking flow: use DST to decide ──
        if nlu.intent in ("book_ticket", "select_or_modify"):
            return self._plan_booking(state, nlu, slots)

        # ── Fallback: change-showtime, change-cinema, seat_query, etc ──
        if slots.get("changeCinema"):
            return AgentPlan(action="search_nearby_cinemas",
                params=self._pick(slots, ["location", "city", "movieName", "genre"]),
                state="selecting_cinema")

        if (
            slots.get("showtimeId")
            and state.pending_action == "get_seats"
            and state.state == "selecting_seats"
        ):
            return AgentPlan(action="get_seats",
                params=self._pick(slots, ["showtimeId", "ticketCount", "seatPreference", "seatPositions"]),
                state="selecting_seats")

        return AgentPlan(action="general_answer",
            reason="无法路由，交给大模型自由回答",
            params={"query": state.last_user_text},
            state="answering")

    # ── Booking flow with DST ──

    def _plan_booking(self, state: AgentState, nlu: NLUResult,
                      slots: dict[str, Any]) -> AgentPlan:
        tr = tracker.assess(state, nlu)

        # Movie validation gate: before asking for anything, verify the
        # movie actually has showtimes. LLM chat may have mentioned films
        # that aren't in our database.
        movie_name = slots.get("movieName")
        if tr.flow == "book_ticket" and movie_name and not self._movie_is_verified(
            movie_name,
            state,
        ):
            if slots.get("seatPositions"):
                return AgentPlan(
                    action="search_showtimes",
                    reason="用户已给出具体座位，直接查场次并尝试锁座",
                    params=self._pick(
                        slots,
                        [
                            "movieName",
                            "movieId",
                            "date",
                            "timeRange",
                            "ticketCount",
                            "cinemaId",
                            "cinemaName",
                            "hallType",
                            "notHallType",
                            "maxPrice",
                            "seatPositions",
                            "excludeShowtimeId",
                        ],
                    ),
                    state="selecting_showtime",
                )
            return AgentPlan(
                action="search_movies",
                reason="先验证电影是否有场次",
                params=self._pick(
                    slots,
                    [
                        "movieName",
                        "genre",
                        "date",
                        "cinemaId",
                        "cinemaName",
                        "hallType",
                        "notHallType",
                        "maxPrice",
                        "recommendationCriteria",
                    ],
                ),
                state="selecting_movie",
            )

        if tr.ready and tr.flow == "book_ticket":
            if movie_name or slots.get("movieId"):
                return AgentPlan(
                    action="search_showtimes",
                    reason="购票信息齐全，查场次",
                    params=self._pick(slots, ["movieName", "movieId", "date", "timeRange",
                        "ticketCount", "cinemaId", "cinemaName", "hallType",
                        "notHallType", "maxPrice"]),
                    state="selecting_showtime",
                )
            return AgentPlan(
                action="search_movies",
                reason="先搜电影让用户选",
                params=self._pick(slots, ["movieName", "genre", "date",
                    "cinemaId", "cinemaName", "hallType", "recommendationCriteria"]),
                state="selecting_movie",
            )

        if tr.next_ask:
            return AgentPlan(
                action="ask",
                reason=f"缺少 {tr.next_ask}",
                params={"missing": tr.missing, "filled": tr.filled,
                        "next_ask": tr.next_ask, "flow": tr.flow},
                state=tr.stage,
            )

        return AgentPlan(
            action="general_answer",
            reason="购票流程无可用动作",
            params={"query": state.last_user_text},
            state="answering",
        )

    @staticmethod
    def _movie_is_verified(movie_name: str, state: "AgentState") -> bool:
        """Check if this movie name was already returned by a backend search.
        Movie names from LLM chat (general_answer) are NOT verified."""
        candidates = state.selected.get("movie_candidates") or []
        if isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, dict) and (
                    item.get("movieName") == movie_name
                    or item.get("movieId") == movie_name
                ):
                    return True
        return False

    @staticmethod
    def _pick(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        return {key: source[key] for key in keys
                if key in source and source[key] not in (None, "")}

    @staticmethod
    def _resume_after_optional_skip(slots: dict[str, Any], current_state: str) -> str:
        if slots.get("orderId"):
            return "paying"
        if slots.get("showtimeId") and slots.get("seatIds"):
            return "confirming"
        if slots.get("showtimeId"):
            return "selecting_seats"
        return current_state or "idle"


task_planner = TaskPlanner()
