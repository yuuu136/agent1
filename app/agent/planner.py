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

        if nlu.intent == "faq":
            return AgentPlan(
                action="answer_with_rag",
                reason="用户询问票务知识",
                params={"query": state.last_user_text},
                state="answering",
            )

        # ── Low confidence: let LLM handle freely ──
        if (
            nlu.intent_source == "llm"
            and nlu.confidence < 0.55
            and not slots.get("changeShowtime")
            and not any(
                key in nlu.slots
                for key in ["movieId", "cinemaId", "showtimeId"]
            )
        ):
            return AgentPlan(
                action="general_answer",
                reason=f"LLM 意图置信度低({nlu.confidence:.0%})",
                params={"query": state.last_user_text},
                state="answering",
            )

        # ── Direct-route intents (no DST needed) ──
        if nlu.intent == "smalltalk":
            if nlu.intent_source == "llm":
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
            if slots.get("orderId") or state.selected.get("order"):
                return AgentPlan(
                    action="cancel_order",
                    reason="用户取消当前锁座订单",
                    params={
                        "orderId": slots.get("orderId"),
                        "cancelFlow": True,
                    },
                    state="idle",
                )
            return AgentPlan(action="cancel", reason="用户取消当前流程", state="idle")

        if nlu.intent == "multi_movie_booking" or len(
            nlu.slots.get("movieNames") or [],
        ) > 1:
            movie_names = [
                str(name)
                for name in nlu.slots.get("movieNames", [])
                if str(name).strip()
            ]
            movie_list = "、".join(f"《{name}》" for name in movie_names[:3])
            return AgentPlan(
                action="ask",
                reason="当前订单仅支持一部电影",
                params={
                    "message": (
                        f"这次同时识别到了{movie_list}。"
                        "一笔订单暂时只能购买一部电影，请告诉我先订哪一部。"
                    ),
                    "missing": ["movieName"],
                    "next_ask": "movieName",
                },
                state="collecting_movie",
            )

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

        # “换一场”必须在普通影片搜索前处理：保留当前影片和筛选条件，
        # 只排除正在看的原场次，避免退回电影卡列表。
        if slots.get("changeShowtime"):
            showtime_params = self._pick(
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
                    "maxPrice",
                    "pricePreference",
                    "timePreference",
                    "location",
                    "nearbyFirst",
                    "cinemaLimit",
                    "showtimeLimit",
                    "seatType",
                    "seatPositions",
                    "excludeShowtimeId",
                    "autoSelectShowtime",
                    "autoSelectSeats",
                    "skipSnacks",
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
            if not any(
                showtime_params.get(key) not in [None, ""]
                for key in [
                    "movieId",
                    "movieName",
                    "genre",
                    "cinemaId",
                    "cinemaName",
                    "date",
                    "timeRange",
                    "location",
                ]
            ):
                return AgentPlan(
                    action="ask",
                    reason="没有可用于换场的当前场次上下文",
                    params={
                        "message": "你还没有选定要更换的场次，请先告诉我想看哪部电影或选择一个场次。",
                        "missing": ["movieName"],
                        "next_ask": "movieName",
                    },
                    state="collecting_movieName",
                )
            return AgentPlan(
                action="search_showtimes",
                params=showtime_params,
                state="selecting_showtime",
            )

        if nlu.intent == "search_showtimes":
            if (
                slots.get("nearbyFirst")
                and slots.get("cinemaId") in [None, ""]
                and slots.get("location") in [None, ""]
            ):
                return AgentPlan(
                    action="ask",
                    reason="按最近影院查询需要当前位置",
                    params={
                        "message": "需要先获取你的经纬度位置，才能按最近影院帮你找场次。请允许定位后再试一次。",
                        "missing": ["location"],
                        "next_ask": "location",
                    },
                    state="collecting_location",
                )
            return AgentPlan(
                action="search_showtimes",
                reason="用户请求查看指定影片的场次",
                params=self._pick(
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
                        "notHallType",
                        "maxPrice",
                        "pricePreference",
                        "timePreference",
                        "location",
                        "nearbyFirst",
                        "cinemaLimit",
                        "showtimeLimit",
                        "seatType",
                        "autoSelectShowtime",
                        "autoSelectSeats",
                        "skipSnacks",
                    ],
                ),
                state="selecting_showtime",
            )

        if nlu.intent == "search_movies":
            return AgentPlan(
                action="search_movies",
                reason="用户请求查看影片",
                params=self._pick(slots, ["movieName", "genre", "date", "timeRange",
                    "ticketCount",
                    "cinemaId", "cinemaName", "hallType", "notHallType",
                    "maxPrice", "recommendationCriteria", "movieLimit", "actor"]),
                state="selecting_movie",
            )

        if nlu.intent == "nearby_cinema":
            return AgentPlan(
                action="search_nearby_cinemas",
                reason="用户询问附近影院",
                params=self._pick(
                    slots,
                    ["location", "cinemaLimit", "hallType"],
                ),
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
            quantity_value = self._positive_int(quantity)
            if quantity not in [None, ""] and quantity_value is None:
                return AgentPlan(
                    action="ask",
                    reason="零食数量无效",
                    params={
                        "message": "请先选择至少 1 份零食，再加入订单。",
                        "missing": [],
                    },
                    state="selecting_snacks",
                )
            if (
                snack_params.get("snackIds")
                and not snack_params.get("snackItems")
                and quantity_value is not None
            ):
                snack_params["snackItems"] = [
                    {
                        "snackId": snack_params["snackIds"][0],
                        "quantity": quantity_value,
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
                seat_ids = slots.get("seatIds")
                expected_count = self._positive_int(slots.get("ticketCount"))
                if expected_count is None and isinstance(seat_ids, list):
                    expected_count = len(seat_ids)
                    slots["ticketCount"] = expected_count
                if (
                    expected_count is not None
                    and isinstance(seat_ids, list)
                    and len(seat_ids) != expected_count
                ):
                    return AgentPlan(
                        action="ask",
                        reason="已选座位数量与购票数量不一致",
                        params={
                            "message": (
                                f"你要买 {expected_count} 张票，"
                                f"目前选了 {len(seat_ids)} 个座位，请补齐后再确认。"
                            ),
                            "missing": [],
                        },
                        state="selecting_seats",
                    )
                return AgentPlan(action="lock_seats", params=self._pick(slots,
                    ["showtimeId", "seatIds", "ticketCount", "movieId", "movieName",
                     "cinemaId", "cinemaName", "hallName", "hallType", "language",
                     "date", "time", "startAt", "endAt"]), state="locking_seats")
            if slots.get("showtimeId"):
                return AgentPlan(action="get_seats",
                    params=self._pick(slots, ["showtimeId", "ticketCount", "seatPreference", "seatType"]),
                    state="selecting_seats")
            return AgentPlan(action="confirm_selection", params=slots, state="confirming")

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
                        "seatType",
                        "seatPositions",
                        "autoSelectSeats",
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
                if (
                    nlu.intent == "seat_query"
                    and not any(
                        slots.get(key) not in [None, ""]
                        for key in [
                            "movieId",
                            "movieName",
                            "genre",
                            "cinemaId",
                            "cinemaName",
                            "date",
                            "timeRange",
                        ]
                    )
                ):
                    return AgentPlan(
                        action="ask",
                        reason="换座位需要先确定场次",
                        params={
                            "message": "需要先选择具体场次后才能换座位。你可以先告诉我想看哪部电影，或从订单里选择要换座的订单。",
                            "missing": ["showtimeId"],
                            "next_ask": "showtimeId",
                        },
                        state="selecting_showtime",
                    )
                if nlu.intent == "seat_query":
                    candidates = state.selected.get("showtime_candidates")
                    if isinstance(candidates, list) and len(candidates) > 1:
                        return AgentPlan(
                            action="ask",
                            reason="选座前需要先确认具体场次",
                            params={
                                "message": "想选哪一场呢？请先从上面的场次中选择一场，再帮你选座。",
                                "missing": ["showtimeId"],
                                "next_ask": "showtimeId",
                            },
                            state="selecting_showtime",
                        )
                return AgentPlan(action="search_showtimes",
                    params=self._pick(slots, ["movieId", "movieName", "genre", "date",
                        "timeRange", "ticketCount", "cinemaId", "cinemaName",
                        "hallType", "pricePreference", "timePreference",
                        "seatPreference", "seatType", "seatPositions", "autoSelectShowtime",
                        "autoSelectSeats", "skipSnacks"]),
                    state="selecting_showtime")
            return AgentPlan(action="get_seats",
                params=self._pick(
                    slots,
                    [
                        "showtimeId",
                        "ticketCount",
                        "seatPreference",
                        "seatType",
                        "seatPositions",
                        "autoSelectSeats",
                        "skipSnacks",
                    ],
                ),
                state="selecting_seats")

        if (
            nlu.intent == "select_or_modify"
            and slots.get("movieId")
            and (
                "movieId" in nlu.slots
                or state.state == "selecting_movie"
                or state.pending_action == "search_movies"
            )
        ):
            return AgentPlan(
                action="search_showtimes",
                reason="用户已选择影片，直接查询该影片场次",
                params=self._pick(
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
                        "notHallType",
                        "maxPrice",
                        "pricePreference",
                        "timePreference",
                        "seatPreference",
                        "seatType",
                        "seatPositions",
                        "autoSelectShowtime",
                        "autoSelectSeats",
                        "skipSnacks",
                    ],
                ),
                state="selecting_showtime",
            )

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
                params=self._pick(slots, ["showtimeId", "ticketCount", "seatPreference", "seatType", "seatPositions"]),
                state="selecting_seats")

        return AgentPlan(action="general_answer",
            reason="无法路由，交给大模型自由回答",
            params={"query": state.last_user_text},
            state="answering")

    # ── Booking flow with DST ──

    def _plan_booking(self, state: AgentState, nlu: NLUResult,
                      slots: dict[str, Any]) -> AgentPlan:
        tr = tracker.assess(state, nlu)
        if (
            tr.flow == "book_ticket"
            and slots.get("genre")
            and slots.get("movieName") in [None, ""]
            and slots.get("movieId") in [None, ""]
            and not slots.get("autoSelectShowtime")
            and (
                not tr.next_ask
                or "cinemaName" in tr.missing
            )
        ):
            return AgentPlan(
                action="search_movies",
                reason="用户给出类型但未指定影片，先搜片让用户选择",
                params=self._pick(slots, ["genre", "date", "timeRange",
                    "ticketCount", "cinemaId", "cinemaName", "hallType",
                    "maxPrice", "recommendationCriteria", "seatPreference",
                    "seatType", "seatPositions", "autoSelectSeats",
                    "skipSnacks"]),
                state="selecting_movie",
            )
        # Movie validation gate: before asking for anything, verify the
        # movie actually has showtimes. LLM chat may have mentioned films
        # that aren't in our database.
        movie_name = slots.get("movieName")
        if tr.flow == "book_ticket" and slots.get("autoSelectShowtime"):
            showtime = self._auto_selectable_showtime_candidate(state)
            if showtime:
                params = self._pick(
                    slots,
                    [
                        "ticketCount",
                        "seatPreference",
                        "seatType",
                        "autoSelectSeats",
                        "skipSnacks",
                    ],
                )
                params.update(
                    {
                        key: showtime[key]
                        for key in [
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
                        ]
                        if showtime.get(key) not in [None, ""]
                    }
                )
                return AgentPlan(
                    action="get_seats",
                    reason="用户授权从当前场次中自动选择最早可售场次",
                    params=params,
                    state="selecting_seats",
                )
        if (
            tr.flow == "book_ticket"
            and slots.get("autoSelectShowtime")
            and not movie_name
            and not slots.get("movieId")
        ):
            movie = self._auto_selectable_movie_candidate(state)
            if movie:
                params = self._pick(
                    slots,
                    [
                        "date",
                        "timeRange",
                        "ticketCount",
                        "cinemaId",
                        "cinemaName",
                        "hallType",
                        "notHallType",
                        "maxPrice",
                        "pricePreference",
                        "timePreference",
                        "location",
                        "nearbyFirst",
                        "seatType",
                        "autoSelectShowtime",
                        "autoSelectSeats",
                        "skipSnacks",
                    ],
                )
                params["movieId"] = movie.get("movieId")
                params["movieName"] = movie.get("movieName")
                return AgentPlan(
                    action="search_showtimes",
                    reason="用户授权从当前推荐影片中自动选片并选场",
                    params=params,
                    state="selecting_showtime",
                )

        if tr.next_ask:
            return AgentPlan(
                action="ask",
                reason=f"缺少 {tr.next_ask}",
                params={"missing": tr.missing, "filled": tr.filled,
                        "next_ask": tr.next_ask, "flow": tr.flow},
                state=tr.stage,
            )

        if (
            tr.flow == "book_ticket"
            and movie_name
            and not slots.get("movieId")
            and not self._movie_is_verified(movie_name, state)
        ):
            if slots.get("seatPositions") or self._has_complete_booking_slots(slots):
                params = self._pick(
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
                        "seatPreference",
                        "seatType",
                        "seatPositions",
                        "excludeShowtimeId",
                        "autoSelectShowtime",
                        "autoSelectSeats",
                        "skipSnacks",
                    ],
                )
                if self._has_complete_booking_slots(slots):
                    params.setdefault("autoSelectShowtime", True)
                    params.setdefault("autoSelectSeats", True)
                    params.setdefault("seatPreference", slots.get("seatPreference") or "middle")
                    params.setdefault("skipSnacks", True)
                return AgentPlan(
                    action="search_showtimes",
                    reason="购票槽位齐全，直接查场次并尝试出订单",
                    params=params,
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
                        "timeRange",
                        "ticketCount",
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
                    params=self._ready_booking_params(slots),
                    state="selecting_showtime",
                )
            return AgentPlan(
                action="search_movies",
                reason="先搜电影让用户选",
                params=self._pick(slots, ["movieName", "genre", "date", "timeRange",
                        "ticketCount", "cinemaId", "cinemaName", "hallType",
                        "maxPrice", "recommendationCriteria",
                        "skipSnacks"]),
                state="selecting_movie",
            )

        return AgentPlan(
            action="general_answer",
            reason="购票流程无可用动作",
            params={"query": state.last_user_text},
            state="answering",
        )

    @staticmethod
    def _auto_selectable_movie_candidate(
        state: AgentState,
    ) -> dict[str, Any] | None:
        candidates = state.selected.get("movie_candidates") or []
        if not isinstance(candidates, list):
            return None

        valid = [
            movie
            for movie in candidates
            if isinstance(movie, dict)
            and movie.get("movieId") not in [None, ""]
            and movie.get("movieName") not in [None, ""]
        ]
        if not valid:
            return None

        return next(
            (
                movie
                for movie in valid
                if isinstance(movie.get("upcomingShowtimes"), list)
                and movie["upcomingShowtimes"]
            ),
            valid[0],
        )

    @staticmethod
    def _auto_selectable_showtime_candidate(
        state: AgentState,
    ) -> dict[str, Any] | None:
        candidates = state.selected.get("showtime_candidates") or []
        if not isinstance(candidates, list):
            return None

        valid = [
            showtime
            for showtime in candidates
            if isinstance(showtime, dict)
            and showtime.get("showtimeId") not in [None, ""]
        ]
        if not valid:
            return None

        return min(
            valid,
            key=lambda item: str(
                item.get("startAt")
                or f"{item.get('date') or '9999-12-31'}T{item.get('time') or '23:59'}"
            ),
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

    def _ready_booking_params(self, slots: dict[str, Any]) -> dict[str, Any]:
        params = self._pick(slots, ["movieName", "movieId", "date", "timeRange",
            "ticketCount", "cinemaId", "cinemaName", "hallType",
            "notHallType", "maxPrice", "seatPreference", "seatType",
            "seatPositions", "snackRequests", "autoSelectShowtime",
            "autoSelectSeats", "skipSnacks"])
        if self._has_complete_booking_slots(slots):
            params.setdefault("autoSelectShowtime", True)
            params.setdefault("autoSelectSeats", True)
            params.setdefault("seatPreference", slots.get("seatPreference") or "middle")
            params.setdefault("skipSnacks", True)
        return params

    @staticmethod
    def _has_complete_booking_slots(slots: dict[str, Any]) -> bool:
        has_movie = slots.get("movieName") not in [None, ""] or slots.get("movieId") not in [None, ""]
        has_cinema = (
            slots.get("cinemaName") not in [None, ""]
            or slots.get("cinemaId") not in [None, ""]
            or (
                slots.get("nearbyFirst")
                and slots.get("location") not in [None, ""]
            )
        )
        has_seat = any(
            slots.get(key) not in [None, "", [], {}]
            for key in ["seatPreference", "seatType", "seatPositions", "seatIds", "autoSelectSeats"]
        )
        return all(
            [
                has_movie,
                slots.get("ticketCount") not in [None, ""],
                slots.get("date") not in [None, ""],
                slots.get("timeRange") not in [None, ""],
                has_cinema,
                has_seat,
            ]
        )

    @staticmethod
    def _pick(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        return {key: source[key] for key in keys
                if key in source and source[key] not in (None, "")}

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

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
