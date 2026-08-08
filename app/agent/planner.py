from typing import Any

from app.schemas.agent import AgentPlan, AgentState, NLUResult


class TaskPlanner:
    def plan(self, state: AgentState, nlu: NLUResult) -> AgentPlan:
        slots = {**state.slots, **nlu.slots}

        if nlu.intent == "faq":
            return AgentPlan(
                action="answer_with_rag",
                reason="规则、FAQ 或稳定知识问题走 RAG",
                params={"query": state.last_user_text},
                state="answering",
            )

        if nlu.intent == "cancel":
            return AgentPlan(
                action="cancel",
                reason="用户取消当前购票流程",
                state="idle",
            )

        if nlu.intent == "smalltalk":
            return AgentPlan(
                action="smalltalk",
                reason="问候或闲聊不应继承旧购票流程",
                params={"query": state.last_user_text},
                state="idle",
            )

        if nlu.intent == "skip_snacks":
            return AgentPlan(
                action="confirm_selection",
                reason="用户明确跳过零食",
                params={"skipSnacks": True},
                state=self._resume_after_optional_skip(slots, state.state),
            )

        if nlu.intent == "skip_coupon":
            return AgentPlan(
                action="confirm_selection",
                reason="用户明确跳过优惠券",
                params={"skipCoupon": True},
                state=self._resume_after_optional_skip(slots, state.state),
            )

        if nlu.intent == "price_query":
            return AgentPlan(
                action="answer_price",
                reason="用户询问当前场次的票价，只读已有选座上下文",
                params=self._pick(
                    slots,
                    [
                        "showtimeId",
                        "price",
                        "ticketCount",
                        "movieName",
                        "cinemaName",
                        "date",
                        "time",
                        "hallName",
                        "startAt",
                    ],
                ),
                state=state.state or "selecting_showtime",
            )

        if nlu.intent == "search_movies":
            return AgentPlan(
                action="search_movies",
                reason="用户请求查看影片列表",
                params=self._pick(
                    slots,
                    [
                        "movieName",
                        "genre",
                        "city",
                        "cinemaId",
                        "cinemaName",
                        "date",
                        "hallType",
                        "recommendationCriteria",
                        "movieLimit",
                    ],
                ),
                state="selecting_movie",
            )

        if nlu.intent == "select_snacks":
            return AgentPlan(
                action="confirm_selection",
                reason="用户已选择零食",
                params=self._pick(slots, ["snackIds", "snackItems"]),
                state="selecting_snacks",
            )

        if nlu.intent == "nearby_cinema":
            return AgentPlan(
                action="search_nearby_cinemas",
                reason="用户询问附近影院，根据当前位置查询票务数据库中的影院",
                params=self._pick(slots, ["location", "city", "movieName", "genre"]),
                state="selecting_cinema",
            )

        if slots.get("changeCinema"):
            return AgentPlan(
                action="search_nearby_cinemas",
                reason="用户要求更换影院，重新查询票务数据库中的附近影院",
                params=self._pick(slots, ["location", "city", "movieName", "genre"]),
                state="selecting_cinema",
            )

        if slots.get("changeShowtime") and not (
            slots.get("showtimeId") and "showtimeId" in nlu.slots
        ):
            missing_plan = self._plan_missing_required_slots(slots)
            if missing_plan:
                return missing_plan
            return AgentPlan(
                action="search_showtimes",
                reason="用户要求更换场次，保留电影和时间条件重新查询",
                params=self._pick(
                    slots,
                    [
                        "movieName",
                        "genre",
                        "date",
                        "timeRange",
                        "ticketCount",
                        "city",
                        "cinemaId",
                        "cinemaName",
                        "hallType",
                        "pricePreference",
                        "timePreference",
                        "seatPositions",
                    ],
                ),
                state="selecting_showtime",
            )

        if nlu.intent == "snack":
            return AgentPlan(
                action="recommend_snacks",
                reason="用户有零食需求，调用零食 MCP",
                params=self._pick(
                    slots,
                    ["orderId", "ticketCount", "cinemaId", "cinemaName", "snackIds"],
                ),
                state="selecting_snacks",
            )

        if nlu.intent == "order_query":
            list_requested = any(
                phrase in nlu.reference_text
                for phrase in ["我的订单", "订单记录", "历史订单"]
            )
            if slots.get("orderId") and not list_requested:
                return AgentPlan(
                    action="get_order",
                    reason="用户查看当前订单详情",
                    params=self._pick(slots, ["orderId"]),
                    state="answering",
                )
            return AgentPlan(
                action="list_orders",
                reason="用户查看订单记录",
                params={},
                state="answering",
            )

        if nlu.intent == "refund_status_query":
            if slots.get("orderId"):
                return AgentPlan(
                    action="get_refund_status",
                    reason="用户查询当前订单退票/退款状态",
                    params=self._pick(slots, ["orderId"]),
                    state="answering",
                )
            return AgentPlan(
                action="list_orders",
                reason="用户查询退票/退款状态但未指定订单，先展示订单列表",
                params={},
                state="answering",
            )

        if nlu.intent == "refund_order":
            if slots.get("orderId"):
                return AgentPlan(
                    action="refund_order",
                    reason="用户申请当前订单退票退款",
                    params=self._pick(slots, ["orderId"]),
                    state="refunding",
                )
            return AgentPlan(
                action="list_orders",
                reason="用户申请退票但未指定订单，先展示可操作订单",
                params={},
                state="answering",
            )

        if nlu.intent == "location_query":
            return AgentPlan(
                action="get_current_location",
                reason="用户询问当前地理位置，优先读取浏览器经纬度并反向解析地址",
                params=self._pick(slots, ["location"]),
                state="answering",
            )

        if nlu.intent == "pay_order":
            if not slots.get("orderId"):
                if not slots.get("lockId"):
                    return AgentPlan(
                        action="confirm_selection",
                        reason="没有可支付订单，需要先选择场次和座位",
                        params=slots,
                        state="confirming",
                    )
                return AgentPlan(
                    action="create_order",
                    reason="支付前需要先有订单",
                    params=self._pick(slots, ["showtimeId", "seatIds", "snackIds"]),
                    state="creating_order",
                )
            return AgentPlan(
                action="pay_order",
                reason="用户确认支付",
                params=self._pick(slots, ["orderId"]),
                state="paying",
            )

        if nlu.intent == "confirm_order":
            if slots.get("orderId"):
                return AgentPlan(
                    action="pay_order",
                    reason="订单已经创建，确认后进入支付",
                    params=self._pick(slots, ["orderId"]),
                    state="paying",
                )
            if slots.get("showtimeId") and slots.get("seatIds"):
                return AgentPlan(
                    action="lock_seats",
                    reason="用户确认场次和座位，先锁座",
                    params=self._pick(
                        slots,
                        [
                            "showtimeId",
                            "seatIds",
                            "ticketCount",
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
                        ],
                    ),
                    state="locking_seats",
                )
            if slots.get("showtimeId"):
                return AgentPlan(
                    action="get_seats",
                    reason="用户确认座位但还没有座位 ID，返回座位图继续选择",
                    params=self._pick(slots, ["showtimeId", "ticketCount", "seatPreference"]),
                    state="selecting_seats",
                )
            return AgentPlan(
                action="confirm_selection",
                reason="确认当前选择",
                params=slots,
                state="confirming",
            )

        if (
            nlu.intent in {"select_showtime", "seat_query"}
            or (
                slots.get("showtimeId")
                and "showtimeId" in nlu.slots
            )
        ):
            if not slots.get("showtimeId"):
                missing_plan = self._plan_missing_required_slots(slots)
                if missing_plan:
                    return missing_plan
                return AgentPlan(
                    action="search_showtimes",
                    reason="用户指定了座位偏好但尚未选择场次，先返回可选场次",
                    params=self._pick(
                        slots,
                        [
                            "movieId",
                            "movieName",
                            "genre",
                            "date",
                            "timeRange",
                            "ticketCount",
                            "city",
                            "cinemaId",
                            "cinemaName",
                            "hallType",
                            "pricePreference",
                            "timePreference",
                            "seatPreference",
                            "seatPositions",
                        ],
                    ),
                    state="selecting_showtime",
                )
            return AgentPlan(
                action="get_seats",
                reason="用户选择了场次，查询座位图",
                params=self._pick(
                    slots,
                    ["showtimeId", "ticketCount", "seatPreference", "seatPositions"],
                ),
                state="selecting_seats",
            )

        if nlu.intent in {"book_ticket", "select_or_modify"}:
            missing_plan = self._plan_missing_required_slots(slots)
            if missing_plan:
                return missing_plan

            if slots.get("movieId") or slots.get("movieName"):
                return AgentPlan(
                    action="search_showtimes",
                    reason="已有具体电影，查询场次",
                    params=self._pick(
                        slots,
                        [
                            "movieId",
                            "movieName",
                            "date",
                            "timeRange",
                            "ticketCount",
                            "city",
                            "cinemaId",
                            "cinemaName",
                            "hallType",
                            "pricePreference",
                            "timePreference",
                            "seatPositions",
                        ],
                    ),
                    state="selecting_showtime",
                )

            return AgentPlan(
                action="search_movies",
                reason="购票流程先查电影",
                params=self._pick(
                    slots,
                    [
                        "genre",
                        "city",
                        "cinemaId",
                        "cinemaName",
                        "date",
                        "hallType",
                        "movieLimit",
                    ],
                ),
                state="selecting_movie",
            )

        if nlu.intent == "seat_query" or slots.get("showtimeId"):
            return AgentPlan(
                action="get_seats",
                reason="已有场次或用户询问座位，查询座位图",
                params=self._pick(
                    slots,
                    ["showtimeId", "ticketCount", "seatPreference", "seatPositions"],
                ),
                state="selecting_seats",
            )

        return AgentPlan(
            action="smalltalk",
            reason="没有识别到明确任务",
            params={"query": state.last_user_text},
            state="idle",
        )

    def _plan_missing_required_slots(self, slots: dict[str, Any]) -> AgentPlan | None:
        if (
            not slots.get("movieId")
            and not slots.get("movieName")
            and not slots.get("genre")
        ):
            return AgentPlan(
                action="ask_movie_or_genre",
                reason="缺电影或类型",
                state="collecting_movie",
            )
        if (
            not slots.get("date")
            and not slots.get("timeRange")
            and slots.get("timePreference") != "any"
        ):
            return AgentPlan(action="ask_time", reason="缺观影时间", state="collecting_time")
        if not slots.get("ticketCount"):
            return AgentPlan(
                action="ask_ticket_count",
                reason="缺购票数量",
                state="collecting_ticket_count",
            )
        return None

    def _pick(self, source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        return {key: source[key] for key in keys if key in source and source[key] not in [None, ""]}

    def _resume_after_optional_skip(
        self,
        slots: dict[str, Any],
        current_state: str,
    ) -> str:
        if slots.get("orderId"):
            return "paying"
        if slots.get("showtimeId") and slots.get("seatIds"):
            return "confirming"
        if slots.get("showtimeId"):
            return "selecting_seats"
        return current_state or "idle"


task_planner = TaskPlanner()
