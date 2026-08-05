from typing import Any

from app.clients import mcp_dispatcher
from app.rag.service import rag_service
from app.schemas.agent import AgentPlan, AgentState, ToolResult


class AgentToolbox:
    def execute(self, plan: AgentPlan, state: AgentState) -> ToolResult:
        action = plan.action

        if action == "answer_with_rag":
            return self.answer_with_rag(plan.params.get("query") or state.last_user_text)
        if action == "answer_price":
            return self.answer_price(plan.params, state)
        if action == "search_nearby_cinemas":
            # Nearby lookup must still validate browser coordinates when unauthenticated.
            return mcp_dispatcher.call("spring_boot", "search_nearby_cinemas", plan.params)
        if action in {"search_movies", "search_showtimes", "get_seats"}:
            server = "spring_boot" if plan.params.get("jwt") else "movie_ticket"
            return mcp_dispatcher.call(server, action, plan.params)
        if action in {
            "lock_seats",
            "create_order",
            "pay_order",
            "issue_ticket",
            "get_order",
            "list_orders",
        }:
            # Authenticated browser sessions use the real Java transaction API.
            # Keep the unauthenticated local adapter only for demo/test sessions.
            server = "spring_boot" if plan.params.get("jwt") else "movie_ticket"
            return mcp_dispatcher.call(server, action, plan.params)
        if action == "recommend_snacks":
            return mcp_dispatcher.call("snack", action, plan.params)
        if action == "recommend_coupons":
            return mcp_dispatcher.call("coupon", action, plan.params)
        if action == "create_calendar_event":
            return mcp_dispatcher.call("calendar", "create_event", plan.params)
        if action == "send_ticket_message":
            return mcp_dispatcher.call("notification", "send_ticket_message", plan.params)
        if action == "send_sms":
            return mcp_dispatcher.call("notification", "send_sms", plan.params)

        return ToolResult(
            tool_name=action,
            data={"params": plan.params},
        )

    def answer_with_rag(self, query: str) -> ToolResult:
        try:
            answer = rag_service.answer(query)
            return ToolResult(
                tool_name="rag.answer",
                data=answer,
                message=answer.get("message", ""),
            )
        except Exception as exc:
            return ToolResult(
                tool_name="rag.answer",
                success=False,
                data={"error": str(exc)},
                message="知识库回答暂时不可用，请稍后重试或改为查询业务接口。",
            )

    def answer_price(
        self,
        arguments: dict[str, Any],
        state: AgentState,
    ) -> ToolResult:
        showtime = self._resolve_showtime(arguments, state)
        seat_map = state.selected.get("seat_map") or {}
        raw_seat_map = seat_map.get("raw") if isinstance(seat_map, dict) else {}
        seats = seat_map.get("seats") if isinstance(seat_map, dict) else []

        price = self._first_amount(
            arguments.get("price"),
            state.slots.get("price"),
            showtime.get("price"),
            seat_map.get("price") if isinstance(seat_map, dict) else None,
            seat_map.get("basePrice") if isinstance(seat_map, dict) else None,
            raw_seat_map.get("basePrice") if isinstance(raw_seat_map, dict) else None,
            (
                seats[0].get("price")
                if isinstance(seats, list)
                and seats
                and isinstance(seats[0], dict)
                else None
            ),
        )
        if price is None:
            return ToolResult(
                tool_name="movie_ticket.answer_price",
                success=False,
                data={
                    "showtimeId": arguments.get("showtimeId")
                    or state.slots.get("showtimeId")
                },
                message="当前场次的票价暂未从票务数据库返回，请先重新选择场次。",
            )

        ticket_count = self._positive_int(
            arguments.get("ticketCount") or state.slots.get("ticketCount"),
            default=1,
        )
        total_price = price * ticket_count
        movie_name = (
            arguments.get("movieName")
            or showtime.get("movieName")
            or state.slots.get("movieName")
            or "当前场次"
        )
        data = {
            "showtimeId": arguments.get("showtimeId")
            or state.slots.get("showtimeId")
            or showtime.get("showtimeId"),
            "movieName": movie_name,
            "cinemaName": arguments.get("cinemaName") or showtime.get("cinemaName"),
            "unitPrice": price,
            "ticketCount": ticket_count,
            "totalPrice": total_price,
        }
        return ToolResult(
            tool_name="movie_ticket.answer_price",
            data=data,
            message=(
                f"{movie_name}当前票价为{self._format_amount(price)}元/张，"
                f"{ticket_count}张合计{self._format_amount(total_price)}元。"
                "最终金额以确认订单页面为准。"
            ),
        )

    def _resolve_showtime(
        self,
        arguments: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        showtime_id = arguments.get("showtimeId") or state.slots.get("showtimeId")
        candidates = state.selected.get("showtime_candidates") or []
        if not isinstance(candidates, list):
            return {}
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if showtime_id and str(item.get("showtimeId")) == str(showtime_id):
                return item
        if len(candidates) == 1 and isinstance(candidates[0], dict):
            return candidates[0]
        return {}

    def _first_amount(self, *values: Any) -> float | None:
        for value in values:
            if value in [None, ""]:
                continue
            try:
                amount = float(value)
            except (TypeError, ValueError):
                continue
            if amount >= 0:
                return amount
        return None

    def _positive_int(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _format_amount(self, value: float) -> str:
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")


agent_toolbox = AgentToolbox()
