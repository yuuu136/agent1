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
        if action == "get_current_location":
            return self.get_current_location(plan.params)
        if action == "search_nearby_cinemas":
            # Nearby lookup must still validate browser coordinates when unauthenticated.
            return mcp_dispatcher.call("spring_boot", "search_nearby_cinemas", plan.params)
        if action in {"search_movies", "search_showtimes", "get_seats"}:
            if not plan.params.get("jwt"):
                return self.auth_required(action)
            return mcp_dispatcher.call("spring_boot", action, plan.params)
        if action in {
            "lock_seats",
            "create_order",
            "pay_order",
            "issue_ticket",
            "refund_order",
            "get_refund_status",
            "get_order",
            "list_orders",
        }:
            if not plan.params.get("jwt"):
                return self.auth_required(action)
            return mcp_dispatcher.call("spring_boot", action, plan.params)
        if action == "recommend_snacks":
            if not plan.params.get("jwt"):
                return self.auth_required(action)
            if not plan.params.get("orderId"):
                return ToolResult(
                    tool_name="spring_boot.recommend_snacks",
                    success=False,
                    data={"error": "ORDER_REQUIRED"},
                    message="请先选择座位并创建订单后再选择零食。",
                )
            return mcp_dispatcher.call("spring_boot", action, plan.params)
        if action == "replace_order_snacks":
            if not plan.params.get("jwt"):
                return self.auth_required(action)
            if not plan.params.get("orderId"):
                return ToolResult(
                    tool_name="spring_boot.replace_order_snacks",
                    success=False,
                    data={"error": "ORDER_REQUIRED"},
                    message="请先选择座位并创建订单后再加入零食。",
                )
            return mcp_dispatcher.call("spring_boot", action, plan.params)

        return ToolResult(
            tool_name=action,
            data={"params": plan.params},
        )

    def auth_required(self, action: str) -> ToolResult:
        return ToolResult(
            tool_name=f"spring_boot.{action}",
            success=False,
            data={"error": "AUTH_REQUIRED"},
            message="请先登录后再使用真实票务服务。",
        )

    def get_current_location(self, arguments: dict[str, Any]) -> ToolResult:
        normalized = self._normalize_location(arguments.get("location"))
        if not normalized:
            return ToolResult(
                tool_name="location.current",
                success=False,
                data={"error": "LOCATION_REQUIRED"},
                message="未获取到浏览器经纬度，请允许定位后再查询你的具体位置。",
            )

        longitude, latitude = self._split_location(normalized)
        data: dict[str, Any] = {
            "location": normalized,
            "longitude": longitude,
            "latitude": latitude,
            "source": "browser",
        }
        reverse_result = mcp_dispatcher.call(
            "amap",
            "regeocode",
            {"location": normalized, "extensions": "base"},
        )
        if reverse_result.success:
            regeocode = reverse_result.data.get("regeocode") or {}
            address = self._regeocode_address(regeocode)
            if address:
                data["address"] = address
            component = regeocode.get("addressComponent")
            if isinstance(component, dict):
                data["addressComponent"] = component
            if address:
                message = (
                    f"你当前的位置是：{address}。"
                    f"经度 {longitude:.6f}，纬度 {latitude:.6f}。"
                )
            else:
                message = (
                    f"已获取你的当前位置：经度 {longitude:.6f}，"
                    f"纬度 {latitude:.6f}。"
                )
        else:
            message = (
                f"已获取你的当前位置：经度 {longitude:.6f}，"
                f"纬度 {latitude:.6f}。地址解析暂不可用。"
            )

        return ToolResult(
            tool_name="location.current",
            data=data,
            message=message,
        )

    def _normalize_location(self, value: Any) -> str | None:
        if isinstance(value, dict):
            longitude = value.get("longitude") or value.get("lng")
            latitude = value.get("latitude") or value.get("lat")
            if longitude in [None, ""] or latitude in [None, ""]:
                return None
            value = f"{longitude},{latitude}"
        if value in [None, ""]:
            return None
        text = str(value).strip()
        try:
            longitude, latitude = self._split_location(text)
        except (TypeError, ValueError):
            return None
        return f"{longitude:.6f},{latitude:.6f}"

    def _split_location(self, value: str) -> tuple[float, float]:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError("location must be longitude,latitude")
        longitude = float(parts[0])
        latitude = float(parts[1])
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("location is out of range")
        return longitude, latitude

    def _regeocode_address(self, regeocode: dict[str, Any]) -> str:
        formatted = regeocode.get("formatted_address")
        if formatted:
            return str(formatted)

        component = regeocode.get("addressComponent")
        if not isinstance(component, dict):
            return ""
        parts = [
            component.get("province"),
            component.get("city"),
            component.get("district"),
            component.get("township"),
            (component.get("streetNumber") or {}).get("street")
            if isinstance(component.get("streetNumber"), dict)
            else None,
        ]
        return "".join(str(part) for part in parts if part not in [None, ""])

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
                tool_name="spring_boot.answer_price",
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
            tool_name="spring_boot.answer_price",
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
