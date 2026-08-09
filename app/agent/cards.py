from typing import Any
from datetime import datetime

from app.schemas.agent import AgentPlan, ToolResult


class CardBuilder:
    def build(self, plan: AgentPlan, result: ToolResult) -> list[dict[str, Any]]:
        if not result.success:
            return result.cards
        if result.data.get("showSnackRecommendations") and result.data.get("snacks"):
            return self.snack_cards(result.data.get("snacks", []))
        if result.data.get("paymentReady"):
            return self.payment_cards(result.data)
        if result.data.get("navigation"):
            return [self.navigation_card(result.data["navigation"])]
        if plan.action == "search_movies":
            return self.movie_cards(result.data.get("movies", []))
        if plan.action == "search_showtimes":
            return self.showtime_cards(result.data.get("showtimes", []))
        if plan.action == "get_seats":
            return self.seat_cards(result.data)
        if plan.action == "recommend_snacks":
            return self.snack_cards(result.data.get("snacks", []))
        if plan.action == "search_nearby_cinemas":
            return self.cinema_cards(result.data.get("cinemas", []))
        if plan.action == "confirm_selection":
            if not result.data.get("showtimeId") or not result.data.get("seatIds"):
                return []
            return self.confirm_cards(result.data)
        if plan.action in {"lock_seats", "create_order"}:
            return self.confirm_cards(result.data)
        if plan.action == "pay_order":
            if result.data.get("ticketStatus") == "issued":
                return self.ticket_cards(result.data)
            return self.payment_cards(result.data)
        if plan.action in {"refund_order", "get_refund_status"}:
            return self.refund_cards(result.data)
        if plan.action == "get_order":
            if result.data.get("ticketStatus") == "issued":
                return self.ticket_cards(result.data)
            return self.order_cards([result.data]) if result.data else []
        if plan.action == "list_orders":
            records = result.data.get("records", [])
            return self.order_cards(records)
        return result.cards

    def movie_cards(self, movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "movie",
                "id": item.get("movieId"),
                "title": item.get("movieName"),
                "image": item.get("posterUrl") or item.get("poster") or item.get("image"),
                "poster": item.get("poster") or item.get("posterUrl") or item.get("image"),
                "posterUrl": item.get("posterUrl") or item.get("poster") or item.get("image"),
                "meta": {
                    "类型：": item.get("genre"),
                    "时长：": self._format_duration(item.get("durationMinutes")),
                    "状态：": item.get("status"),
                    **(
                        {"评分：": item.get("score")}
                        if self._has_valid_score(item.get("score"))
                        else {}
                    ),
                },
                "payload": self._movie_payload(item),
                "actions": [
                    {
                        "event": "select_movie",
                        "label": "选择电影",
                        "payload": item,
                    }
                ],
            }
            for item in movies
        ]

    @staticmethod
    def _has_valid_score(value: Any) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    def _movie_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item)
        if not self._has_valid_score(payload.get("score")):
            payload.pop("score", None)
            payload.pop("rating", None)
        return payload

    def showtime_cards(self, showtimes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for item in showtimes:
            cinema = item.get("cinema") or {}
            hall = item.get("hall") or {}
            start_at = str(item.get("startAt") or "")
            cinema_name = item.get("cinemaName") or cinema.get("name")
            hall_name = item.get("hallName") or hall.get("name") or item.get("hallType")
            date = item.get("date") or start_at[:10]
            time = item.get("time") or start_at[11:16]
            end_at = str(item.get("endAt") or "")
            end_time = item.get("endTime") or end_at[11:16]
            payload = {
                **item,
                "cinemaName": cinema_name,
                "hallName": hall_name,
                "date": date,
                "time": time,
                "endTime": end_time,
            }
            subtitle = " · ".join(
                part for part in [
                    f"影院：{cinema_name}" if cinema_name else "",
                    self._distance_text(item.get("distance")),
                    f"影厅：{hall_name}" if hall_name else "",
                    f"{date} {time}{f' - {end_time}' if end_time else ''}".strip()
                    if date or time
                    else "",
                ] if part
            )
            cards.append(
                {
                    "type": "showtime",
                    "id": item.get("showtimeId"),
                    "title": item.get("movieName"),
                    "subtitle": subtitle,
                    "meta": {
                        "price": item.get("price"),
                        "remainingSeats": item.get("remainingSeats"),
                        "distance": item.get("distance"),
                    },
                    "payload": payload,
                    "actions": [{"event": "select_showtime", "label": "选择这场", "payload": payload}],
                }
            )
        return cards

    @staticmethod
    def _distance_text(value: Any) -> str:
        try:
            distance = float(value)
        except (TypeError, ValueError):
            return ""
        if distance < 0:
            return ""
        if distance < 1:
            return f"距离：{distance * 1000:.0f}米"
        return f"距离：{distance:.1f}公里"

    def navigation_card(self, navigation: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "navigation",
            "id": navigation.get("showtimeId") or navigation.get("path") or "navigation",
            "title": navigation.get("title") or "进入选座",
            "subtitle": navigation.get("subtitle"),
            "payload": navigation,
            "actions": [
                {
                    "event": "navigate",
                    "label": navigation.get("label") or "进入选座",
                    "payload": navigation,
                }
            ],
        }

    def seat_cards(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        seats = data.get("seats", [])
        raw = data.get("raw") or {}
        price = (
            data.get("price")
            or data.get("basePrice")
            or raw.get("basePrice")
            or (
                seats[0].get("price")
                if seats and isinstance(seats[0], dict)
                else None
            )
        )
        return [
            {
                "type": "seat_map",
                "id": data.get("showtimeId"),
                "title": "选择座位",
                "meta": {"price": price} if price is not None else {},
                "seats": seats,
                "actions": [{"event": "select_seats", "label": "确认座位"}],
            }
        ]

    def snack_cards(self, snacks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "snack",
                "id": item.get("snackId"),
                "title": item.get("name"),
                "image": item.get("image"),
                "meta": {
                    "price": item.get("price"),
                    "stock": item.get("availableStock"),
                },
                "payload": item,
                "actions": [{"event": "select_snacks", "label": "加入零食", "payload": item}],
            }
            for item in snacks
        ]

    def cinema_cards(self, cinemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "cinema",
                "id": item.get("cinemaId"),
                "title": item.get("cinemaName"),
                "meta": {
                    "distance": item.get("distance"),
                    "address": item.get("address"),
                    "district": item.get("district"),
                    "minPrice": item.get("minPrice"),
                    "services": item.get("services"),
                    "location": item.get("location"),
                    "tel": item.get("tel"),
                    "type": item.get("type"),
                },
                "actions": [{"event": "select_cinema", "label": "选择影院", "payload": item}],
            }
            for item in cinemas
        ]

    def confirm_cards(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "confirm_order",
                "id": data.get("orderId") or data.get("lockId") or "confirm",
                "title": "确认订单",
                "subtitle": self._order_subtitle(data),
                "meta": self._order_meta(data),
                "payload": data,
                "actions": [{"event": "confirm_order", "label": "确认并继续"}],
            }
        ]

    def payment_cards(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        qr_code = data.get("qrCode")
        return [
            {
                "type": "payment",
                "id": data.get("orderId") or "payment",
                "title": "确认支付",
                "subtitle": self._order_subtitle(data),
                "meta": self._order_meta(data),
                "qrCode": qr_code,
                "payload": data,
                "actions": [{
                    "event": "pay_order",
                    "label": "去支付宝付款",
                }],
            }
        ]

    def ticket_cards(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        order_id = data.get("orderId")
        return [
            {
                "type": "ticket",
                "id": order_id or "ticket",
                "title": "电子票",
                "meta": {
                    "orderId": order_id,
                    "ticketStatus": data.get("ticketStatus"),
                    "calendar": data.get("calendar"),
                    "notification": data.get("notification"),
                },
                "payload": data,
                "actions": [
                    {
                        "event": "view_ticket",
                        "label": "查看电子票",
                        "payload": {
                            "orderId": order_id,
                            "path": f"/orders/{order_id}/tickets" if order_id else "",
                        },
                    }
                ]
                if order_id
                else [],
            }
        ]

    def order_cards(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "order",
                "id": item.get("orderId") or item.get("orderNo") or "order",
                "title": item.get("movieName") or "订单详情",
                "subtitle": self._order_subtitle(item),
                "meta": self._order_meta(item),
                "payload": item,
                "actions": self._order_actions(item),
            }
            for item in orders
            if isinstance(item, dict)
        ]

    def refund_cards(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "refund",
                "id": data.get("orderId") or "refund",
                "title": "退票结果",
                "meta": {
                    "订单号": data.get("orderId"),
                    "状态": data.get("status"),
                    "金额": self._format_amount(data.get("amount")),
                    "手续费": self._format_amount(data.get("serviceFee")),
                    "退款请求号": data.get("outRequestNo"),
                    "更新时间": data.get("updatedAt"),
                },
                "payload": data,
            }
        ]

    def _order_meta(self, data: dict[str, Any]) -> dict[str, Any]:
        start_at = data.get("startAt")
        end_at = data.get("endAt")
        seats = self._seat_summary(data)
        meta = {
            "订单号": data.get("orderNo") or data.get("orderId"),
            "电影": data.get("movieName"),
            "影院": data.get("cinemaName"),
            "影厅": data.get("hallName"),
            "厅型": data.get("hallType"),
            "语言": data.get("language"),
            "日期": self._format_date(start_at) or data.get("date"),
            "开始": self._format_time(start_at) or data.get("time"),
            "结束": self._format_time(end_at),
            "座位": seats,
            "应付": self._format_amount(data.get("amount")),
            "状态": data.get("statusDesc") or data.get("status"),
        }
        return {key: value for key, value in meta.items() if value not in [None, ""]}

    def _order_actions(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        status = str(data.get("status") or "").upper()
        order_id = data.get("orderId") or data.get("orderNo")
        if not order_id:
            return []
        actions: list[dict[str, Any]] = [
            {"event": "get_order", "label": "查看订单", "payload": data}
        ]
        if status in {"TICKETED", "PAID", "SUCCESS"}:
            actions.append(
                {
                    "event": "refund_order",
                    "label": "申请退票",
                    "payload": data,
                }
            )
        if status in {"REFUNDING", "REFUND_PENDING", "REFUNDED"}:
            actions.append(
                {
                    "event": "get_refund_status",
                    "label": "查看退款",
                    "payload": data,
                }
            )
        return actions

    def _order_subtitle(self, data: dict[str, Any]) -> str:
        parts = [
            data.get("movieName"),
            data.get("cinemaName"),
            data.get("hallName"),
            self._format_time(data.get("startAt")) or data.get("time"),
        ]
        return " · ".join(str(part) for part in parts if part)

    def _seat_summary(self, data: dict[str, Any]) -> str:
        seats = data.get("seats") or data.get("seatIds") or []
        if not isinstance(seats, list):
            return str(seats) if seats else ""
        labels = []
        for seat in seats:
            if isinstance(seat, dict):
                row = seat.get("rowNo") or seat.get("row")
                number = seat.get("seatNo") or seat.get("number")
                if row not in [None, ""] and number not in [None, ""]:
                    labels.append(f"{row}排{number}座")
                    continue
                if seat.get("seatId"):
                    labels.append(str(seat["seatId"]))
                    continue
            elif seat not in [None, ""]:
                labels.append(str(seat))
        return "、".join(labels)

    def _format_amount(self, value: Any) -> str:
        if value in [None, ""]:
            return ""
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return str(value)
        formatted = str(int(amount)) if amount.is_integer() else f"{amount:.2f}".rstrip("0").rstrip(".")
        return f"{formatted}元"

    def _format_duration(self, value: Any) -> str:
        if value in [None, ""]:
            return ""
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{minutes}分钟"

    def _format_date(self, value: Any) -> str:
        parsed = self._parse_datetime(value)
        return parsed.strftime("%Y-%m-%d") if parsed else ""

    def _format_time(self, value: Any) -> str:
        parsed = self._parse_datetime(value)
        return parsed.strftime("%H:%M") if parsed else ""

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None


card_builder = CardBuilder()
