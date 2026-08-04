from typing import Any

from app.schemas.agent import AgentPlan, ToolResult


class CardBuilder:
    def build(self, plan: AgentPlan, result: ToolResult) -> list[dict[str, Any]]:
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
        if plan.action == "recommend_coupons":
            return self.coupon_cards(result.data.get("coupons", []))
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
        return result.cards

    def movie_cards(self, movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "movie",
                "id": item.get("movieId"),
                "title": item.get("movieName"),
                "meta": {
                    "genre": item.get("genre"),
                    "score": item.get("score"),
                },
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

    def showtime_cards(self, showtimes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "showtime",
                "id": item.get("showtimeId"),
                "title": item.get("movieName"),
                "subtitle": f"{item.get('cinemaName')} {item.get('date')} {item.get('time')}",
                "meta": {"price": item.get("price"), "remainingSeats": item.get("remainingSeats")},
                "actions": [{"event": "select_showtime", "label": "选择这场", "payload": item}],
            }
            for item in showtimes
        ]

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
                "meta": {"price": item.get("price")},
                "actions": [{"event": "select_snacks", "label": "加入套餐", "payload": item}],
            }
            for item in snacks
        ]

    def coupon_cards(self, coupons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "coupon",
                "id": item.get("couponId"),
                "title": item.get("name"),
                "meta": {"discount": item.get("discount")},
                "actions": [{"event": "select_coupon", "label": "使用优惠", "payload": item}],
            }
            for item in coupons
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
                "payload": data,
                "actions": [{"event": "confirm_order", "label": "确认并继续"}],
            }
        ]

    def payment_cards(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "payment",
                "id": data.get("orderId") or "payment",
                "title": "模拟支付",
                "meta": {
                    "orderId": data.get("orderId"),
                    "status": data.get("status"),
                },
                "actions": [{"event": "pay_order", "label": "确认支付"}],
            }
        ]

    def ticket_cards(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "ticket",
                "id": data.get("orderId") or "ticket",
                "title": "电子票",
                "meta": {
                    "orderId": data.get("orderId"),
                    "ticketStatus": data.get("ticketStatus"),
                    "calendar": data.get("calendar"),
                    "notification": data.get("notification"),
                },
            }
        ]


card_builder = CardBuilder()
