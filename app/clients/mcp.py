import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx

from app.schemas.agent import ToolResult
from app.utils.config_handler import agent_config


class MCPClient(Protocol):
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        pass


class SpringBootMovieTicketMCP:
    """Queries the real Spring Boot ticketing database for user-facing data."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout_seconds: int = 15,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        handlers = {
            "search_nearby_cinemas": self.search_nearby_cinemas,
            "search_movies": self.search_movies,
            "search_showtimes": self.search_showtimes,
            "get_seats": self.get_seats,
            "lock_seats": self.lock_seats,
            "recommend_snacks": self.recommend_snacks,
            "replace_order_snacks": self.replace_order_snacks,
            "create_order": self.create_order,
            "pay_order": self.pay_order,
            "issue_ticket": self.issue_ticket,
            "refund_order": self.refund_order,
            "get_refund_status": self.get_refund_status,
            "get_order": self.get_order,
            "list_orders": self.list_orders,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(
                tool_name=f"spring_boot.{tool_name}",
                success=False,
                message=f"Unknown Spring Boot tool: {tool_name}",
            )
        try:
            return handler(arguments)
        except (httpx.HTTPError, ValueError) as exc:
            error_text = str(exc)
            if "未登录" in error_text or "Token" in error_text or "token" in error_text:
                return ToolResult(
                    tool_name=f"spring_boot.{tool_name}",
                    success=False,
                    data={"error": "AUTH_REQUIRED"},
                    message="登录状态已失效，请重新登录后再查询电影票。",
                )
            if isinstance(exc, httpx.RequestError):
                message = "票务数据库暂时无法连接，请确认 Spring Boot 服务已启动。"
            elif error_text:
                message = error_text
            else:
                message = "票务数据库查询失败，请确认 Spring Boot 服务和登录状态正常。"
            return ToolResult(
                tool_name=f"spring_boot.{tool_name}",
                success=False,
                data={"error": error_text},
                message=message,
            )

    def search_movies(self, arguments: dict[str, Any]) -> ToolResult:
        keyword = arguments.get("movieName") or arguments.get("keyword")
        cinema_id = arguments.get("cinemaId")
        recommendation_criteria = arguments.get("recommendationCriteria")
        genre = arguments.get("genre")
        if (
            str(keyword or "").strip().casefold()
            == str(genre or "").strip().casefold()
        ):
            # "爱情电影" is a genre query, not a movie title containing "爱情".
            keyword = None
        if cinema_id:
            data = self._get_business(
                "/api/user/showtimes",
                arguments,
                {
                    "cinemaId": cinema_id,
                    "date": self._spring_date(arguments.get("date")),
                    "hallType": arguments.get("hallType"),
                },
            )
            all_movies = self._format_showtime_movies(data, arguments)
            movies = self._filter_movies(
                all_movies,
                keyword,
                arguments.get("genre"),
            )
            movies = self._filter_movie_cards_by_showtime_constraints(
                movies,
                arguments,
            )
            fallback_reason = None
            if not movies:
                movies, fallback_reason = self._fallback_cinema_movie_cards(
                    arguments,
                    all_movies,
                    keyword,
                )
            movies = self._rank_recommended_movies(
                movies,
                recommendation_criteria,
            )
            movies = self._apply_movie_limit(
                movies,
                arguments.get("movieLimit") or (3 if fallback_reason else None),
            )
            cinema = data.get("cinema") or {}
            cinema_name = (
                cinema.get("name")
                or arguments.get("cinemaName")
                or "当前影院"
            )
            return ToolResult(
                tool_name="spring_boot.search_movies",
                data={
                    "movies": movies,
                    "cinemaId": cinema.get("id") or cinema_id,
                    "cinemaName": cinema_name,
                    "recommendationCriteria": recommendation_criteria,
                    "fallbackReason": fallback_reason,
                    "source": "spring_boot_database",
                },
                message=self._fallback_movie_search_message(
                    fallback_reason,
                    arguments,
                    movies,
                )
                or self._movie_search_message(
                    movies,
                    recommendation_criteria,
                    cinema_name=cinema_name,
                    has_showtimes=True,
                    genre=arguments.get("genre"),
                    keyword=keyword,
                    date=arguments.get("date"),
                ),
            )

        all_movies = self._load_movie_cards(
            arguments,
            keyword=keyword,
            genre=arguments.get("genre"),
        )
        movies = self._filter_movie_cards_by_showtime_constraints(
            all_movies,
            arguments,
        )
        fallback_reason = None
        if not movies:
            movies, fallback_reason = self._fallback_movie_cards(
                arguments,
                all_movies,
            )
        movies = self._rank_recommended_movies(movies, recommendation_criteria)
        movies = self._apply_movie_limit(
            movies,
            arguments.get("movieLimit") or (3 if fallback_reason else None),
        )
        return ToolResult(
            tool_name="spring_boot.search_movies",
            data={
                "movies": movies,
                "recommendationCriteria": recommendation_criteria,
                "fallbackReason": fallback_reason,
                "source": "spring_boot_database",
            },
            message=self._fallback_movie_search_message(
                fallback_reason,
                arguments,
                movies,
            )
            or self._movie_search_message(
                movies,
                recommendation_criteria,
                genre=arguments.get("genre"),
                keyword=keyword,
                date=arguments.get("date"),
            ),
        )

    def search_showtimes(self, arguments: dict[str, Any]) -> ToolResult:
        movie_id = arguments.get("movieId")
        movie_name = str(arguments.get("movieName") or "").strip()
        movie_ids: list[Any] = []

        if not movie_id and movie_name:
            movie_result = self.search_movies({**arguments, "keyword": movie_name})
            if not movie_result.success:
                return movie_result
            movies = movie_result.data.get("movies", [])
            movie = self._pick_movie(movies, movie_name)
            if not movie:
                return ToolResult(
                    tool_name="spring_boot.search_showtimes",
                    data={"showtimes": [], "movies": movies},
                    message=f"没有找到《{movie_name}》的相关信息。",
                )
            movie_id = movie.get("movieId")
        elif not movie_id and arguments.get("genre"):
            movie_result = self.search_movies({**arguments, "keyword": None, "movieName": None})
            if not movie_result.success:
                return movie_result
            movies = movie_result.data.get("movies", [])
            movie_ids = [movie.get("movieId") for movie in movies if movie.get("movieId")]
            if not movie_ids:
                return ToolResult(
                    tool_name="spring_boot.search_showtimes",
                    data={"showtimes": [], "movies": movies},
                    message=f"没有找到{arguments.get('genre')}类型的影片，换个类型试试？",
                )

        date_value = self._spring_date(arguments.get("date"))
        showtimes: list[dict[str, Any]] = []
        query_movie_ids = movie_ids[:5] if movie_ids else [movie_id]
        for query_movie_id in query_movie_ids:
            data = self._get_business(
                "/api/user/showtimes",
                arguments,
                {
                    "movieId": query_movie_id,
                    "cinemaId": arguments.get("cinemaId"),
                    "date": date_value,
                    "hallType": arguments.get("hallType"),
                },
            )
            showtimes.extend(self._format_showtime_groups(data))
        showtimes = self._filter_showtimes(
            showtimes,
            date_value,
            arguments.get("timeRange"),
            arguments.get("ticketCount"),
            arguments.get("pricePreference"),
            arguments.get("timePreference"),
        )
        response_data: dict[str, Any] = {
            "showtimes": showtimes,
            "source": "spring_boot_database",
        }
        if self._should_auto_navigate(arguments, showtimes):
            showtime = showtimes[0]
            response_data["showtimeId"] = showtime.get("showtimeId")
            response_data["navigation"] = {
                "type": "seat_selection",
                "showtimeId": showtime.get("showtimeId"),
                "path": f"/showtimes/{showtime.get('showtimeId')}/seats",
                "title": "已匹配到场次，正在进入选座",
                "subtitle": self._showtime_subtitle(showtime),
                "label": "进入选座",
            }
        return ToolResult(
            tool_name="spring_boot.search_showtimes",
            data=response_data,
            message=self._showtime_search_message(arguments, showtimes, response_data),
            suggestions=(
                ["换一家影院", "换个时间", "换个类型"]
                if not showtimes
                else []
            ),
        )

    def get_seats(self, arguments: dict[str, Any]) -> ToolResult:
        showtime_id = arguments.get("showtimeId")
        if not showtime_id:
            return ToolResult(
                tool_name="spring_boot.get_seats",
                success=False,
                data={"error": "SHOWTIME_REQUIRED"},
                message="缺少场次 ID，无法查询座位图。",
            )
        data = self._get_business(
            f"/api/user/showtimes/{showtime_id}/seats",
            arguments,
            {},
        )
        seats = []
        for row in data.get("rows") or []:
            row_no = self._first_present(row, "rowNo", "row")
            for seat in row.get("seats") or []:
                seats.append(
                    {
                        "seatId": self._first_present(seat, "id", "seatId"),
                        "row": self._first_present(seat, "rowNo", "row") or row_no,
                        "number": self._first_present(seat, "seatNo", "number"),
                        "status": self._normalize_seat_status(
                            self._first_present(seat, "status", "seatStatus")
                        ),
                        "price": self._first_present(seat, "price", "unitPrice")
                        or data.get("basePrice"),
                    }
                )
        return ToolResult(
            tool_name="spring_boot.get_seats",
            data={
                "showtimeId": data.get("showtimeId") or showtime_id,
                "seats": seats,
                "raw": data,
            },
            message="已从票务数据库加载座位图。",
        )

    def _normalize_seat_status(self, value: Any) -> str:
        numeric_statuses = {
            0: "available",
            1: "locked",
            2: "sold",
            3: "unavailable",
            4: "couple",
        }
        if isinstance(value, int):
            return numeric_statuses.get(value, "unavailable")
        text = str(value or "AVAILABLE").strip().lower()
        return {
            "available": "available",
            "locked": "locked",
            "sold": "sold",
            "unavailable": "unavailable",
            "couple": "couple",
        }.get(text, "unavailable")

    def lock_seats(self, arguments: dict[str, Any]) -> ToolResult:
        """Persist the AI purchase draft, then use Java's real lock-and-order API."""
        showtime_id = self._required_long(arguments.get("showtimeId"), "场次 ID")
        seat_ids = self._long_list(arguments.get("seatIds"), "座位 ID")
        if not seat_ids:
            raise ValueError("请选择至少一个座位。")

        draft = self._sync_purchase_draft(arguments, showtime_id, seat_ids)
        draft_version = self._required_draft_version(draft)
        raw = self._post_business(
            "/api/user/orders/lock",
            arguments,
            {
                "showtimeId": showtime_id,
                "seatIds": seat_ids,
                "draftVersion": draft_version,
            },
        )
        movie = raw.get("movie") or {}
        cinema = raw.get("cinema") or {}
        data = {
            "orderId": raw.get("orderId"),
            "orderNo": raw.get("orderNo"),
            "showtimeId": showtime_id,
            "seatIds": seat_ids,
            "amount": raw.get("amount"),
            "status": "PAYMENT_PENDING",
            "expiresAt": raw.get("expiresAt"),
            "remainingSeconds": raw.get("remainingSeconds"),
            "movieId": movie.get("id") or arguments.get("movieId"),
            "movieName": movie.get("name") or arguments.get("movieName"),
            "cinemaId": cinema.get("id") or arguments.get("cinemaId"),
            "cinemaName": cinema.get("name") or arguments.get("cinemaName"),
            "hallName": raw.get("hallName") or arguments.get("hallName"),
            "hallType": raw.get("hallType") or arguments.get("hallType"),
            "language": raw.get("language") or arguments.get("language"),
            "date": raw.get("date") or arguments.get("date"),
            "time": raw.get("time") or arguments.get("time"),
            "startAt": raw.get("startAt") or arguments.get("startAt"),
            "endAt": raw.get("endAt") or arguments.get("endAt"),
            "seats": raw.get("seats") or [],
            "draftVersion": draft_version,
            "orderCreated": True,
            "source": "spring_boot_database",
        }
        return ToolResult(
            tool_name="spring_boot.lock_seats",
            data=data,
            message=f"已锁定 {len(seat_ids)} 个座位，订单已创建。",
        )

    def recommend_snacks(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = self._required_long(arguments.get("orderId"), "订单 ID")
        raw = self._get_business(
            f"/api/user/orders/{order_id}/snacks",
            arguments,
            {},
        )
        options = raw.get("options") or []
        snacks = [
            self._format_snack_option(item)
            for item in options
            if isinstance(item, dict)
        ]
        cinema_name = raw.get("cinemaName") or arguments.get("cinemaName") or "当前影院"
        return ToolResult(
            tool_name="spring_boot.recommend_snacks",
            data={
                "orderId": raw.get("orderId") or order_id,
                "cinemaId": raw.get("cinemaId") or arguments.get("cinemaId"),
                "cinemaName": cinema_name,
                "ticketAmount": raw.get("ticketAmount"),
                "snackAmount": raw.get("snackAmount"),
                "totalAmount": raw.get("totalAmount"),
                "snacks": snacks,
                "source": "spring_boot_database",
            },
            message=f"已从{cinema_name}加载 {len(snacks)} 个可选零食套餐。",
        )

    def replace_order_snacks(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = self._required_long(arguments.get("orderId"), "订单 ID")
        items = self._snack_items(arguments)
        raw = self._put_business(
            f"/api/user/orders/{order_id}/snacks",
            arguments,
            {"items": items},
        )
        data = self._format_snack_selection(raw, fallback_order_id=order_id)
        return ToolResult(
            tool_name="spring_boot.replace_order_snacks",
            data=data,
            message=(
                "零食已加入订单，"
                f"零食金额 {self._format_yuan(data.get('snackAmount'))}，"
                f"合计 {self._format_yuan(data.get('totalAmount'))}。"
            ),
        )

    def refund_order(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = self._required_long(arguments.get("orderId"), "订单 ID")
        raw = self._post_business(
            f"/api/user/orders/{order_id}/refund",
            arguments,
            {},
        )
        data = self._format_refund(raw, fallback_order_id=order_id)
        return ToolResult(
            tool_name="spring_boot.refund_order",
            data=data,
            message=data.get("message") or self._refund_message(data),
        )

    def get_refund_status(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = self._required_long(arguments.get("orderId"), "订单 ID")
        raw = self._get_business(
            f"/api/user/orders/{order_id}/refund",
            arguments,
            {},
        )
        data = self._format_refund(raw, fallback_order_id=order_id)
        return ToolResult(
            tool_name="spring_boot.get_refund_status",
            data=data,
            message=data.get("message") or self._refund_message(data),
        )

    def create_order(self, arguments: dict[str, Any]) -> ToolResult:
        """Compatibility wrapper: Java creates the order inside /orders/lock."""
        order_id = self._required_long(arguments.get("orderId"), "订单 ID")
        result = self.get_order({**arguments, "orderId": order_id})
        if not result.success:
            return result
        result.tool_name = "spring_boot.create_order"
        result.message = "订单已创建，等待支付。"
        return result

    def pay_order(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = self._required_long(arguments.get("orderId"), "订单 ID")
        idempotency_key = str(
            arguments.get("idempotencyKey")
            or f"agent-{uuid.uuid4().hex}"
        )
        raw = self._post_business(
            f"/api/user/orders/{order_id}/pay/qrcode",
            arguments,
            {"idempotencyKey": idempotency_key},
        )
        payment_status = str(raw.get("paymentStatus") or raw.get("status") or "").upper()
        qr_code = str(raw.get("qrCode") or raw.get("payForm") or "").strip()

        if payment_status != "SUCCESS" and not qr_code:
            raise ValueError("支付宝沙箱二维码未返回。")

        order_result = self.get_order({**arguments, "orderId": order_id})
        data = dict(order_result.data)
        tickets = data.get("tickets") or []
        ticket_status = "issued" if str(data.get("status") or "").upper() == "TICKETED" or tickets else None

        data.update({
            "orderId": data.get("orderId") or raw.get("orderId") or order_id,
            "paymentStatus": payment_status or "PENDING",
            "status": data.get("status") or ("PAYMENT_PENDING" if payment_status != "SUCCESS" else "SUCCESS"),
            "paidAmount": data.get("amount"),
            "qrCode": qr_code or None,
            "ticketStatus": ticket_status,
            "ticketCodes": [
                item.get("ticketCode")
                for item in tickets
                if isinstance(item, dict) and item.get("ticketCode")
            ],
            "source": "spring_boot_database",
        })
        return ToolResult(
            tool_name="spring_boot.pay_order",
            data=data,
            message=(
                "支付成功，电子票已出票。"
                if data["ticketStatus"] == "issued"
                else "支付成功。"
                if payment_status == "SUCCESS"
                else "支付二维码已生成，请扫码支付。"
            ),
        )

    def issue_ticket(self, arguments: dict[str, Any]) -> ToolResult:
        """Java's pay endpoint issues tickets in the same transaction."""
        order_id = self._required_long(arguments.get("orderId"), "订单 ID")
        result = self.get_order({**arguments, "orderId": order_id})
        if not result.success:
            return result
        data = dict(result.data)
        data["ticketStatus"] = "issued" if str(data.get("status", "")).upper() == "TICKETED" else None
        data["source"] = "spring_boot_database"
        return ToolResult(
            tool_name="spring_boot.issue_ticket",
            data=data,
            message="订单已经出票。" if data.get("ticketStatus") == "issued" else "订单尚未出票。",
        )

    def get_order(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = self._required_long(arguments.get("orderId"), "订单 ID")
        raw = self._get_business(
            f"/api/user/orders/{order_id}",
            arguments,
            {},
        )
        data = self._format_order(raw, order_id)
        status = str(data.get("status") or "").upper()
        if status == "TICKETED" or data.get("ticketStatus") == "issued":
            message = "支付成功，电子票已出票。"
        elif status in {"PAYMENT_PENDING", "PENDING"}:
            message = "订单仍待支付，请完成支付宝沙箱支付后再查看。"
        elif status in {"CANCELLED", "CANCELED", "CLOSED"}:
            message = "订单已关闭或取消。"
        else:
            message = "订单已加载。"
        return ToolResult(
            tool_name="spring_boot.get_order",
            data=data,
            message=message,
        )

    def list_orders(self, arguments: dict[str, Any]) -> ToolResult:
        raw = self._get_business(
            "/api/user/orders",
            arguments,
            {
                "page": arguments.get("page", 1),
                "size": 5,
                "status": arguments.get("status"),
            },
        )
        records = raw.get("records") or []
        return ToolResult(
            tool_name="spring_boot.list_orders",
            data={
                "records": [self._format_order(item) for item in records],
                "total": raw.get("total", len(records)),
                "page": raw.get("page", arguments.get("page", 1)),
                "size": raw.get("size", 5),
                "source": "spring_boot_database",
            },
            message="目前只展示近期5笔订单，更多订单请查看订单列表。",
        )

    def search_nearby_cinemas(self, arguments: dict[str, Any]) -> ToolResult:
        location = AMapMCP()._normalize_location(arguments.get("location"))
        if not location:
            return ToolResult(
                tool_name="spring_boot.search_nearby_cinemas",
                success=False,
                data={"error": "LOCATION_REQUIRED"},
                message="未获取到当前位置，无法按数据库距离匹配影院。请允许浏览器定位后重试。",
            )

        lng, lat = self._split_location(location)
        data = self._get_business(
            "/api/user/cinemas/nearby",
            arguments,
            {
                "page": arguments.get("page", 1),
                "size": arguments.get("size", 10),
                "lat": lat,
                "lng": lng,
                "radius": arguments.get("radius", 5),
            },
        )
        records = data.get("records") or []
        cinemas = [self._format_cinema(item) for item in records]
        return ToolResult(
            tool_name="spring_boot.search_nearby_cinemas",
            data={
                "cinemas": cinemas,
                "source": "spring_boot_database",
                "queryLocation": {"longitude": lng, "latitude": lat},
                "total": data.get("total", len(cinemas)),
            },
            message=(
                f"已按当前位置在票务数据库中找到 {len(cinemas)} 家影院。"
                if cinemas
                else "当前位置附近暂未匹配到数据库中的影院。"
            ),
        )

    def _split_location(self, location: str) -> tuple[float, float]:
        parts = [part.strip() for part in location.split(",")]
        if len(parts) != 2:
            raise ValueError("location must be longitude,latitude")
        return float(parts[0]), float(parts[1])

    def _headers(self, arguments: dict[str, Any]) -> dict[str, str]:
        jwt = str(arguments.get("jwt") or "").strip()
        return {"Authorization": f"Bearer {jwt}"} if jwt else {}

    def _required_long(self, value: Any, label: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label}无效。")
        if parsed <= 0:
            raise ValueError(f"{label}无效。")
        return parsed

    def _first_present(self, data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = data.get(key)
            if value not in [None, ""]:
                return value
        return None

    def _long_list(self, values: Any, label: str) -> list[int]:
        if not isinstance(values, list):
            raise ValueError(f"{label}列表无效。")
        result = []
        for value in values:
            result.append(self._required_long(value, label))
        return result

    def _required_draft_version(self, draft: dict[str, Any]) -> int:
        try:
            version = int(draft.get("version", 0))
        except (TypeError, ValueError):
            version = 0
        if version < 0:
            raise ValueError("购票草稿版本号无效。")
        return version

    def _sync_purchase_draft(
        self,
        arguments: dict[str, Any],
        showtime_id: int,
        seat_ids: list[int],
    ) -> dict[str, Any]:
        current = self._get_business(
            "/api/user/draft/current",
            arguments,
            {},
        )
        version = self._required_draft_version(current)
        body: dict[str, Any] = {
            "version": version,
            "showtimeId": showtime_id,
            "ticketCount": len(seat_ids),
            "seats": seat_ids,
            "sourceMode": "AI",
        }
        for key in ("movieId", "cinemaId"):
            value = arguments.get(key)
            if value in [None, ""]:
                continue
            try:
                body[key] = self._required_long(value, key)
            except ValueError:
                # NLU may still carry a non-database identifier; Java only accepts Long.
                continue
        start_at = arguments.get("startAt")
        end_at = arguments.get("endAt")
        if start_at or end_at:
            body["dateTime"] = {
                "start": str(start_at or ""),
                "end": str(end_at or ""),
            }
        return self._post_business(
            "/api/user/draft",
            arguments,
            body,
        )

    def _format_order(
        self,
        raw: dict[str, Any],
        fallback_order_id: Any = None,
    ) -> dict[str, Any]:
        order_id = raw.get("id") or raw.get("orderId") or fallback_order_id
        movie = raw.get("movie") or {}
        cinema = raw.get("cinema") or {}
        items = raw.get("items") or []
        seats = [
            {
                "rowNo": item.get("rowNo"),
                "seatNo": item.get("seatNo"),
                "zone": item.get("zone"),
                "price": item.get("unitPrice"),
                "ticketCode": item.get("ticketCode"),
            }
            for item in items
            if isinstance(item, dict)
        ]
        return {
            "orderId": order_id,
            "orderNo": raw.get("orderNo"),
            "showtimeId": raw.get("showtimeId"),
            "movieId": movie.get("id"),
            "movieName": movie.get("name") or raw.get("movieName"),
            "cinemaId": cinema.get("id"),
            "cinemaName": cinema.get("name") or raw.get("cinemaName"),
            "hallName": raw.get("hallName"),
            "startAt": raw.get("startAt"),
            "endAt": raw.get("endAt"),
            "amount": raw.get("amount"),
            "status": raw.get("status"),
            "statusDesc": raw.get("statusDesc"),
            "expiresAt": raw.get("expiresAt"),
            "seats": seats,
            "tickets": raw.get("tickets") or [],
            "ticketStatus": "issued"
            if str(raw.get("status") or "").upper() == "TICKETED"
            else None,
            "source": "spring_boot_database",
        }

    def _format_snack_option(self, raw: dict[str, Any]) -> dict[str, Any]:
        price_fen = raw.get("priceFen")
        try:
            price = int(price_fen) / 100 if price_fen not in [None, ""] else None
        except (TypeError, ValueError):
            price = None
        return {
            "snackId": raw.get("id") or raw.get("snackId"),
            "name": raw.get("name"),
            "description": raw.get("description"),
            "image": raw.get("image"),
            "price": price,
            "priceFen": price_fen,
            "availableStock": raw.get("availableStock"),
            "selectedQuantity": raw.get("selectedQuantity"),
            "status": raw.get("status"),
        }

    def _snack_items(self, arguments: dict[str, Any]) -> list[dict[str, int]]:
        raw_items = arguments.get("snackItems") or arguments.get("items")
        if isinstance(raw_items, list) and raw_items:
            items: list[dict[str, int]] = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                snack_id = self._required_long(item.get("snackId"), "零食 ID")
                quantity = int(item.get("quantity") or 1)
                if quantity <= 0:
                    continue
                items.append({"snackId": snack_id, "quantity": quantity})
            return items

        snack_ids = self._long_list(arguments.get("snackIds") or [], "零食 ID")
        return [{"snackId": snack_id, "quantity": 1} for snack_id in snack_ids]

    def _format_snack_selection(
        self,
        raw: dict[str, Any],
        fallback_order_id: int | None = None,
    ) -> dict[str, Any]:
        selected = raw.get("selected") or raw.get("items") or []
        options = raw.get("options") or []
        data = {
            "orderId": raw.get("orderId") or fallback_order_id,
            "cinemaId": raw.get("cinemaId"),
            "cinemaName": raw.get("cinemaName"),
            "ticketAmount": raw.get("ticketAmount"),
            "snackAmount": raw.get("snackAmount"),
            "totalAmount": raw.get("totalAmount"),
            "selectedSnacks": [
                self._format_order_snack_item(item)
                for item in selected
                if isinstance(item, dict)
            ],
            "snacks": [
                self._format_snack_option(item)
                for item in options
                if isinstance(item, dict)
            ],
            "source": "spring_boot_database",
        }
        if data["totalAmount"] not in [None, ""]:
            data["amount"] = data["totalAmount"]
        return data

    def _format_order_snack_item(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "snackId": raw.get("snackId"),
            "name": raw.get("name") or raw.get("snackName"),
            "image": raw.get("image"),
            "unitPrice": raw.get("unitPrice"),
            "quantity": raw.get("quantity"),
            "amount": raw.get("amount"),
            "inventoryStatus": raw.get("inventoryStatus"),
        }

    def _format_refund(
        self,
        raw: dict[str, Any],
        fallback_order_id: int | None = None,
    ) -> dict[str, Any]:
        return {
            "orderId": raw.get("orderId") or fallback_order_id,
            "status": raw.get("status"),
            "amount": raw.get("amount"),
            "serviceFee": raw.get("serviceFee"),
            "outRequestNo": raw.get("outRequestNo"),
            "message": raw.get("message"),
            "updatedAt": raw.get("updatedAt"),
            "source": "spring_boot_database",
        }

    def _refund_message(self, data: dict[str, Any]) -> str:
        status = str(data.get("status") or "").upper()
        amount = self._format_yuan(data.get("amount"))
        if status == "SUCCESS":
            return f"退票申请已完成，退款金额 {amount}。"
        if status == "FAIL":
            return data.get("message") or "退票失败，请查看订单详情或联系客服。"
        return f"退票申请已提交，当前状态：{data.get('status') or '处理中'}，金额 {amount}。"

    def _format_yuan(self, value: Any) -> str:
        if value in [None, ""]:
            return "0元"
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return f"{value}元"
        formatted = str(int(amount)) if amount.is_integer() else f"{amount:.2f}".rstrip("0").rstrip(".")
        return f"{formatted}元"

    def _get_business(
        self,
        path: str,
        arguments: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value not in [None, ""]},
            headers=self._headers(arguments),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in [0, 1, None]:
            message = payload.get("msg") or payload.get("message") or "票务数据库查询失败。"
            raise ValueError(message)
        return payload.get("data") or {}

    def _post_business(
        self,
        path: str,
        arguments: dict[str, Any],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}{path}",
            json=body,
            headers=self._headers(arguments),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in [0, 1, None]:
            message = payload.get("msg") or payload.get("message") or "票务数据库请求失败。"
            raise ValueError(message)
        return payload.get("data") or {}

    def _put_business(
        self,
        path: str,
        arguments: dict[str, Any],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        response = httpx.put(
            f"{self.base_url}{path}",
            json=body,
            headers=self._headers(arguments),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in [0, 1, None]:
            message = payload.get("msg") or payload.get("message") or "票务数据库请求失败。"
            raise ValueError(message)
        return payload.get("data") or {}

    def _load_movie_cards(
        self,
        arguments: dict[str, Any],
        keyword: Any = None,
        genre: Any = None,
        date: Any = None,
    ) -> list[dict[str, Any]]:
        data = self._get_business(
            "/api/agent/movies",
            arguments,
            {
                "page": arguments.get("page", 1),
                "size": arguments.get("size", 10),
                "keyword": keyword,
                "genre": genre,
                "date": date or self._spring_date(arguments.get("date")),
                "status": arguments.get("status"),
            },
        )
        return [self._format_movie(item) for item in data.get("records") or []]

    def _fallback_movie_cards(
        self,
        arguments: dict[str, Any],
        requested_movies: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not self._has_showtime_constraints(arguments):
            return [], None

        today_movies = self._today_high_rating_fallback(arguments)
        if today_movies:
            return today_movies, "today_high_rating"

        tomorrow_movies = self._tomorrow_same_type_fallback(
            arguments,
            requested_movies,
        )
        if tomorrow_movies:
            return tomorrow_movies, "tomorrow_same_type"

        return [], None

    def _fallback_cinema_movie_cards(
        self,
        arguments: dict[str, Any],
        all_movies: list[dict[str, Any]],
        keyword: Any,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not self._has_showtime_constraints(arguments):
            return [], None

        today_args = {
            **arguments,
            "timeRange": None,
            "movieLimit": arguments.get("movieLimit") or 3,
        }
        today_movies = self._filter_movie_cards_by_showtime_constraints(
            all_movies,
            today_args,
        )
        today_movies = self._rank_recommended_movies(today_movies, "high_rating")
        today_movies = self._apply_movie_limit(today_movies, today_args["movieLimit"])
        if today_movies:
            return today_movies, "today_high_rating"

        tomorrow_args = {
            **arguments,
            "date": "tomorrow",
            "movieLimit": arguments.get("movieLimit") or 3,
        }
        data = self._get_business(
            "/api/user/showtimes",
            arguments,
            {
                "cinemaId": arguments.get("cinemaId"),
                "date": self._spring_date(tomorrow_args.get("date")),
                "hallType": arguments.get("hallType"),
            },
        )
        tomorrow_movies = self._format_showtime_movies(data, tomorrow_args)
        tomorrow_movies = self._filter_movies(
            tomorrow_movies,
            keyword,
            arguments.get("genre"),
        )
        tomorrow_movies = self._filter_movie_cards_by_showtime_constraints(
            tomorrow_movies,
            tomorrow_args,
        )
        tomorrow_movies = self._rank_recommended_movies(
            tomorrow_movies,
            arguments.get("recommendationCriteria"),
        )
        tomorrow_movies = self._apply_movie_limit(
            tomorrow_movies,
            tomorrow_args["movieLimit"],
        )
        if tomorrow_movies:
            return tomorrow_movies, "tomorrow_same_type"

        return [], None

    def _today_high_rating_fallback(
        self,
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not arguments.get("date"):
            return []
        fallback_args = {
            **arguments,
            "date": "today",
            "timeRange": None,
            "movieLimit": arguments.get("movieLimit") or 3,
        }
        movies = self._load_movie_cards(
            fallback_args,
            keyword=None,
            genre=None,
        )
        movies = self._filter_movie_cards_by_showtime_constraints(
            movies,
            fallback_args,
        )
        movies = self._rank_recommended_movies(movies, "high_rating")
        return self._apply_movie_limit(movies, fallback_args["movieLimit"])

    def _tomorrow_same_type_fallback(
        self,
        arguments: dict[str, Any],
        requested_movies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not arguments.get("genre") and not arguments.get("movieName"):
            return []
        fallback_args = {
            **arguments,
            "date": "tomorrow",
            "movieLimit": arguments.get("movieLimit") or 3,
        }
        movies = requested_movies or self._load_movie_cards(
            fallback_args,
            keyword=arguments.get("movieName") or arguments.get("keyword"),
            genre=arguments.get("genre"),
        )
        movies = self._filter_movie_cards_by_showtime_constraints(
            movies,
            fallback_args,
        )
        movies = self._rank_recommended_movies(
            movies,
            arguments.get("recommendationCriteria"),
        )
        return self._apply_movie_limit(movies, fallback_args["movieLimit"])

    @staticmethod
    def _has_showtime_constraints(arguments: dict[str, Any]) -> bool:
        return any(
            arguments.get(key)
            for key in ["date", "timeRange", "ticketCount", "hallType"]
        )

    def _format_movie(self, item: dict[str, Any]) -> dict[str, Any]:
        poster = (
            item.get("posterUrl")
            or item.get("poster")
            or item.get("moviePoster")
            or item.get("cover")
            or item.get("coverUrl")
            or item.get("image")
        )
        upcoming = item.get("upcomingShowtimes") or []
        showtimes = [
            {
                "showtimeId": s.get("showtimeId"),
                "cinemaName": s.get("cinemaName"),
                "hallName": s.get("hallName"),
                "startAt": s.get("startAt"),
                "date": str(s.get("startAt") or "")[:10],
                "time": str(s.get("startAt") or "")[11:16],
                "endAt": s.get("endAt"),
                "price": s.get("price"),
                "remainingSeats": s.get("remainingSeats"),
            }
            for s in upcoming
            if isinstance(s, dict)
        ]
        return {
            "movieId": item.get("id") or item.get("movieId"),
            "movieName": item.get("name") or item.get("movieName"),
            "genre": item.get("genre"),
            "score": item.get("rating") or item.get("score"),
            "durationMinutes": item.get("duration"),
            "poster": poster,
            "posterUrl": poster,
            "status": item.get("statusDesc") or item.get("status"),
            "upcomingShowtimes": showtimes,
        }

    def _format_showtime_movies(
        self,
        data: dict[str, Any],
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:
        cinema = data.get("cinema") or {}
        movies: list[dict[str, Any]] = []
        for group in data.get("movies") or []:
            showtimes = group.get("showtimes") or []
            formatted_showtimes = [
                {
                    "showtimeId": item.get("id") or item.get("showtimeId"),
                    "cinemaName": cinema.get("name") or arguments.get("cinemaName"),
                    "hallName": item.get("hallName"),
                    "startAt": item.get("startAt"),
                    "date": str(item.get("startAt") or "")[:10],
                    "time": str(item.get("startAt") or "")[11:16],
                    "price": item.get("basePrice") or item.get("price"),
                    "remainingSeats": item.get("remainingSeats"),
                }
                for item in showtimes
                if isinstance(item, dict)
            ]
            prices = [
                item.get("price")
                for item in formatted_showtimes
                if item.get("price") not in [None, ""]
            ]
            remaining_seats = [
                item.get("remainingSeats")
                for item in formatted_showtimes
                if item.get("remainingSeats") not in [None, ""]
            ]
            movies.append(
                {
                    "movieId": group.get("id") or group.get("movieId"),
                    "movieName": group.get("name") or group.get("movieName"),
                    "genre": group.get("genre"),
                    "score": group.get("rating") or group.get("score"),
                    "durationMinutes": group.get("duration")
                    or group.get("durationMinutes"),
                    "poster": group.get("poster") or group.get("posterUrl"),
                    "posterUrl": group.get("posterUrl") or group.get("poster"),
                    "status": group.get("statusDesc") or group.get("status"),
                    "showtimeCount": len(formatted_showtimes),
                    "minPrice": min(prices) if prices else None,
                    "remainingSeats": sum(int(value) for value in remaining_seats),
                    "cinemaId": cinema.get("id") or arguments.get("cinemaId"),
                    "cinemaName": cinema.get("name") or arguments.get("cinemaName"),
                    "upcomingShowtimes": formatted_showtimes,
                }
            )
        return movies

    def _filter_movie_cards_by_showtime_constraints(
        self,
        movies: list[dict[str, Any]],
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:
        date_value = self._spring_date(arguments.get("date"))
        time_range = arguments.get("timeRange")
        ticket_count = arguments.get("ticketCount")
        if not date_value and not time_range and not ticket_count:
            return movies

        filtered_movies: list[dict[str, Any]] = []
        for movie in movies:
            showtimes = movie.get("upcomingShowtimes") or []
            if not showtimes:
                continue

            matching = self._filter_showtimes_by_constraints(
                showtimes,
                date_value,
                time_range,
                ticket_count,
            )
            if not matching:
                continue

            updated = dict(movie)
            updated["upcomingShowtimes"] = matching
            updated["showtimeCount"] = len(matching)
            prices = [
                item.get("price")
                for item in matching
                if item.get("price") not in [None, ""]
            ]
            if prices:
                updated["minPrice"] = min(prices)
            remaining_seats = [
                item.get("remainingSeats")
                for item in matching
                if item.get("remainingSeats") not in [None, ""]
            ]
            if remaining_seats:
                updated["remainingSeats"] = sum(int(value) for value in remaining_seats)
            filtered_movies.append(updated)

        return filtered_movies

    def _filter_showtimes_by_constraints(
        self,
        showtimes: list[dict[str, Any]],
        date_value: str | None,
        time_range: Any,
        ticket_count: Any,
    ) -> list[dict[str, Any]]:
        min_seats = int(ticket_count or 0)
        requested_time = str(time_range or "")
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        filtered: list[dict[str, Any]] = []
        for showtime in showtimes:
            start_at = self._parse_showtime_datetime(showtime.get("startAt"))
            if start_at is not None and start_at <= now:
                continue
            showtime_date = str(showtime.get("date") or showtime.get("startAt") or "")[:10]
            if date_value and showtime_date != date_value:
                continue
            remaining = showtime.get("remainingSeats")
            if min_seats and remaining is not None and int(remaining) < min_seats:
                continue
            showtime_time = showtime.get("time") or str(showtime.get("startAt") or "")[11:16]
            if requested_time and not self._time_matches(showtime_time, requested_time):
                continue
            filtered.append(showtime)
        return filtered

    def _filter_movies(
        self,
        movies: list[dict[str, Any]],
        keyword: Any,
        genre: Any,
    ) -> list[dict[str, Any]]:
        normalized_keyword = str(keyword or "").strip().casefold()
        normalized_genre = str(genre or "").strip().casefold()
        filtered = []
        for movie in movies:
            movie_name = str(movie.get("movieName") or "").casefold()
            movie_genre = str(movie.get("genre") or "").casefold()
            if normalized_keyword and normalized_keyword not in movie_name:
                continue
            if normalized_genre and movie_genre and normalized_genre not in movie_genre:
                continue
            filtered.append(movie)
        return filtered

    def _rank_recommended_movies(
        self,
        movies: list[dict[str, Any]],
        criteria: Any,
    ) -> list[dict[str, Any]]:
        normalized_criteria = str(criteria or "").strip()
        if not normalized_criteria:
            return movies

        genre_ranks = {
            "couple": {"爱情": 3, "喜剧": 2, "动画": 1},
            "family": {"动画": 3, "喜剧": 2, "爱情": 1},
        }

        def number(movie: dict[str, Any], *keys: str) -> float:
            for key in keys:
                try:
                    return float(movie.get(key))
                except (TypeError, ValueError):
                    continue
            return 0.0

        def rank(movie: dict[str, Any]) -> tuple[float, ...]:
            genre = str(movie.get("genre") or "")
            score = number(movie, "score", "rating")
            popularity = number(movie, "hotScore", "popularity", "heat")
            box_office = number(movie, "boxOffice", "boxOfficeAmount")
            status = str(movie.get("status") or "")
            is_hot = 1.0 if "热" in status or "上映" in status else 0.0

            if normalized_criteria in genre_ranks:
                genre_rank = max(
                    (
                        value
                        for name, value in genre_ranks[normalized_criteria].items()
                        if name in genre
                    ),
                    default=0,
                )
                return (genre_rank, score)
            if normalized_criteria == "high_rating":
                return (score,)
            if normalized_criteria == "box_office":
                return (box_office, popularity, score)
            if normalized_criteria == "hot":
                return (is_hot, popularity, box_office, score)
            return (score,)

        return sorted(movies, key=rank, reverse=True)

    @staticmethod
    def _apply_movie_limit(
        movies: list[dict[str, Any]],
        limit: Any,
    ) -> list[dict[str, Any]]:
        try:
            size = int(limit)
        except (TypeError, ValueError):
            return movies
        if size <= 0:
            return movies
        return movies[: min(size, 10)]

    def _movie_search_message(
        self,
        movies: list[dict[str, Any]],
        criteria: Any,
        cinema_name: str | None = None,
        has_showtimes: bool = False,
        genre: Any = None,
        keyword: Any = None,
        date: Any = None,
    ) -> str:
        count = len(movies)
        genre_label = str(genre or "").strip()
        keyword_label = str(keyword or "").strip()
        date_label = self._display_date(date) if date else ""

        if count == 0:
            if genre_label:
                if date_label:
                    return f"{date_label}没有{genre_label}类型的排片。要不要换个类型或日期看看？"
                return f"{genre_label}类型的影片暂时没有排片。要不要看看现在有哪些电影正在上映？"
            if keyword_label:
                return f"没有找到与「{keyword_label}」相关的影片，换个关键词试试？"
            if cinema_name:
                return f"{cinema_name}暂时没有正在上映的影片，看看其他影院？"
            return "当前没有正在上映的影片。过几天再来看看吧。"

        head = self._movie_search_head(count, genre_label, keyword_label, criteria, cinema_name, has_showtimes, date_label)
        detail = self._movie_showtime_lines(movies)
        if detail:
            return f"{head}\n\n{detail}"
        return head

    def _fallback_movie_search_message(
        self,
        fallback_reason: str | None,
        arguments: dict[str, Any],
        movies: list[dict[str, Any]],
    ) -> str | None:
        if not fallback_reason or not movies:
            return None

        constraints = self._showtime_constraints(arguments)
        if fallback_reason == "today_high_rating":
            message = (
                f"按{constraints}暂时没有符合条件的场次，"
                "先为你推荐今天仍可观看的高分电影。"
            )
        elif fallback_reason == "tomorrow_same_type":
            message = (
                f"按{constraints}今天暂时没有符合条件的场次，"
                "为你找到明天相同类型的可选场次。"
            )
        else:
            return None

        detail = self._movie_showtime_lines(movies)
        return f"{message}\n\n{detail}" if detail else message

    @staticmethod
    def _movie_search_head(
        count: int,
        genre_label: str,
        keyword_label: str,
        criteria: Any,
        cinema_name: str | None,
        has_showtimes: bool,
        date_label: str = "",
    ) -> str:
        date_prefix = f"{date_label}" if date_label else ""
        if genre_label:
            return f"为你找到了 {count} 部{date_prefix}{genre_label}片"
        if keyword_label:
            return f"为你找到了 {count} 部与「{keyword_label}」相关的{date_prefix}影片"
        labels = {
            "couple": "适合情侣一起观看的",
            "family": "适合亲子或家庭的",
            "high_rating": "高分口碑",
            "box_office": "当前可购的",
            "hot": "当前热映的",
            "general": "值得关注的",
        }
        label = labels.get(str(criteria or "").strip())
        if label:
            return f"为你找到了 {count} 部{label}{date_prefix}影片"
        if cinema_name:
            action = "正在排片" if has_showtimes else "正在上映"
            return f"{cinema_name}有 {count} 部电影{date_prefix}{action}"
        if date_label:
            return f"{date_prefix}有 {count} 部电影正在上映"
        return f"当前有 {count} 部电影正在上映"

    @staticmethod
    def _movie_showtime_lines(movies: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for movie in movies[:5]:  # 最多列出 5 部
            name = movie.get("movieName", "")
            showtimes = movie.get("upcomingShowtimes") or []
            if not showtimes:
                continue
            parts: list[str] = []
            for st in showtimes[:2]:  # 每部最多 2 场
                cinema = st.get("cinemaName") or ""
                hall = st.get("hallName") or ""
                start = st.get("startAt") or ""
                end = st.get("endAt") or ""
                price = st.get("price") or ""
                price_text = f" {price}元" if price else ""
                start_text = str(start)[:16].replace("T", " ") if start else ""
                end_text = str(end)[:16].replace("T", " ") if end else ""
                time_text = f"{start_text} - {end_text}" if start_text and end_text else start_text
                where = f"{cinema} {hall}".strip()
                parts.append(
                    " · ".join(
                        part for part in [where, time_text, price_text.strip()] if part
                    )
                )
            if parts:
                lines.append(f"• 《{name}》")
                lines.extend(f"  {part}" for part in parts)
        return "\n".join(lines)

    def _pick_movie(
        self,
        movies: list[dict[str, Any]],
        movie_name: str,
    ) -> dict[str, Any] | None:
        normalized = movie_name.replace(" ", "").casefold()
        for movie in movies:
            candidate = str(movie.get("movieName") or "").replace(" ", "").casefold()
            if candidate == normalized:
                return movie
        for movie in movies:
            candidate = str(movie.get("movieName") or "").replace(" ", "").casefold()
            if normalized in candidate or candidate in normalized:
                return movie
        return movies[0] if len(movies) == 1 else None

    def _spring_date(self, date_value: Any) -> str | None:
        if date_value in [None, ""]:
            return None
        text = str(date_value)
        now = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        if text in {"today", "今天", "今晚"}:
            return now.isoformat()
        if text in {"tomorrow", "明天", "明晚"}:
            return (now + timedelta(days=1)).isoformat()
        if text in {"after_tomorrow", "后天"}:
            return (now + timedelta(days=2)).isoformat()
        return text

    def _format_showtime_groups(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        cinema = data.get("cinema") or {}
        showtimes: list[dict[str, Any]] = []
        for group in data.get("movies") or []:
            for item in group.get("showtimes") or []:
                start_at = item.get("startAt")
                showtimes.append(
                    {
                        "showtimeId": item.get("id"),
                        "movieId": group.get("id"),
                        "movieName": group.get("name"),
                        "cinemaId": cinema.get("id"),
                        "cinemaName": cinema.get("name"),
                        "hallName": item.get("hallName"),
                        "hallType": item.get("hallType"),
                        "language": item.get("language"),
                        "date": str(start_at or "")[:10],
                        "time": str(start_at or "")[11:16],
                        "startAt": start_at,
                        "endAt": item.get("endAt"),
                        "price": item.get("basePrice"),
                        "remainingSeats": item.get("remainingSeats"),
                    }
                )
        return showtimes

    def _filter_showtimes(
        self,
        showtimes: list[dict[str, Any]],
        date_value: str | None,
        time_range: Any,
        ticket_count: Any,
        price_preference: Any = None,
        time_preference: Any = None,
    ) -> list[dict[str, Any]]:
        requested_time = str(time_range or "")
        min_seats = int(ticket_count or 0)
        filtered = []
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        for showtime in showtimes:
            start_at = self._parse_showtime_datetime(showtime.get("startAt"))
            if start_at is not None and start_at <= now:
                continue
            showtime_date = str(showtime.get("date") or showtime.get("startAt") or "")[:10]
            if date_value and showtime_date != date_value:
                continue
            remaining = showtime.get("remainingSeats")
            if min_seats and remaining is not None and int(remaining) < min_seats:
                continue
            if requested_time and not self._time_matches(showtime.get("time"), requested_time):
                continue
            filtered.append(showtime)
        if str(price_preference or "") == "lower":
            prices = [
                float(showtime["price"])
                for showtime in filtered
                if showtime.get("price") not in [None, ""]
            ]
            if prices:
                max_price = min(prices) + 10
                filtered = [
                    showtime
                    for showtime in filtered
                    if showtime.get("price") in [None, ""]
                    or float(showtime["price"]) <= max_price
                ]
            filtered.sort(key=lambda item: float(item.get("price") or 0))
        elif str(time_preference or "") == "later":
            filtered.sort(key=lambda item: str(item.get("time") or ""))
            if requested_time:
                later = [
                    showtime
                    for showtime in filtered
                    if str(showtime.get("time") or "") > requested_time
                ]
                if later:
                    filtered = later
        elif str(time_preference or "") == "earlier":
            filtered.sort(key=lambda item: str(item.get("time") or ""), reverse=True)
            if requested_time:
                earlier = [
                    showtime
                    for showtime in filtered
                    if str(showtime.get("time") or "") < requested_time
                ]
                if earlier:
                    filtered = earlier
        return filtered

    def _parse_showtime_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return parsed.astimezone(ZoneInfo("Asia/Shanghai"))

    def _time_matches(self, showtime_time: Any, requested_time: str) -> bool:
        if not requested_time:
            return True
        value = str(showtime_time or "")
        if requested_time in {"evening", "晚上"}:
            return value >= "18:00"
        if requested_time in {"afternoon", "下午"}:
            return "12:00" <= value < "19:00"
        if requested_time in {"morning", "上午"}:
            return value < "12:00"
        # 用户说“16点”时匹配该小时内的场次，避免把晚上的场次一并返回。
        match = re.fullmatch(r"(\d{2}):(\d{2})", requested_time)
        if match:
            hour, minute = match.groups()
            if minute == "00":
                return value.startswith(f"{hour}:")
            return value == requested_time
        return value >= requested_time

    def _should_auto_navigate(
        self,
        arguments: dict[str, Any],
        showtimes: list[dict[str, Any]],
    ) -> bool:
        return (
            len(showtimes) == 1
            and bool(arguments.get("movieName") or arguments.get("movieId"))
            and bool(arguments.get("date") or arguments.get("timeRange"))
            and bool(arguments.get("ticketCount"))
        )

    def _showtime_search_message(
        self,
        arguments: dict[str, Any],
        showtimes: list[dict[str, Any]],
        response_data: dict[str, Any],
    ) -> str:
        if response_data.get("navigation"):
            return "已匹配到唯一场次，准备进入选座。"
        if showtimes:
            return f"为你找到了 {len(showtimes)} 个符合条件的场次。"

        constraints = self._showtime_constraints(arguments)
        if constraints:
            return f"按{constraints}没有找到合适的场次，可以换影院、换时间或换类型看看。"
        return "没有找到合适的场次，可以换影院、换时间或换类型看看。"

    def _showtime_constraints(self, arguments: dict[str, Any]) -> str:
        parts = []
        if arguments.get("cinemaName"):
            parts.append(str(arguments["cinemaName"]))
        if arguments.get("movieName"):
            parts.append(f"《{arguments['movieName']}》")
        elif arguments.get("genre"):
            parts.append(f"{arguments['genre']}类型")
        if arguments.get("date"):
            parts.append(self._display_date(arguments["date"]))
        if arguments.get("timeRange"):
            parts.append(self._display_time_range(arguments["timeRange"]))
        if arguments.get("ticketCount"):
            parts.append(f"{arguments['ticketCount']}张票")
        if arguments.get("hallType"):
            parts.append(str(arguments["hallType"]))
        return "、".join(parts)

    def _display_date(self, value: Any) -> str:
        text = str(value)
        return {
            "today": "今天",
            "今晚": "今天",
            "tomorrow": "明天",
            "明晚": "明天",
            "after_tomorrow": "后天",
            "weekend": "周末",
        }.get(text, text)

    def _display_time_range(self, value: Any) -> str:
        text = str(value)
        return {
            "evening": "晚上",
            "afternoon": "下午",
            "morning": "上午",
        }.get(text, f"{text}后")

    def _showtime_subtitle(self, showtime: dict[str, Any]) -> str:
        start_at = str(showtime.get("startAt") or "")
        end_at = str(showtime.get("endAt") or "")
        start_text = start_at[:16].replace("T", " ") if start_at else ""
        end_text = end_at[:16].replace("T", " ") if end_at else ""
        time_text = f"{start_text} - {end_text}" if start_text and end_text else (
            start_text or end_text or showtime.get("time")
        )
        parts = [
            showtime.get("movieName"),
            showtime.get("cinemaName"),
            showtime.get("hallName"),
            time_text,
        ]
        return " ".join(str(part) for part in parts if part)

    def _format_cinema(self, item: dict[str, Any]) -> dict[str, Any]:
        cinema_id = item.get("id") or item.get("cinemaId")
        cinema_name = item.get("name") or item.get("cinemaName")
        longitude = item.get("longitude")
        latitude = item.get("latitude")
        return {
            "cinemaId": cinema_id,
            "cinemaName": cinema_name,
            "address": item.get("address"),
            "district": item.get("district"),
            "services": (
                item.get("services")
                or item.get("serviceTags")
                or item.get("service")
            ),
            "location": (
                f"{longitude},{latitude}"
                if longitude is not None and latitude is not None
                else None
            ),
            "distance": item.get("distance"),
            "hallTypes": item.get("hallTypes"),
            "minPrice": item.get("minPrice"),
        }


class AMapMCP:
    def __init__(
        self,
        key_env: str = "AMAP_WEB_SERVICE_KEY",
        base_url: str = "https://restapi.amap.com",
        timeout_seconds: int = 15,
    ) -> None:
        self.key_env = key_env
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        handlers = {
            "search_nearby_cinemas": self.search_nearby_cinemas,
            "geocode": self.geocode,
            "regeocode": self.regeocode,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(
                tool_name=f"amap.{tool_name}",
                success=False,
                message=f"Unknown AMap MCP tool: {tool_name}",
            )

        try:
            return handler(arguments)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            message = self._failure_message(exc)
            return ToolResult(
                tool_name=f"amap.{tool_name}",
                success=False,
                data={"error": str(exc)},
                message=message,
            )

    def _failure_message(self, error: Exception) -> str:
        """Keep upstream failure details useful without exposing the API key."""
        detail = str(error).strip()
        if detail.startswith("Missing environment variable:"):
            return "高德地图未配置 API Key，请检查 AMAP_WEB_SERVICE_KEY。"
        if detail.startswith("AMap API error:"):
            return f"高德地图服务返回错误：{detail.removeprefix('AMap API error:').strip()}"
        if isinstance(error, httpx.HTTPStatusError):
            return f"高德地图 HTTP 请求失败：{error.response.status_code}。"
        if isinstance(error, httpx.RequestError):
            return "高德地图暂时无法连接，请检查 Agent 服务的网络权限或代理配置。"
        return "高德地图请求失败，请稍后重试。"

    def search_nearby_cinemas(self, arguments: dict[str, Any]) -> ToolResult:
        location = self._normalize_location(arguments.get("location"))
        city = arguments.get("city")
        keyword = arguments.get("keywords") or arguments.get("keyword") or "电影院"

        if not location and not city:
            return ToolResult(
                tool_name="amap.search_nearby_cinemas",
                success=False,
                data={"error": "LOCATION_REQUIRED"},
                message="未获取到当前位置，无法准确查询附近影院。请允许浏览器定位后重试。",
            )

        if location:
            payload = self._get(
                "/v3/place/around",
                {
                    "keywords": keyword,
                    "location": location,
                    "radius": arguments.get("radius", 5000),
                    "offset": arguments.get("offset", 10),
                    "page": arguments.get("page", 1),
                    "extensions": "base",
                },
            )
        else:
            payload = self._get(
                "/v3/place/text",
                {
                    "keywords": keyword,
                    "city": city,
                    "citylimit": "true" if city else "false",
                    "offset": arguments.get("offset", 10),
                    "page": arguments.get("page", 1),
                    "extensions": "base",
                },
            )

        cinemas = [self._format_cinema(poi) for poi in payload.get("pois", [])]
        return ToolResult(
            tool_name="amap.search_nearby_cinemas",
            data={"cinemas": cinemas, "raw": payload},
            message="AMap cinema search completed." if cinemas else "AMap returned no matching cinemas.",
        )

    def geocode(self, arguments: dict[str, Any]) -> ToolResult:
        payload = self._get(
            "/v3/geocode/geo",
            {"address": arguments.get("address"), "city": arguments.get("city")},
        )
        return ToolResult(
            tool_name="amap.geocode",
            data={"geocodes": payload.get("geocodes", []), "raw": payload},
            message="AMap geocode completed.",
        )

    def regeocode(self, arguments: dict[str, Any]) -> ToolResult:
        payload = self._get(
            "/v3/geocode/regeo",
            {
                "location": self._normalize_location(arguments.get("location")),
                "radius": arguments.get("radius", 1000),
                "extensions": arguments.get("extensions", "base"),
            },
        )
        return ToolResult(
            tool_name="amap.regeocode",
            data={"regeocode": payload.get("regeocode", {}), "raw": payload},
            message="AMap regeocode completed.",
        )

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        key = os.getenv(self.key_env)
        if not key:
            raise RuntimeError(f"Missing environment variable: {self.key_env}")

        response = httpx.get(
            f"{self.base_url}{path}",
            params={
                "key": key,
                "output": "json",
                **{k: v for k, v in params.items() if v not in [None, ""]},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "1":
            info = payload.get("info") or "UNKNOWN_ERROR"
            infocode = payload.get("infocode") or ""
            raise RuntimeError(f"AMap API error: {info} {infocode}".strip())
        return payload

    def _format_cinema(self, poi: dict[str, Any]) -> dict[str, Any]:
        return {
            "cinemaId": poi.get("id"),
            "cinemaName": poi.get("name"),
            "address": poi.get("address"),
            "location": poi.get("location"),
            "distance": poi.get("distance"),
            "tel": poi.get("tel"),
            "type": poi.get("type"),
        }

    def _normalize_location(self, location: Any) -> str | None:
        if not location:
            return None
        if isinstance(location, dict):
            lng = location.get("longitude") or location.get("lng")
            lat = location.get("latitude") or location.get("lat")
            if lng and lat:
                return f"{lng},{lat}"
            return None

        value = str(location).strip()
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            return value

        first, second = parts
        try:
            first_float = float(first)
            second_float = float(second)
        except ValueError:
            return value

        if abs(first_float) <= 90 and abs(second_float) > 90:
            return f"{second},{first}"
        return value


class MCPDispatcher:
    def __init__(self) -> None:
        mcp_settings = agent_config.get("mcp", {})
        self.clients: dict[str, MCPClient] = {
            "spring_boot": self._spring_boot_client(),
            "amap": self._amap_client(mcp_settings),
        }

    def call(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        client = self.clients.get(server_name)
        if not client:
            return ToolResult(
                tool_name=f"{server_name}.{tool_name}",
                success=False,
                message=f"MCP server is not registered: {server_name}",
            )
        return client.call_tool(tool_name, arguments)

    def _amap_client(self, settings: dict[str, Any]) -> AMapMCP:
        server_settings = settings.get("amap", {})
        return AMapMCP(
            key_env=server_settings.get("key_env", "AMAP_WEB_SERVICE_KEY"),
            base_url=server_settings.get("base_url") or "https://restapi.amap.com",
            timeout_seconds=server_settings.get("timeout_seconds", 15),
        )

    def _spring_boot_client(self) -> SpringBootMovieTicketMCP:
        server_settings = agent_config.get("spring_boot", {})
        return SpringBootMovieTicketMCP(
            base_url=server_settings.get("base_url") or "http://localhost:8080",
            timeout_seconds=server_settings.get("timeout_seconds", 15),
        )


mcp_dispatcher = MCPDispatcher()
