import os
import json
import re
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.schemas.agent import ToolResult
from app.utils.config_handler import agent_config
from app.utils.tool_path import get_project_abs_path


class MCPClient(Protocol):
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        pass


class HttpMCPClient:
    def __init__(
        self,
        server_name: str,
        base_url: str | None = None,
        token_env: str | None = None,
        timeout_seconds: int = 15,
    ) -> None:
        self.server_name = server_name
        self.base_url = base_url
        self.token_env = token_env
        self.timeout_seconds = timeout_seconds

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self.base_url:
            return ToolResult(
                tool_name=f"{self.server_name}.{tool_name}",
                success=False,
                message=f"{self.server_name} MCP base_url is not configured.",
                data={"arguments": arguments, "configured": False},
            )

        headers = {}
        if self.token_env:
            token = os.getenv(self.token_env)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/tools/{tool_name}",
                json={"arguments": arguments},
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            return ToolResult(
                tool_name=f"{self.server_name}.{tool_name}",
                success=False,
                data={"error": str(exc)},
                message=f"{self.server_name} MCP request failed.",
            )

        return ToolResult(
            tool_name=f"{self.server_name}.{tool_name}",
            success=bool(payload.get("success", True)),
            data=payload.get("data", payload),
            message=payload.get("message", ""),
        )


class LocalMovieTicketMCP:
    """Stateful local movie-ticket core used before the Java backend is deployed."""

    GENRE_ALIASES = {
        "comedy": "喜剧",
        "drama": "剧情",
        "action": "动作",
        "sci-fi": "科幻",
        "science fiction": "科幻",
        "animation": "动画",
        "horror": "恐怖",
        "romance": "爱情",
    }

    def __init__(self) -> None:
        self._lock = RLock()
        self.movies = [
            {
                "movieId": "m_1001",
                "movieName": "流浪地球3",
                "genre": "科幻",
                "score": 9.1,
                "durationMinutes": 150,
                "status": "NOW_SHOWING",
            },
            {
                "movieId": "m_1002",
                "movieName": "喜剧之王",
                "genre": "喜剧",
                "score": 8.8,
                "durationMinutes": 120,
                "status": "NOW_SHOWING",
            },
            {
                "movieId": "m_1003",
                "movieName": "星际探险",
                "genre": "科幻",
                "score": 8.6,
                "durationMinutes": 135,
                "status": "NOW_SHOWING",
            },
        ]
        self.showtimes = [
            {
                "showtimeId": "st_2001",
                "movieId": "m_1001",
                "movieName": "流浪地球3",
                "cinemaId": "c_1001",
                "cinemaName": "Cinema One",
                "hallName": "1号IMAX厅",
                "hallType": "IMAX",
                "date": "today",
                "time": "19:30",
                "price": 42,
            },
            {
                "showtimeId": "st_2002",
                "movieId": "m_1001",
                "movieName": "流浪地球3",
                "cinemaId": "c_1002",
                "cinemaName": "Cinema Two",
                "hallName": "2号激光厅",
                "hallType": "普通",
                "date": "today",
                "time": "21:10",
                "price": 39,
            },
            {
                "showtimeId": "st_2003",
                "movieId": "m_1002",
                "movieName": "喜剧之王",
                "cinemaId": "c_1001",
                "cinemaName": "Cinema One",
                "hallName": "3号厅",
                "hallType": "普通",
                "date": "today",
                "time": "20:00",
                "price": 35,
            },
            {
                "showtimeId": "st_2004",
                "movieId": "m_1003",
                "movieName": "星际探险",
                "cinemaId": "c_1002",
                "cinemaName": "Cinema Two",
                "hallName": "1号IMAX厅",
                "hallType": "IMAX",
                "date": "today",
                "time": "18:40",
                "price": 45,
            },
        ]
        self.seats: dict[str, list[dict[str, Any]]] = {
            showtime["showtimeId"]: self._build_seats()
            for showtime in self.showtimes
        }
        self.locks: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        handlers = {
            "search_movies": self.search_movies,
            "search_showtimes": self.search_showtimes,
            "get_seats": self.get_seats,
            "lock_seats": self.lock_seats,
            "create_order": self.create_order,
            "pay_order": self.pay_order,
            "issue_ticket": self.issue_ticket,
            "get_order": self.get_order,
            "list_orders": self.list_orders,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(
                tool_name=f"movie_ticket.{tool_name}",
                success=False,
                message=f"Unknown movie_ticket MCP tool: {tool_name}",
            )
        with self._lock:
            self._release_expired_locks()
            try:
                return handler(arguments)
            except (KeyError, TypeError, ValueError) as exc:
                return ToolResult(
                    tool_name=f"movie_ticket.{tool_name}",
                    success=False,
                    data={"error": str(exc)},
                    message=str(exc),
                )

    def search_movies(self, arguments: dict[str, Any]) -> ToolResult:
        keyword = str(
            arguments.get("movieName")
            or arguments.get("keyword")
            or ""
        ).strip().casefold()
        genre = self._normalize_genre(arguments.get("genre"))
        movies = [
            deepcopy(movie)
            for movie in self.movies
            if (
                not keyword
                or keyword in movie["movieName"].casefold()
            )
            and (
                not genre
                or genre in movie["genre"].casefold()
            )
        ]
        return ToolResult(
            tool_name="movie_ticket.search_movies",
            data={"movies": movies},
            message=f"已找到 {len(movies)} 部符合条件的电影。",
        )

    def search_showtimes(self, arguments: dict[str, Any]) -> ToolResult:
        movie_name = str(arguments.get("movieName") or "").strip().casefold()
        movie_id = arguments.get("movieId")
        cinema_id = arguments.get("cinemaId")
        genre = self._normalize_genre(arguments.get("genre"))
        requested_date = arguments.get("date")
        requested_time = arguments.get("timeRange")
        price_preference = arguments.get("pricePreference")

        showtimes = []
        for showtime in self.showtimes:
            movie = self._movie_by_id(showtime["movieId"])
            if movie_id and showtime["movieId"] != movie_id:
                continue
            if movie_name and movie_name not in showtime["movieName"].casefold():
                continue
            if genre and genre not in movie["genre"].casefold():
                continue
            if cinema_id and showtime["cinemaId"] != cinema_id:
                continue
            if requested_date:
                showtime = {**showtime, "date": requested_date}
            if requested_time and not self._time_matches(showtime["time"], requested_time):
                continue
            if price_preference == "lower" and showtime["price"] > 40:
                continue
            item = deepcopy(showtime)
            item["remainingSeats"] = self._available_count(item["showtimeId"])
            showtimes.append(item)

        showtimes.sort(
            key=lambda item: (
                item["price"] if price_preference == "lower" else 0,
                item["time"],
            )
        )
        return ToolResult(
            tool_name="movie_ticket.search_showtimes",
            data={"showtimes": showtimes},
            message=f"已找到 {len(showtimes)} 个符合条件的场次。",
        )

    def get_seats(self, arguments: dict[str, Any]) -> ToolResult:
        showtime_id = arguments.get("showtimeId")
        self._showtime_by_id(showtime_id)
        return ToolResult(
            tool_name="movie_ticket.get_seats",
            data={
                "showtimeId": showtime_id,
                "seats": deepcopy(self.seats[showtime_id]),
            },
            message="座位图已加载。",
        )

    def lock_seats(self, arguments: dict[str, Any]) -> ToolResult:
        showtime_id = arguments.get("showtimeId")
        seat_ids = arguments.get("seatIds") or []
        user_id = arguments.get("userId") or "anonymous"
        self._showtime_by_id(showtime_id)
        if not seat_ids:
            raise ValueError("请选择至少一个座位。")
        seat_map = {seat["seatId"]: seat for seat in self.seats[showtime_id]}
        missing = [seat_id for seat_id in seat_ids if seat_id not in seat_map]
        if missing:
            raise ValueError(f"座位不存在：{', '.join(missing)}")
        unavailable = [
            seat_id
            for seat_id in seat_ids
            if seat_map[seat_id]["status"] != "available"
        ]
        if unavailable:
            return ToolResult(
                tool_name="movie_ticket.lock_seats",
                success=False,
                data={
                    "conflictSeatIds": unavailable,
                    "showtimeId": showtime_id,
                    "seats": deepcopy(self.seats[showtime_id]),
                },
                message=f"座位已被占用：{', '.join(unavailable)}",
            )

        lock_id = f"lock_{uuid.uuid4().hex[:10]}"
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
        for seat_id in seat_ids:
            seat_map[seat_id].update(
                {
                    "status": "locked",
                    "lockId": lock_id,
                    "lockOwner": user_id,
                    "lockExpiresAt": expires_at.isoformat(),
                }
            )
        self.locks[lock_id] = {
            "lockId": lock_id,
            "showtimeId": showtime_id,
            "seatIds": list(seat_ids),
            "userId": user_id,
            "expiresAt": expires_at.isoformat(),
            "status": "locked",
        }
        showtime = self._showtime_by_id(showtime_id)
        total_amount = showtime["price"] * len(seat_ids)
        return ToolResult(
            tool_name="movie_ticket.lock_seats",
            data={
                "lockId": lock_id,
                "showtimeId": showtime_id,
                "seatIds": list(seat_ids),
                "amount": total_amount,
                "expiresAt": expires_at.isoformat(),
            },
            message=f"已锁定 {len(seat_ids)} 个座位，有效期 15 分钟。",
        )

    def create_order(self, arguments: dict[str, Any]) -> ToolResult:
        lock_id = arguments.get("lockId")
        lock = self._active_lock(lock_id)
        order_id = f"ord_{uuid.uuid4().hex[:12]}"
        showtime = self._showtime_by_id(lock["showtimeId"])
        amount = showtime["price"] * len(lock["seatIds"])
        order = {
            "orderId": order_id,
            "lockId": lock_id,
            "showtimeId": lock["showtimeId"],
            "seatIds": list(lock["seatIds"]),
            "movieName": showtime["movieName"],
            "cinemaName": showtime["cinemaName"],
            "hallName": showtime["hallName"],
            "date": showtime["date"],
            "time": showtime["time"],
            "amount": amount,
            "status": "PAYMENT_PENDING",
            "expiresAt": lock["expiresAt"],
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        self.orders[order_id] = order
        return ToolResult(
            tool_name="movie_ticket.create_order",
            data=deepcopy(order),
            message=f"订单已创建，应付 {amount} 元。",
        )

    def pay_order(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = arguments.get("orderId")
        order = self.orders.get(order_id)
        if not order:
            raise ValueError("订单不存在。")
        if order["status"] == "TICKETED":
            return ToolResult(
                tool_name="movie_ticket.pay_order",
                data=deepcopy(order),
                message="订单已经出票，无需重复支付。",
            )
        if order["status"] != "PAYMENT_PENDING":
            raise ValueError(f"订单当前状态不可支付：{order['status']}")
        self._active_lock(order["lockId"])
        order["status"] = "PAID"
        order["paidAt"] = datetime.now(timezone.utc).isoformat()
        self._mark_order_seats(order, "sold")
        return ToolResult(
            tool_name="movie_ticket.pay_order",
            data=deepcopy(order),
            message="支付成功。",
        )

    def issue_ticket(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = arguments.get("orderId")
        order = self.orders.get(order_id)
        if not order:
            raise ValueError("订单不存在。")
        if order["status"] == "TICKETED":
            return ToolResult(
                tool_name="movie_ticket.issue_ticket",
                data=deepcopy(order),
                message="订单已经出票。",
            )
        if order["status"] != "PAID":
            raise ValueError("订单尚未支付，不能出票。")
        order["status"] = "TICKETED"
        order["ticketStatus"] = "issued"
        order["ticketCodes"] = [
            f"TKT-{uuid.uuid4().hex[:10].upper()}"
            for _ in order["seatIds"]
        ]
        order["issuedAt"] = datetime.now(timezone.utc).isoformat()
        return ToolResult(
            tool_name="movie_ticket.issue_ticket",
            data=deepcopy(order),
            message="出票成功。",
        )

    def get_order(self, arguments: dict[str, Any]) -> ToolResult:
        order_id = arguments.get("orderId")
        order = self.orders.get(order_id)
        if not order:
            raise ValueError("订单不存在。")
        return ToolResult(
            tool_name="movie_ticket.get_order",
            data=deepcopy(order),
            message="订单已加载。",
        )

    def list_orders(self, arguments: dict[str, Any]) -> ToolResult:
        status = str(arguments.get("status") or "").strip().upper()
        orders = []
        for order in self.orders.values():
            if status and str(order.get("status", "")).upper() != status:
                continue
            orders.append(deepcopy(order))
        orders.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
        return ToolResult(
            tool_name="movie_ticket.list_orders",
            data={"orders": orders},
            message=f"已加载 {len(orders)} 个订单。",
        )

    def _build_seats(self) -> list[dict[str, Any]]:
        seats = []
        for row in ["A", "B", "C", "D", "E", "F"]:
            for number in range(1, 9):
                seats.append(
                    {
                        "seatId": f"{row}{number}",
                        "row": row,
                        "number": number,
                        "status": "available",
                        "zone": "middle" if row in {"C", "D"} else "standard",
                    }
                )
        return seats

    def _movie_by_id(self, movie_id: str) -> dict[str, Any]:
        for movie in self.movies:
            if movie["movieId"] == movie_id:
                return movie
        raise ValueError(f"电影不存在：{movie_id}")

    def _normalize_genre(self, genre: Any) -> str:
        normalized = str(genre or "").strip().casefold()
        return self.GENRE_ALIASES.get(normalized, normalized)

    def _showtime_by_id(self, showtime_id: str) -> dict[str, Any]:
        for showtime in self.showtimes:
            if showtime["showtimeId"] == showtime_id:
                return showtime
        raise ValueError(f"场次不存在：{showtime_id}")

    def _available_count(self, showtime_id: str) -> int:
        return sum(
            seat["status"] == "available"
            for seat in self.seats[showtime_id]
        )

    def _time_matches(self, showtime_time: str, requested_time: str) -> bool:
        if not requested_time:
            return True
        if requested_time in {"evening", "晚上"}:
            return showtime_time >= "18:00"
        if requested_time in {"afternoon", "下午"}:
            return "12:00" <= showtime_time < "18:00"
        if requested_time in {"morning", "上午"}:
            return showtime_time < "12:00"
        return showtime_time >= str(requested_time)

    def _active_lock(self, lock_id: str | None) -> dict[str, Any]:
        if not lock_id or lock_id not in self.locks:
            raise ValueError("座位锁定不存在或已失效。")
        lock = self.locks[lock_id]
        if lock["status"] != "locked":
            raise ValueError("座位锁定不可用。")
        if datetime.fromisoformat(lock["expiresAt"]) <= datetime.now(timezone.utc):
            self._release_lock(lock)
            raise ValueError("座位锁定已过期，请重新选座。")
        return lock

    def _release_expired_locks(self) -> None:
        now = datetime.now(timezone.utc)
        for lock in list(self.locks.values()):
            if (
                lock["status"] == "locked"
                and datetime.fromisoformat(lock["expiresAt"]) <= now
            ):
                self._release_lock(lock)

    def _release_lock(self, lock: dict[str, Any]) -> None:
        for seat in self.seats[lock["showtimeId"]]:
            if seat.get("lockId") == lock["lockId"]:
                seat.update(
                    {
                        "status": "available",
                        "lockId": None,
                        "lockOwner": None,
                        "lockExpiresAt": None,
                    }
                )
        lock["status"] = "expired"

    def _mark_order_seats(self, order: dict[str, Any], status: str) -> None:
        for seat in self.seats[order["showtimeId"]]:
            if seat["seatId"] in order["seatIds"]:
                seat.update(
                    {
                        "status": status,
                        "lockId": None,
                        "lockOwner": None,
                        "lockExpiresAt": None,
                    }
                )
        lock = self.locks.get(order["lockId"])
        if lock:
            lock["status"] = "consumed"

class LocalSnackMCP:
    """Temporary snack adapter behind the unified MCP dispatcher."""

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name != "recommend_snacks":
            return ToolResult(
                tool_name=f"snack.{tool_name}",
                success=False,
                message=f"Unknown snack MCP tool: {tool_name}",
            )
        return ToolResult(
            tool_name="snack.recommend_snacks",
            data={
                "snacks": [
                    {"snackId": "sn_1", "name": "Popcorn combo", "price": 35},
                    {"snackId": "sn_2", "name": "Two drinks", "price": 18},
                ]
            },
            message="Snacks loaded from temporary local adapter.",
        )


class LocalCouponMCP:
    """Temporary coupon adapter behind the unified MCP dispatcher."""

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name != "recommend_coupons":
            return ToolResult(
                tool_name=f"coupon.{tool_name}",
                success=False,
                message=f"Unknown coupon MCP tool: {tool_name}",
            )
        return ToolResult(
            tool_name="coupon.recommend_coupons",
            data={
                "coupons": [
                    {"couponId": "cp_1", "name": "10 off 80", "discount": 10},
                    {"couponId": "cp_2", "name": "Member 10% off", "discount": "10%"},
                ]
            },
            message="Coupons loaded from temporary local adapter.",
        )


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
            return ToolResult(
                tool_name=f"spring_boot.{tool_name}",
                success=False,
                data={"error": str(exc)},
                message="票务数据库查询失败，请确认 Spring Boot 服务和登录状态正常。",
            )

    def search_movies(self, arguments: dict[str, Any]) -> ToolResult:
        keyword = arguments.get("movieName") or arguments.get("keyword")
        data = self._get_business(
            "/api/user/movies",
            arguments,
            {
                "page": arguments.get("page", 1),
                "size": arguments.get("size", 10),
                "keyword": keyword,
                "genre": arguments.get("genre"),
                "status": arguments.get("status"),
            },
        )
        records = data.get("records") or []
        movies = [self._format_movie(item) for item in records]
        return ToolResult(
            tool_name="spring_boot.search_movies",
            data={"movies": movies, "source": "spring_boot_database"},
            message=f"已在票务数据库中找到 {len(movies)} 部影片。",
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
                    message=f"票务数据库中没有找到《{movie_name}》。",
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
                    message=f"票务数据库中没有找到{arguments.get('genre')}类型影片。",
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
            row_no = row.get("rowNo")
            for seat in row.get("seats") or []:
                seats.append(
                    {
                        "seatId": seat.get("id"),
                        "row": row_no,
                        "number": seat.get("seatNo"),
                        "status": str(seat.get("status") or "AVAILABLE").lower(),
                        "price": seat.get("price") or data.get("basePrice"),
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

    def _format_movie(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "movieId": item.get("id") or item.get("movieId"),
            "movieName": item.get("name") or item.get("movieName"),
            "genre": item.get("genre"),
            "score": item.get("rating") or item.get("score"),
            "durationMinutes": item.get("duration"),
            "status": item.get("statusDesc") or item.get("status"),
        }

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
        time_range: Any,
        ticket_count: Any,
        price_preference: Any = None,
        time_preference: Any = None,
    ) -> list[dict[str, Any]]:
        requested_time = str(time_range or "")
        min_seats = int(ticket_count or 0)
        filtered = []
        for showtime in showtimes:
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

    def _time_matches(self, showtime_time: Any, requested_time: str) -> bool:
        if not requested_time:
            return True
        value = str(showtime_time or "")
        if requested_time in {"evening", "晚上"}:
            return value >= "18:00"
        if requested_time in {"afternoon", "下午"}:
            return "12:00" <= value < "18:00"
        if requested_time in {"morning", "上午"}:
            return value < "12:00"
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
            return f"已在票务数据库中找到 {len(showtimes)} 个符合条件的场次。"

        constraints = self._showtime_constraints(arguments)
        if constraints:
            return f"按{constraints}查询，票务数据库中没有符合条件的场次。可以换影院、换时间或换类型。"
        return "票务数据库中没有符合条件的场次。可以换影院、换时间或换类型。"

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
        parts = [
            showtime.get("movieName"),
            showtime.get("cinemaName"),
            showtime.get("date"),
            showtime.get("time"),
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


class CalendarMCP:
    def __init__(self, http_client: HttpMCPClient) -> None:
        self.http_client = http_client

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        return self.http_client.call_tool(tool_name, arguments)


class LocalCalendarMCP:
    """Create importable ICS events without a third-party calendar API."""

    def __init__(
        self,
        events_path: str = "data/calendar/events.json",
        ics_dir: str = "data/calendar/ics",
        timezone_name: str = "Asia/Shanghai",
    ) -> None:
        self.events_path = Path(get_project_abs_path(events_path))
        self.ics_dir = Path(get_project_abs_path(ics_dir))
        try:
            self.timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            self.timezone = timezone.utc

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name != "create_event":
            return ToolResult(
                tool_name=f"calendar.{tool_name}",
                success=False,
                message=f"Unknown local calendar tool: {tool_name}",
            )
        try:
            return self.create_event(arguments)
        except (OSError, TypeError, ValueError) as exc:
            return ToolResult(
                tool_name="calendar.create_event",
                success=False,
                data={"error": str(exc)},
                message="本地日历事件生成失败。",
            )

    def create_event(self, arguments: dict[str, Any]) -> ToolResult:
        ticket = arguments.get("ticket") or {}
        slots = arguments.get("slots") or {}
        event_id = str(uuid.uuid4())

        start_at = self._resolve_start(ticket, slots)
        end_at = self._resolve_end(ticket, slots, start_at)
        movie = ticket.get("movieName") or slots.get("movieName") or "电影"
        cinema = ticket.get("cinemaName") or slots.get("cinemaName") or "影院"
        hall = ticket.get("hallName") or slots.get("hallName") or ""
        address = ticket.get("address") or slots.get("address") or ""
        seats = ticket.get("seats") or slots.get("seatIds") or []
        if isinstance(seats, list):
            seat_text = ",".join(str(item) for item in seats)
        else:
            seat_text = str(seats)

        event = {
            "id": event_id,
            "summary": f"{movie} 观影",
            "movie": movie,
            "cinema": cinema,
            "hall": hall,
            "address": address,
            "seats": seat_text,
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "created",
        }

        events = self._read_events()
        events.append(event)
        self._write_events(events)

        ics_path = self.ics_dir / f"{event_id}.ics"
        ics_path.parent.mkdir(parents=True, exist_ok=True)
        ics_path.write_text(self._to_ics(event), encoding="utf-8")

        return ToolResult(
            tool_name="calendar.create_event",
            success=True,
            data={
                "calendarEventId": event_id,
                "eventsPath": str(self.events_path),
                "icsPath": str(ics_path),
                "event": event,
            },
            message="已生成本地观影日历文件。",
        )

    def _resolve_start(
        self,
        ticket: dict[str, Any],
        slots: dict[str, Any],
    ) -> datetime:
        direct_value = ticket.get("startAt") or slots.get("startAt")
        parsed = self._parse_datetime(direct_value)
        if parsed:
            return parsed

        date_value = slots.get("date") or ticket.get("date") or "today"
        time_value = (
            slots.get("timeRange")
            or slots.get("time")
            or ticket.get("time")
            or "20:00"
        )
        local_date = self._parse_date(date_value)
        hour, minute = self._parse_clock(time_value)
        return datetime(
            local_date.year,
            local_date.month,
            local_date.day,
            hour,
            minute,
            tzinfo=self.timezone,
        )

    def _resolve_end(
        self,
        ticket: dict[str, Any],
        slots: dict[str, Any],
        start_at: datetime,
    ) -> datetime:
        direct_value = ticket.get("endAt") or slots.get("endAt")
        parsed = self._parse_datetime(direct_value)
        if parsed:
            return parsed
        duration = ticket.get("durationMinutes") or slots.get("durationMinutes") or 150
        return start_at + timedelta(minutes=int(duration))

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=self.timezone)
        if not isinstance(value, str):
            return None
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=self.timezone)

    def _parse_date(self, value: Any):
        now = datetime.now(self.timezone)
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"today", "今天", "今晚"}:
                return now.date()
            if normalized in {"tomorrow", "明天", "明晚"}:
                return (now + timedelta(days=1)).date()
            if normalized in {"weekend", "周末"}:
                days_until_saturday = (5 - now.weekday()) % 7
                return (now + timedelta(days=days_until_saturday)).date()
            try:
                return datetime.strptime(normalized, "%Y-%m-%d").date()
            except ValueError:
                pass
        return now.date()

    def _parse_clock(self, value: Any) -> tuple[int, int]:
        if isinstance(value, str):
            match = re.search(r"(\d{1,2})(?::|点)(\d{1,2})?", value)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2) or 0)
                if "下午" in value or "晚上" in value or "晚" in value:
                    if hour < 12:
                        hour += 12
                return min(hour, 23), min(minute, 59)
            normalized = value.lower()
            if normalized in {"morning", "上午"}:
                return 10, 0
            if normalized in {"afternoon", "下午"}:
                return 14, 0
            if normalized in {"evening", "晚上"}:
                return 19, 30
        return 20, 0

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        with self.events_path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Calendar events must be a JSON array: {self.events_path}")
        return data

    def _write_events(self, events: list[dict[str, Any]]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text(
            json.dumps(events, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _to_ics(self, event: dict[str, Any]) -> str:
        start = self._to_utc(event["start_at"])
        end = self._to_utc(event["end_at"])
        location = ", ".join(
            item for item in [event.get("cinema"), event.get("address")] if item
        )
        description = (
            f"电影：{event.get('movie', '')}\\n"
            f"影厅：{event.get('hall', '')}\\n"
            f"座位：{event.get('seats', '')}"
        )
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Movie Ticket Agent//CN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            "BEGIN:VEVENT",
            f"UID:{event['id']}@movie-ticket-agent",
            f"DTSTAMP:{self._utc_now()}",
            f"DTSTART:{start}",
            f"DTEND:{end}",
            f"SUMMARY:{self._ics_escape(event.get('summary', '观影'))}",
            f"LOCATION:{self._ics_escape(location)}",
            f"DESCRIPTION:{self._ics_escape(description)}",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
        return "\r\n".join(lines) + "\r\n"

    def _to_utc(self, value: str) -> str:
        parsed = self._parse_datetime(value)
        if not parsed:
            raise ValueError(f"Invalid calendar datetime: {value}")
        return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _ics_escape(self, value: str) -> str:
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\r\n", "\\n")
            .replace("\n", "\\n")
        )


class AppleCalDAVMCP(LocalCalendarMCP):
    """Upload generated ICS events to an iCloud calendar through CalDAV."""

    def __init__(
        self,
        username: str,
        app_password: str,
        calendar_name: str = "",
        server_url: str = "https://caldav.icloud.com/",
        events_path: str = "data/calendar/events.json",
        ics_dir: str = "data/calendar/ics",
        timezone_name: str = "Asia/Shanghai",
    ) -> None:
        super().__init__(
            events_path=events_path,
            ics_dir=ics_dir,
            timezone_name=timezone_name,
        )
        self.username = username
        self.app_password = app_password
        self.calendar_name = calendar_name
        self.server_url = server_url

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name != "create_event":
            return ToolResult(
                tool_name=f"calendar.{tool_name}",
                success=False,
                message=f"Unknown iCloud calendar tool: {tool_name}",
            )
        try:
            return self.create_event(arguments)
        except Exception as exc:
            return ToolResult(
                tool_name="calendar.create_event",
                success=False,
                data={"error": str(exc)},
                message="iCloud 日历同步失败，请检查 CalDAV 配置和 App 专用密码。",
            )

    def create_event(self, arguments: dict[str, Any]) -> ToolResult:
        if not self.username or not self.app_password:
            raise ValueError(
                "Missing APPLE_CALDAV_USERNAME or APPLE_CALDAV_APP_PASSWORD."
            )

        result = super().create_event(arguments)
        ics_path = Path(result.data["icsPath"])
        remote = self._upload_ics(ics_path)
        result.data["remote"] = remote
        result.message = "观影日程已生成并同步到 iCloud 日历。"
        return result

    def _upload_ics(self, ics_path: Path) -> dict[str, Any]:
        try:
            from caldav import get_davclient
        except ImportError as exc:
            raise RuntimeError(
                "未安装 caldav，请执行 .\\.venv\\Scripts\\python.exe -m pip install caldav"
            ) from exc

        with get_davclient(
            url=self.server_url,
            username=self.username,
            password=self.app_password,
            features="icloud",
        ) as client:
            principal = client.principal()
            calendars = principal.get_calendars()
            if not calendars:
                raise RuntimeError("iCloud 账户下没有可用日历。")

            calendar = self._select_calendar(calendars)
            remote_event = calendar.add_event(
                ics_path.read_text(encoding="utf-8")
            )
            return {
                "calendarName": getattr(calendar, "name", self.calendar_name),
                "eventUrl": str(getattr(remote_event, "url", "") or ""),
            }

    def _select_calendar(self, calendars: list[Any]) -> Any:
        if self.calendar_name:
            wanted = self.calendar_name.casefold()
            for calendar in calendars:
                name = str(getattr(calendar, "name", "") or "")
                if name.casefold() == wanted:
                    return calendar
            raise RuntimeError(f"找不到 iCloud 日历：{self.calendar_name}")
        return calendars[0]


class LocalSmsOutboxMCP:
    def __init__(
        self,
        outbox_path: str = "data/notifications/sms_outbox.json",
    ) -> None:
        self.outbox_path = Path(get_project_abs_path(outbox_path))

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name not in {"send_sms", "send_ticket_message"}:
            return ToolResult(
                tool_name=f"notification.{tool_name}",
                success=False,
                message=f"Unknown notification MCP tool: {tool_name}",
            )

        try:
            return self.write_sms(arguments)
        except (OSError, ValueError) as exc:
            return ToolResult(
                tool_name=f"notification.{tool_name}",
                success=False,
                data={"error": str(exc)},
                message="Local SMS outbox write failed.",
            )

    def write_sms(self, arguments: dict[str, Any]) -> ToolResult:
        phone = arguments.get("phone") or arguments.get("phoneNumber") or arguments.get("to")
        if not phone:
            raise ValueError("Missing phone number for SMS.")

        template_params = (
            arguments.get("template_params")
            or arguments.get("templateParam")
            or self._ticket_template_params(arguments)
        )
        message = arguments.get("message") or self._render_message(template_params)
        record = {
            "id": str(uuid.uuid4()),
            "channel": "sms",
            "phone": str(phone),
            "message": message,
            "template_params": template_params,
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        outbox = self._read_outbox()
        outbox.append(record)
        self._write_outbox(outbox)

        return ToolResult(
            tool_name="notification.send_sms",
            success=True,
            data={
                "notificationId": record["id"],
                "phone": phone,
                "outbox_path": str(self.outbox_path),
                "record": record,
            },
            message="SMS saved to local outbox.",
        )

    def _read_outbox(self) -> list[dict[str, Any]]:
        if not self.outbox_path.exists():
            return []
        with self.outbox_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"SMS outbox must contain a JSON array: {self.outbox_path}")
        return data

    def _write_outbox(self, outbox: list[dict[str, Any]]) -> None:
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox_path.open("w", encoding="utf-8") as f:
            json.dump(outbox, f, ensure_ascii=False, indent=2)

    def _render_message(self, params: dict[str, Any]) -> str:
        movie = params.get("movie") or "电影"
        cinema = params.get("cinema") or "影院"
        showtime = params.get("time") or "观影时间"
        seats = params.get("seats") or "座位"
        return f"您已成功出票：{movie}，{cinema}，{showtime}，座位：{seats}。请准时观影。"

    def _ticket_template_params(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ticket = arguments.get("ticket") or {}
        slots = arguments.get("slots") or {}
        return {
            "movie": slots.get("movieName") or ticket.get("movieName") or "电影",
            "cinema": slots.get("cinemaName") or ticket.get("cinemaName") or "影院",
            "time": slots.get("timeRange") or ticket.get("time") or "观影时间",
            "seats": ",".join(slots.get("seatIds", [])) if isinstance(slots.get("seatIds"), list) else slots.get("seatIds", ""),
        }


class MCPDispatcher:
    def __init__(self) -> None:
        mcp_settings = agent_config.get("mcp", {})
        self.clients: dict[str, MCPClient] = {
            "movie_ticket": LocalMovieTicketMCP(),
            "spring_boot": self._spring_boot_client(),
            "snack": LocalSnackMCP(),
            "coupon": LocalCouponMCP(),
            "amap": self._amap_client(mcp_settings),
            "calendar": self._calendar_client(mcp_settings),
            "notification": self._notification_client(mcp_settings),
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

    def _http_client(self, server_name: str, settings: dict[str, Any]) -> HttpMCPClient:
        server_settings = settings.get(server_name, {})
        return HttpMCPClient(
            server_name=server_name,
            base_url=server_settings.get("base_url"),
            token_env=server_settings.get("token_env"),
            timeout_seconds=server_settings.get("timeout_seconds", 15),
        )

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

    def _calendar_client(self, settings: dict[str, Any]) -> MCPClient:
        server_settings = settings.get("calendar", {})
        provider = server_settings.get("provider", "local_ics")
        if provider == "local_ics":
            return LocalCalendarMCP(
                events_path=server_settings.get(
                    "events_path",
                    "data/calendar/events.json",
                ),
                ics_dir=server_settings.get(
                    "ics_dir",
                    "data/calendar/ics",
                ),
                timezone_name=server_settings.get(
                    "timezone",
                    "Asia/Shanghai",
                ),
            )
        if provider == "apple_caldav":
            return AppleCalDAVMCP(
                username=os.getenv(
                    server_settings.get(
                        "username_env",
                        "APPLE_CALDAV_USERNAME",
                    ),
                    "",
                ),
                app_password=os.getenv(
                    server_settings.get(
                        "app_password_env",
                        "APPLE_CALDAV_APP_PASSWORD",
                    ),
                    "",
                ),
                calendar_name=os.getenv(
                    server_settings.get(
                        "calendar_name_env",
                        "APPLE_CALDAV_CALENDAR_NAME",
                    ),
                    "",
                ),
                server_url=server_settings.get(
                    "server_url",
                    "https://caldav.icloud.com/",
                ),
                events_path=server_settings.get(
                    "events_path",
                    "data/calendar/events.json",
                ),
                ics_dir=server_settings.get(
                    "ics_dir",
                    "data/calendar/ics",
                ),
                timezone_name=server_settings.get(
                    "timezone",
                    "Asia/Shanghai",
                ),
            )
        return CalendarMCP(self._http_client("calendar", settings))

    def _notification_client(self, settings: dict[str, Any]) -> MCPClient:
        server_settings = settings.get("notification", {})
        provider = server_settings.get("provider", "local_sms_outbox")
        if provider == "local_sms_outbox":
            return LocalSmsOutboxMCP(
                outbox_path=server_settings.get(
                    "outbox_path",
                    "data/notifications/sms_outbox.json",
                )
            )
        return NotificationMCP(self._http_client("notification", settings))


mcp_dispatcher = MCPDispatcher()
