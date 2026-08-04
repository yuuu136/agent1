import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.clients import mcp_dispatcher
from app.schemas.ticketing import (
    ApiResponse,
    CreateOrderRequest,
    MovieQuery,
    PayOrderRequest,
    ShowtimeQuery,
    UpdateDraftRequest,
)


router = APIRouter(prefix="/api/v1", tags=["ticketing"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])

_drafts: dict[int, dict[str, Any]] = {}
_draft_sequence = 1000


def _trace_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _response(data: Any, trace_id: str, message: str = "success") -> ApiResponse:
    return ApiResponse(code=0, message=message, data=data, traceId=trace_id)


def _raise_tool_error(result, trace_id: str, status_code: int = 400) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={
            "code": status_code,
            "message": result.message or "业务处理失败",
            "data": result.data,
            "traceId": trace_id,
        },
    )


def _movie_ticket(tool_name: str, arguments: dict[str, Any]):
    return mcp_dispatcher.call("movie_ticket", tool_name, arguments)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_draft_id() -> int:
    global _draft_sequence
    _draft_sequence += 1
    return _draft_sequence


def _new_draft(user_id: str = "demo-user") -> dict[str, Any]:
    draft_id = _next_draft_id()
    draft = {
        "draftId": draft_id,
        "version": 1,
        "userId": user_id,
        "state": "COLLECTING",
        "movieId": None,
        "movieName": None,
        "cinemaId": None,
        "cinemaName": None,
        "showtimeId": None,
        "date": None,
        "time": None,
        "ticketCount": 2,
        "seatIds": [],
        "budget": None,
        "seatZone": None,
        "orderId": None,
        "createdAt": _now_iso(),
        "updatedAt": _now_iso(),
    }
    _drafts[draft_id] = draft
    return draft


def _draft_state(draft: dict[str, Any]) -> str:
    if draft.get("orderId"):
        return "FROZEN"
    if draft.get("seatIds"):
        return "SEAT_SELECTING"
    if draft.get("showtimeId"):
        return "SHOWTIME_CONFIRMED"
    if draft.get("cinemaId"):
        return "CINEMA_CONFIRMED"
    if draft.get("movieId"):
        return "MOVIE_CONFIRMED"
    return "COLLECTING"


def _active_draft(user_id: str = "demo-user") -> dict[str, Any]:
    active = [
        draft
        for draft in _drafts.values()
        if draft.get("userId") == user_id and draft.get("state") != "ARCHIVED"
    ]
    if active:
        active.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
        return active[0]
    return _new_draft(user_id)


def _apply_draft_update(draft: dict[str, Any], request: UpdateDraftRequest) -> dict[str, Any]:
    if request.version is not None and request.version != draft.get("version"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": 409,
                "message": "草稿版本已变化，请刷新后重试",
                "data": deepcopy(draft),
                "traceId": _trace_id("draft-conflict"),
            },
        )

    before = deepcopy(draft)
    updates = request.model_dump(exclude_unset=True, exclude_none=True)
    updates.pop("draftId", None)
    updates.pop("version", None)

    for key, value in updates.items():
        draft[key] = value

    # 上游条件变化时清理下游选择，保持传统购票与 AI 购票只有一条交易真相。
    if before.get("movieId") and request.movieId and request.movieId != before.get("movieId"):
        draft["cinemaId"] = None
        draft["cinemaName"] = None
        draft["showtimeId"] = None
        draft["seatIds"] = []
    if before.get("cinemaId") and request.cinemaId and request.cinemaId != before.get("cinemaId"):
        draft["showtimeId"] = None
        draft["seatIds"] = []
    if before.get("showtimeId") and request.showtimeId and request.showtimeId != before.get("showtimeId"):
        draft["seatIds"] = []

    draft["version"] = int(draft.get("version", 0)) + 1
    draft["state"] = _draft_state(draft)
    draft["updatedAt"] = _now_iso()
    return draft


@router.get("/drafts/active")
def get_active_draft(userId: str = "demo-user") -> ApiResponse:
    trace_id = _trace_id("draft")
    return _response(deepcopy(_active_draft(userId)), trace_id)


@router.post("/drafts")
def update_draft(request: UpdateDraftRequest) -> ApiResponse:
    trace_id = _trace_id("draft-update")
    draft = (
        _drafts.get(request.draftId)
        if request.draftId is not None
        else _active_draft(request.userId or "demo-user")
    )
    if not draft:
        draft = _new_draft(request.userId or "demo-user")
    updated = _apply_draft_update(draft, request)
    return _response(deepcopy(updated), trace_id, "草稿已同步")


@router.get("/movies")
def list_movies(
    keyword: str | None = None,
    genre: str | None = None,
) -> ApiResponse:
    trace_id = _trace_id("movies")
    result = _movie_ticket(
        "search_movies",
        MovieQuery(keyword=keyword, genre=genre).model_dump(exclude_none=True),
    )
    if not result.success:
        _raise_tool_error(result, trace_id)
    return _response(result.data, trace_id, result.message)


@router.get("/cinemas")
def list_cinemas() -> ApiResponse:
    trace_id = _trace_id("cinemas")
    result = _movie_ticket("search_showtimes", {})
    if not result.success:
        _raise_tool_error(result, trace_id)

    cinemas: dict[str, dict[str, Any]] = {}
    for item in result.data.get("showtimes", []):
        cinema_id = item.get("cinemaId")
        if not cinema_id:
            continue
        cinemas.setdefault(
            cinema_id,
            {
                "cinemaId": cinema_id,
                "cinemaName": item.get("cinemaName"),
                "showtimeCount": 0,
                "minPrice": item.get("price"),
            },
        )
        cinemas[cinema_id]["showtimeCount"] += 1
        if item.get("price") is not None:
            cinemas[cinema_id]["minPrice"] = min(
                cinemas[cinema_id]["minPrice"],
                item["price"],
            )
    return _response({"cinemas": list(cinemas.values())}, trace_id)


@router.get("/showtimes")
def list_showtimes(
    movieId: str | None = None,
    movieName: str | None = None,
    cinemaId: str | None = None,
    genre: str | None = None,
    date: str | None = None,
    timeRange: str | None = None,
    ticketCount: int | None = None,
) -> ApiResponse:
    trace_id = _trace_id("showtimes")
    query = ShowtimeQuery(
        movieId=movieId,
        movieName=movieName,
        cinemaId=cinemaId,
        genre=genre,
        date=date,
        timeRange=timeRange,
        ticketCount=ticketCount,
    )
    result = _movie_ticket(
        "search_showtimes",
        query.model_dump(exclude_none=True),
    )
    if not result.success:
        _raise_tool_error(result, trace_id)
    return _response(result.data, trace_id, result.message)


@router.get("/showtimes/{showtime_id}/seats")
def get_showtime_seats(showtime_id: str) -> ApiResponse:
    trace_id = _trace_id("seats")
    result = _movie_ticket("get_seats", {"showtimeId": showtime_id})
    if not result.success:
        _raise_tool_error(result, trace_id, status_code=404)
    return _response(result.data, trace_id, result.message)


@router.post("/orders")
def create_order(request: CreateOrderRequest) -> ApiResponse:
    trace_id = _trace_id("order-create")
    lock_result = _movie_ticket(
        "lock_seats",
        {
            "showtimeId": request.showtimeId,
            "seatIds": request.seatIds,
            "ticketCount": request.ticketCount or len(request.seatIds),
            "userId": request.userId or "demo-user",
        },
    )
    if not lock_result.success:
        _raise_tool_error(lock_result, trace_id, status_code=409)

    order_result = _movie_ticket(
        "create_order",
        {
            "lockId": lock_result.data.get("lockId"),
            "showtimeId": request.showtimeId,
            "seatIds": request.seatIds,
        },
    )
    if not order_result.success:
        _raise_tool_error(order_result, trace_id, status_code=409)

    data = {
        **order_result.data,
        "lock": lock_result.data,
    }
    if request.draftId in _drafts:
        draft = _drafts[int(request.draftId)]
        draft["orderId"] = data.get("orderId")
        draft["showtimeId"] = request.showtimeId
        draft["seatIds"] = list(request.seatIds)
        draft["state"] = _draft_state(draft)
        draft["version"] = int(draft.get("version", 0)) + 1
        draft["updatedAt"] = _now_iso()
    return _response(data, trace_id, order_result.message)


@router.get("/orders")
def list_orders(
    status: str | None = Query(default=None),
) -> ApiResponse:
    trace_id = _trace_id("orders")
    result = _movie_ticket(
        "list_orders",
        {"status": status} if status else {},
    )
    if not result.success:
        _raise_tool_error(result, trace_id)
    return _response(result.data, trace_id, result.message)


@router.get("/orders/{order_id}")
def get_order(order_id: str) -> ApiResponse:
    trace_id = _trace_id("order")
    result = _movie_ticket("get_order", {"orderId": order_id})
    if not result.success:
        _raise_tool_error(result, trace_id, status_code=404)
    return _response(result.data, trace_id, result.message)


@router.post("/orders/{order_id}/pay")
def pay_order(order_id: str, request: PayOrderRequest | None = None) -> ApiResponse:
    trace_id = _trace_id("order-pay")
    pay_result = _movie_ticket("pay_order", {"orderId": order_id})
    if not pay_result.success:
        _raise_tool_error(pay_result, trace_id, status_code=409)

    issue_result = _movie_ticket("issue_ticket", {"orderId": order_id})
    if issue_result.success:
        pay_result.data.update(issue_result.data)
        pay_result.data["ticketStatus"] = issue_result.data.get("ticketStatus", "issued")

    return _response(pay_result.data, trace_id, issue_result.message or pay_result.message)


@router.get("/admin/overview")
def admin_overview() -> ApiResponse:
    trace_id = _trace_id("admin-overview")
    movie_result = _movie_ticket("search_movies", {})
    showtime_result = _movie_ticket("search_showtimes", {})
    order_result = _movie_ticket("list_orders", {})

    if not movie_result.success:
        _raise_tool_error(movie_result, trace_id)
    if not showtime_result.success:
        _raise_tool_error(showtime_result, trace_id)
    if not order_result.success:
        _raise_tool_error(order_result, trace_id)

    return _response(
        _admin_overview_data(
            movie_result.data,
            showtime_result.data,
            order_result.data,
        ),
        trace_id,
    )


def _admin_overview_data(
    movie_data: dict[str, Any],
    showtime_data: dict[str, Any],
    order_data: dict[str, Any],
) -> dict[str, Any]:
    orders = order_data.get("orders", [])
    paid_orders = [
        order
        for order in orders
        if str(order.get("status", "")).upper() in {"PAID", "TICKETED"}
    ]
    revenue = sum(float(order.get("amount", 0)) for order in paid_orders)
    return {
        "movieCount": len(movie_data.get("movies", [])),
        "showtimeCount": len(showtime_data.get("showtimes", [])),
        "orderCount": len(orders),
        "paidOrderCount": len(paid_orders),
        "revenue": revenue,
        "orders": orders[:10],
        "todayOrderCount": len(orders),
        "todayRevenueRaw": int(revenue * 100),
        "payConversionRate": round(len(paid_orders) / len(orders), 4)
        if orders
        else 0,
        "topMovies": movie_data.get("movies", [])[:5],
        "seatSales": [
            {
                "status": "SOLD",
                "count": sum(len(order.get("seatIds", [])) for order in paid_orders),
            },
            {
                "status": "LOCKED",
                "count": sum(
                    len(order.get("seatIds", []))
                    for order in orders
                    if str(order.get("status", "")).upper() == "PAYMENT_PENDING"
                ),
            },
        ],
    }


@admin_router.get("/dashboard")
def admin_dashboard() -> ApiResponse:
    trace_id = _trace_id("admin-dashboard")
    movie_result = _movie_ticket("search_movies", {})
    showtime_result = _movie_ticket("search_showtimes", {})
    order_result = _movie_ticket("list_orders", {})

    if not movie_result.success:
        _raise_tool_error(movie_result, trace_id)
    if not showtime_result.success:
        _raise_tool_error(showtime_result, trace_id)
    if not order_result.success:
        _raise_tool_error(order_result, trace_id)

    return _response(
        _admin_overview_data(
            movie_result.data,
            showtime_result.data,
            order_result.data,
        ),
        trace_id,
    )
