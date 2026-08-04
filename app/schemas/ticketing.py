from typing import Any

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: Any = None
    traceId: str


class MovieQuery(BaseModel):
    keyword: str | None = None
    genre: str | None = None


class ShowtimeQuery(BaseModel):
    movieId: str | None = None
    movieName: str | None = None
    cinemaId: str | None = None
    genre: str | None = None
    date: str | None = None
    timeRange: str | None = None
    ticketCount: int | None = None


class UpdateDraftRequest(BaseModel):
    draftId: int | None = None
    version: int | None = None
    userId: str | None = None
    movieId: str | None = None
    movieName: str | None = None
    cinemaId: str | None = None
    cinemaName: str | None = None
    showtimeId: str | None = None
    date: str | None = None
    time: str | None = None
    ticketCount: int | None = Field(default=None, ge=1, le=8)
    seatIds: list[str] | None = None
    budget: int | None = None
    seatZone: str | None = None


class CreateOrderRequest(BaseModel):
    showtimeId: str
    seatIds: list[str] = Field(min_length=1)
    ticketCount: int | None = None
    userId: str | None = None
    draftId: int | None = None


class PayOrderRequest(BaseModel):
    idempotencyKey: str | None = None
    phone: str | None = None
    userId: str | None = None
