from typing import Any, Literal

from pydantic import BaseModel, Field


AgentAction = Literal[
    "ask_movie_or_genre",
    "ask_time",
    "ask_ticket_count",
    "answer_with_rag",
    "answer_price",
    "search_nearby_cinemas",
    "search_movies",
    "search_showtimes",
    "get_seats",
    "recommend_snacks",
    "recommend_coupons",
    "confirm_selection",
    "lock_seats",
    "create_order",
    "pay_order",
    "issue_ticket",
    "create_calendar_event",
    "send_ticket_message",
    "cancel",
    "smalltalk",
]


class ChatRequest(BaseModel):
    sessionId: str
    userId: str | None = None
    jwt: str | None = None
    type: str = "text"
    text: str | None = None
    event: str | None = None
    payload: dict[str, Any] | None = None


class StreamChatRequest(BaseModel):
    sessionId: str
    draftId: int | None = None
    message: str
    event: str | None = None
    jwt: str | None = None
    userId: str | None = None
    payload: dict[str, Any] | None = None


class DraftMergeRequest(BaseModel):
    draftId: int
    jwt: str | None = None
    sessionId: str | None = None
    userId: str | None = None


class NLUResult(BaseModel):
    intent: str = "smalltalk"
    confidence: float = 0.0
    slots: dict[str, Any] = Field(default_factory=dict)
    is_modification: bool = False
    reference_text: str = ""
    missing_slots: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    session_id: str
    user_id: str | None = None
    state: str = "idle"
    intent: str = "smalltalk"
    slots: dict[str, Any] = Field(default_factory=dict)
    selected: dict[str, Any] = Field(default_factory=dict)
    pending_action: str | None = None
    last_user_text: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)


class AgentPlan(BaseModel):
    action: AgentAction
    reason: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    state: str = "idle"


class ToolResult(BaseModel):
    tool_name: str
    success: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    cards: list[dict[str, Any]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    message: str
    state: str
    cards: list[dict[str, Any]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    session: dict[str, Any] = Field(default_factory=dict)
