from copy import deepcopy
from typing import Any

from app.schemas.agent import AgentState
from app.utils.config_handler import agent_config


GENRE_VALUES = {"喜剧", "爱情", "动作", "科幻", "动画", "悬疑", "恐怖"}
HALL_TYPE_VALUES = {"IMAX", "杜比", "巨幕", "激光"}
INVALID_MOVIE_NAME_VALUES = {"影院", "选择影院", "择影院", "电影", "选择电影", "择电影"}
INVALID_MOVIE_NAME_VALUES.update(
    {
        "好",
        "好的",
        "好吧",
        "就好",
        "就行",
        "可以",
        "可以的",
        "行",
        "行吧",
        "都可以",
        "都行",
        "随便",
        "不限",
        "时间不限",
        "什么时候都可以",
        "哪个时间都可以",
        "无所谓",
        "嗯",
        "哦",
        "知道了",
        "确认",
        "确认一下",
        "确认订单",
        "这个",
        "这场",
        "这家",
        "就这个",
        "就这场",
        "就这家",
        "不用了",
        "算了",
        "先不买",
        "不买了",
        "别买了",
        "换一场",
        "换个场次",
        "换时间",
        "换个时间",
        "早一点",
        "晚一点",
        "便宜点",
        "换便宜点",
        "不要这个",
        "不要这场",
    }
)
INVALID_MOVIE_NAME_VALUES.update({"换便宜点", "便宜点", "晚一点", "早一点", "换一场", "换时间"})
INVALID_MOVIE_NAME_PATTERNS = ("附近", "周边", "最近", "离我近", "有什么影院")


class InMemorySessionStore:
    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}

    def get(self, session_id: str, user_id: str | None = None) -> AgentState:
        if session_id not in self._states:
            settings = agent_config.get("agent", {})
            self._states[session_id] = AgentState(
                session_id=session_id,
                user_id=user_id,
                state=settings.get("default_state", "idle"),
                slots={
                    "city": settings.get("default_city", ""),
                    "seatPreference": settings.get("default_seat_preference", "middle"),
                },
            )
        state = self._states[session_id]
        if user_id and not state.user_id:
            state.user_id = user_id
        return state.model_copy(deep=True)

    def save(self, state: AgentState) -> None:
        self._states[state.session_id] = state.model_copy(deep=True)


def merge_slots(current_slots: dict[str, Any], new_slots: dict[str, Any],
                 new_intent: str = "") -> dict[str, Any]:
    merged = deepcopy(current_slots)

    # Honour explicit clearing
    clear_slots = new_slots.get("__clearSlots") or []
    if isinstance(clear_slots, list):
        for key in clear_slots:
            if isinstance(key, str):
                merged.pop(key, None)

    # Accumulate new values
    for key, value in new_slots.items():
        if key == "__clearSlots":
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue
        merged[key] = value

    # Basic sanity: movieName shouldn't be a genre keyword or hall type
    movie_name = str(merged.get("movieName") or "").strip()
    if movie_name in INVALID_MOVIE_NAME_VALUES or _is_invalid_movie_name(movie_name):
        merged.pop("movieName", None)
    elif "genre" in new_slots and movie_name in GENRE_VALUES:
        merged.pop("movieName", None)
    elif "hallType" in new_slots and movie_name.upper() in HALL_TYPE_VALUES:
        merged.pop("movieName", None)

    return merged


def _is_invalid_movie_name(movie_name: str) -> bool:
    if not movie_name:
        return False
    normalized = movie_name.strip("的 ")
    if normalized.startswith("想") and normalized[1:] in GENRE_VALUES:
        return True
    return any(pattern in movie_name for pattern in INVALID_MOVIE_NAME_PATTERNS)


session_store = InMemorySessionStore()
