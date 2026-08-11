"""Dialogue State Tracker — knows what each flow needs and what's missing."""

from typing import Any

from app.schemas.agent import AgentState, NLUResult


# ── Flow definitions ──────────────────────────────────────────────
# Each flow lists required slots (must-have before execution) and
# optional slots (nice-to-have, don't block progress).

FLOWS: dict[str, dict[str, Any]] = {
    "book_ticket": {
        "required": [],           # No hard requirement — book_ticket can
        "ask_for": ["movieName", "ticketCount", "date", "timeRange", "cinemaName", "seatPreference"],
        "execute": "search_movies",
        "description": "订票",
    },
    "search_movies": {
        "required": [],
        "ask_for": ["movieName", "genre"],
        "execute": "search_movies",
        "description": "搜电影",
    },
    "select_showtime": {
        "required": ["movieName"],
        "ask_for": ["date", "timeRange"],
        "execute": "search_showtimes",
        "description": "选场次",
    },
    "select_seats": {
        "required": ["showtimeId"],
        "ask_for": [],
        "execute": "get_seats",
        "description": "选座",
    },
    "confirm_order": {
        "required": ["showtimeId", "seatIds"],
        "ask_for": [],
        "execute": "lock_seats",
        "description": "确认订单",
    },
}

# Ask-for order within each flow: first missing from this list gets asked.
ASK_ORDER = ["movieName", "genre", "ticketCount", "date", "timeRange",
             "cinemaName", "seatPreference", "showtimeId", "seatIds"]

# Slots that are "informational" and never asked for directly.
INFO_SLOTS = {"hallType", "seatPreference", "pricePreference", "timePreference",
              "maxPrice", "notHallType", "seatType", "recommendationCriteria",
              "cinemaLimit", "movieLimit", "location", "city", "orderId",
              "snackIds", "snackItems", "snackRequests", "seatPositions",
              "cinemaId", "__clearSlots"}


def _current_flow(state: AgentState, nlu_intent: str) -> str:
    """Determine which flow we're in based on state + NLU intent."""
    # If user is in the middle of a booking, preserve the flow
    if state.state and state.state.startswith(("collecting_", "selecting_", "locking_", "paying")):
        return "book_ticket"
    if nlu_intent in ("book_ticket", "select_or_modify"):
        return "book_ticket"
    if nlu_intent in ("search_showtimes",):
        return "select_showtime"
    if nlu_intent in ("seat_query", "confirm_order"):
        return "select_seats"
    if nlu_intent == "search_movies":
        return "search_movies"
    if nlu_intent == "nearby_cinema":
        return "nearby_cinema"
    return nlu_intent


class TrackerResult:
    """Output of the DST: what's filled, missing, and what to do."""

    def __init__(self, flow: str, filled: dict[str, Any],
                 missing: list[str], stage: str):
        self.flow = flow
        self.filled = filled
        self.missing = missing
        self.stage = stage

    @property
    def ready(self) -> bool:
        """True when all required slots are filled."""
        return len(self.missing) == 0

    @property
    def next_ask(self) -> str | None:
        """The next slot the system should ask for."""
        return self.missing[0] if self.missing else None


class DialogueTracker:
    """Tracks dialogue progress: what slots are filled, what's still needed."""

    def assess(self, state: AgentState, nlu: NLUResult) -> TrackerResult:
        flow = _current_flow(state, nlu.intent)
        flow_def = FLOWS.get(flow, {})

        # Accumulate all known slots from state + NLU
        filled: dict[str, Any] = {}
        for key, value in {**state.slots, **nlu.slots}.items():
            if key in INFO_SLOTS:
                if value not in (None, "", [], {}):
                    filled[key] = value
                continue
            if value not in (None, "", [], {}):
                filled[key] = value

        # Which slots do we care about for this flow?
        ask_for = flow_def.get("ask_for", [])
        required = flow_def.get("required", [])

        # Extract missing slots, preserving order.
        # A filled genre counts as satisfying the movieName requirement.
        missing: list[str] = []
        for slot in ASK_ORDER:
            if slot in ask_for or slot in required:
                if slot not in filled:
                    if slot == "movieName" and "genre" in filled:
                        continue  # genre is sufficient to search
                    if slot == "cinemaName" and (
                        "cinemaId" in filled
                        or ("nearbyFirst" in filled and "location" in filled)
                    ):
                        continue
                    if slot == "seatPreference" and (
                        "seatPositions" in filled
                        or "seatIds" in filled
                        or "autoSelectSeats" in filled
                        or "seatType" in filled
                    ):
                        continue
                    missing.append(slot)

        # Stage name
        if missing:
            stage = f"collecting_{missing[0]}"
        elif flow == "book_ticket":
            stage = "selecting_showtime"
        else:
            stage = flow

        return TrackerResult(flow, filled, missing, stage)


tracker = DialogueTracker()
