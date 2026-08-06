from app.agent.nlu import nlu_engine
from app.agent.planner import task_planner
from app.agent.reference import reference_resolver
from app.agent.tools import agent_toolbox
from app.schemas.agent import AgentState, ChatRequest


def test_price_query_reads_current_showtime_without_reloading_seats() -> None:
    state = AgentState(
        session_id="price-query-test",
        state="selecting_seats",
        pending_action="get_seats",
        slots={"showtimeId": "st_2001", "ticketCount": 2},
        selected={
            "showtime_candidates": [
                {
                    "showtimeId": "st_2001",
                    "movieName": "流浪地球3",
                    "price": 42,
                }
            ],
            "seat_map": {
                "showtimeId": "st_2001",
                "seats": [{"seatId": "638", "row": 6, "number": 3, "price": 42}],
            },
        },
    )
    request = ChatRequest(sessionId=state.session_id, text="这个多少钱")

    nlu = nlu_engine.extract(request)
    resolved = reference_resolver.resolve(state, nlu)
    plan = task_planner.plan(state, resolved)
    result = agent_toolbox.execute(plan, state)

    assert nlu.intent == "price_query"
    assert plan.action == "answer_price"
    assert result.success is True
    assert result.data["unitPrice"] == 42
    assert result.data["totalPrice"] == 84
    assert "42元/张" in result.message


def test_price_range_phrase_is_not_extracted_as_movie_name() -> None:
    result = nlu_engine.extract(
        ChatRequest(sessionId="price-phrase-test", text="什么价位")
    )

    assert result.intent == "price_query"
    assert "movieName" not in result.slots


def test_standalone_movie_title_searches_movie_cards_first() -> None:
    result = nlu_engine.extract(
        ChatRequest(sessionId="movie-title-test", text="奥德赛")
    )

    assert result.intent == "search_movies"
    assert result.slots["movieName"] == "奥德赛"
