from fastapi.testclient import TestClient

from app.agent.cards import card_builder
from app.main import app
from app.schemas.agent import AgentPlan, ToolResult


class MockSpringNearbyResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "code": 1,
            "msg": "success",
            "data": {
                "total": 1,
                "records": [
                    {
                        "id": 1001,
                        "name": "Database Cinema",
                        "address": "Test Road",
                        "longitude": 121.4737,
                        "latitude": 31.2304,
                        "distance": 1.2,
                        "hallTypes": ["IMAX"],
                        "minPrice": 39,
                    }
                ],
            },
        }


def test_chat_booking_text_requires_login_for_real_ticketing() -> None:
    client = TestClient(app)

    response = client.post(
        "/agent/chat",
        json={
            "sessionId": "agent-booking",
            "text": "book movie tickets",
            "payload": {
                "slots": {
                    "genre": "comedy",
                    "date": "tomorrow",
                    "timeRange": "20:00",
                    "ticketCount": 2,
                }
            },
        },
    )

    payload = response.json()
    assert response.status_code == 200
    # DST-based planner may route to collecting_* or selecting_* depending on NLU output
    assert "collecting" in payload["state"] or "selecting" in payload["state"]
    assert payload["cards"] == []
    assert payload["message"] == "请先登录后再使用真实票务服务。"


def test_chat_empty_new_session_returns_greeting() -> None:
    client = TestClient(app)

    response = client.post(
        "/agent/chat",
        json={
            "sessionId": "agent-greeting",
            "text": "",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"] == "greeting"
    assert "想看" in payload["message"]
    assert "附近有什么电影院" in payload["suggestions"]


def test_stream_greeting_ignores_frontend_draft_context() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/agent/chat/stream",
        json={
            "sessionId": "agent-greeting-with-draft",
            "draftId": 1,
            "message": "你好",
        },
    )

    assert response.status_code == 200
    assert '"state": "greeting"' in response.text
    assert "event: done" in response.text


def test_agent_cors_allows_lan_frontend_origin() -> None:
    client = TestClient(app)

    response = client.options(
        "/api/agent/chat/stream",
        headers={
            "Origin": "http://192.168.1.23:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://192.168.1.23:8000"


def test_movie_card_includes_poster_payload_for_frontend_card_face() -> None:
    cards = card_builder.build(
        AgentPlan(action="search_movies", state="selecting_movie"),
        ToolResult(
            tool_name="spring_boot.search_movies",
            data={
                "movies": [
                    {
                        "movieId": 8,
                        "movieName": "蜘蛛侠",
                        "genre": "动作",
                        "score": 9.1,
                        "posterUrl": "https://example.com/spider.jpg",
                    }
                ]
            },
        ),
    )

    assert cards[0]["type"] == "movie"
    assert cards[0]["title"] == "蜘蛛侠"
    assert cards[0]["image"] == "https://example.com/spider.jpg"
    assert cards[0]["posterUrl"] == "https://example.com/spider.jpg"
    assert cards[0]["payload"]["posterUrl"] == "https://example.com/spider.jpg"


def test_get_order_ticketed_result_builds_ticket_card() -> None:
    cards = card_builder.build(
        AgentPlan(action="get_order", state="answering"),
        ToolResult(
            tool_name="spring_boot.get_order",
            data={
                "orderId": 12,
                "status": "TICKETED",
                "ticketStatus": "issued",
                "tickets": [{"ticketCode": "TKT-001"}],
            },
            message="支付成功，电子票已出票。",
        ),
    )

    assert cards[0]["type"] == "ticket"
    assert cards[0]["title"] == "电子票"
    assert cards[0]["actions"][0]["event"] == "view_ticket"
    assert cards[0]["actions"][0]["label"] == "查看电子票"
    assert cards[0]["actions"][0]["payload"]["path"] == "/orders/12/tickets"


def test_chat_nearby_cinema_uses_spring_database_nearby_api(monkeypatch) -> None:
    captured = {}

    def mock_get(*args, **kwargs):
        captured["url"] = args[0]
        captured["params"] = kwargs.get("params")
        captured["headers"] = kwargs.get("headers")
        return MockSpringNearbyResponse()

    monkeypatch.setattr("app.clients.mcp.httpx.get", mock_get)
    client = TestClient(app)

    response = client.post(
        "/agent/chat",
        json={
            "sessionId": "agent-nearby",
            "text": "nearby cinema",
            "jwt": "user-token",
            "payload": {"location": "31.2304,121.4737"},
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"] == "selecting_cinema"
    assert payload["cards"][0]["type"] == "cinema"
    assert payload["cards"][0]["title"] == "Database Cinema"
    assert captured["url"].endswith("/api/user/cinemas/nearby")
    assert captured["params"]["lat"] == 31.2304
    assert captured["params"]["lng"] == 121.4737
    assert captured["headers"]["Authorization"] == "Bearer user-token"


def test_chat_nearby_cinema_does_not_fallback_to_an_unknown_city() -> None:
    client = TestClient(app)

    response = client.post(
        "/agent/chat",
        json={
            "sessionId": "agent-nearby-without-location",
            "text": "附近有什么影院",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"] == "selecting_cinema"
    assert payload["cards"] == []
    assert "位置" in payload["message"] or "定位" in payload["message"]


def test_select_showtime_requires_login_for_real_seat_map() -> None:
    client = TestClient(app)

    response = client.post(
        "/agent/chat",
        json={
            "sessionId": "agent-seat",
            "event": "select_showtime",
            "payload": {"showtimeId": "st_2001", "ticketCount": 2},
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"] == "selecting_seats"
    assert payload["cards"] == []
    assert payload["message"] == "请先登录后再使用真实票务服务。"


def test_confirm_order_requires_login_for_real_lock() -> None:
    client = TestClient(app)

    response = client.post(
        "/agent/chat",
        json={
            "sessionId": "agent-confirm",
            "event": "confirm_order",
            "payload": {"showtimeId": "st_2001", "seatIds": ["A1", "A2"], "ticketCount": 2},
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"] == "locking_seats"
    assert payload["cards"] == []
    assert payload["message"] == "请先登录后再使用真实票务服务。"


def test_stream_chat_returns_sse_events() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/agent/chat/stream",
        json={
            "sessionId": "agent-stream",
            "message": "book movie tickets",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: thinking" in response.text
    assert "event: message" in response.text
    assert "event: done" in response.text
    assert '"node": "nlu"' in response.text
    assert '"node": "planner"' in response.text


def test_full_movie_ticket_flow_without_login_stops_before_database() -> None:
    client = TestClient(app)
    session_id = "agent-full-flow"

    showtime_response = client.post(
        "/agent/chat",
        json={
            "sessionId": session_id,
            "text": "帮我订两张明晚喜剧片",
            "payload": {
                "slots": {
                    "genre": "喜剧",
                    "date": "tomorrow",
                    "timeRange": "20:00",
                    "ticketCount": 2,
                }
            },
        },
    ).json()
    assert showtime_response["cards"] == []
    assert showtime_response["message"] == "请先登录后再使用真实票务服务。"
