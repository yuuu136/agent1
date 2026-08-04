from fastapi.testclient import TestClient

from app.main import app


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


def test_chat_booking_text_returns_showtime_cards() -> None:
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
    assert payload["state"] == "selecting_showtime"
    assert payload["cards"][0]["type"] == "showtime"


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
    assert "电影票智能体" in payload["message"]
    assert "附近有什么电影院" in payload["suggestions"]


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
    assert "当前位置" in payload["message"]


def test_select_showtime_returns_seat_map() -> None:
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
    assert payload["cards"][0]["type"] == "seat_map"


def test_confirm_order_locks_seats_and_creates_order() -> None:
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
    assert payload["session"]["slots"]["orderId"].startswith("ord_")
    assert payload["cards"][0]["type"] == "confirm_order"


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


def test_full_movie_ticket_flow_reaches_ticket_card() -> None:
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
    showtime = showtime_response["cards"][0]["id"]

    client.post(
        "/agent/chat",
        json={
            "sessionId": session_id,
            "event": "select_showtime",
            "payload": {"showtimeId": showtime, "ticketCount": 2},
        },
    )

    order_response = client.post(
        "/agent/chat",
        json={
            "sessionId": session_id,
            "event": "confirm_order",
            "payload": {
                "showtimeId": showtime,
                "seatIds": ["A1", "A2"],
                "ticketCount": 2,
            },
        },
    ).json()
    order_id = order_response["session"]["slots"]["orderId"]

    pay_response = client.post(
        "/agent/chat",
        json={
            "sessionId": session_id,
            "event": "pay_order",
            "payload": {"orderId": order_id},
        },
    ).json()

    assert pay_response["cards"][0]["type"] == "ticket"
    assert pay_response["cards"][0]["meta"]["ticketStatus"] == "issued"
