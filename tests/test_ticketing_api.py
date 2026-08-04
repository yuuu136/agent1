from fastapi.testclient import TestClient

from app.main import app


def test_ticketing_api_traditional_purchase_flow() -> None:
    client = TestClient(app)

    movies = client.get("/api/v1/movies").json()
    assert movies["code"] == 0
    assert movies["data"]["movies"]

    showtimes = client.get("/api/v1/showtimes").json()
    assert showtimes["code"] == 0
    showtime_id = "st_2004"

    seats = client.get(f"/api/v1/showtimes/{showtime_id}/seats").json()
    assert seats["code"] == 0
    assert seats["data"]["seats"]

    order_response = client.post(
        "/api/v1/orders",
        json={
            "showtimeId": showtime_id,
            "seatIds": ["F7", "F8"],
            "ticketCount": 2,
            "userId": "traditional-test",
        },
    ).json()
    assert order_response["code"] == 0
    order_id = order_response["data"]["orderId"]

    pay_response = client.post(
        f"/api/v1/orders/{order_id}/pay",
        json={"idempotencyKey": "ticketing-api-test"},
    ).json()
    assert pay_response["code"] == 0
    assert pay_response["data"]["status"] == "TICKETED"
    assert pay_response["data"]["ticketStatus"] == "issued"


def test_admin_overview_returns_mock_metrics() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/admin/overview")
    payload = response.json()

    assert response.status_code == 200
    assert payload["code"] == 0
    assert payload["data"]["movieCount"] >= 1
    assert payload["data"]["showtimeCount"] >= 1


def test_purchase_draft_sync_and_admin_dashboard_contract() -> None:
    client = TestClient(app)

    draft_payload = client.get("/api/v1/drafts/active").json()
    assert draft_payload["code"] == 0
    draft = draft_payload["data"]

    updated_payload = client.post(
        "/api/v1/drafts",
        json={
            "draftId": draft["draftId"],
            "version": draft["version"],
            "movieId": "m_1001",
            "movieName": "流浪地球3",
            "ticketCount": 2,
        },
    ).json()
    assert updated_payload["code"] == 0
    assert updated_payload["data"]["state"] == "MOVIE_CONFIRMED"
    assert updated_payload["data"]["version"] == draft["version"] + 1

    dashboard_payload = client.get("/api/admin/dashboard").json()
    assert dashboard_payload["code"] == 0
    assert "todayRevenueRaw" in dashboard_payload["data"]
    assert "payConversionRate" in dashboard_payload["data"]
