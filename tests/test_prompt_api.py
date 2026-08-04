from fastapi.testclient import TestClient

from app.main import app


def test_prompt_list_api() -> None:
    client = TestClient(app)

    response = client.get("/agent/prompts")

    assert response.status_code == 200
    assert "rag_answer" in response.json()["prompts"]


def test_prompt_detail_api() -> None:
    client = TestClient(app)

    response = client.get("/agent/prompts/rag_answer")

    assert response.status_code == 200
    assert response.json()["content"].strip()
