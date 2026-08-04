import json

from app.clients.mcp import LocalSmsOutboxMCP


def test_local_sms_outbox_mcp_writes_sms_record(tmp_path) -> None:
    outbox_path = tmp_path / "sms_outbox.json"

    result = LocalSmsOutboxMCP(outbox_path=str(outbox_path)).call_tool(
        "send_sms",
        {
            "phone": "13800138000",
            "template_params": {
                "movie": "Movie",
                "cinema": "Cinema",
                "time": "20:00",
                "seats": "A1,A2",
            },
        },
    )

    assert result.success is True
    assert result.message == "SMS saved to local outbox."

    records = json.loads(outbox_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["phone"] == "13800138000"
    assert records[0]["status"] == "pending"
    assert "Movie" in records[0]["message"]
