import json

from app.clients.mcp import LocalCalendarMCP


def test_local_calendar_mcp_writes_json_and_ics(tmp_path) -> None:
    events_path = tmp_path / "events.json"
    ics_dir = tmp_path / "ics"
    calendar = LocalCalendarMCP(
        events_path=str(events_path),
        ics_dir=str(ics_dir),
        timezone_name="Asia/Shanghai",
    )

    result = calendar.call_tool(
        "create_event",
        {
            "ticket": {
                "movieName": "流浪地球3",
                "cinemaName": "万达影城",
                "hallName": "IMAX厅",
                "ticketStatus": "issued",
            },
            "slots": {
                "date": "tomorrow",
                "timeRange": "20:00",
                "seatIds": ["5排7座", "5排8座"],
                "address": "北京市朝阳区建国路",
            },
        },
    )

    assert result.success is True
    event_id = result.data["calendarEventId"]
    ics_path = ics_dir / f"{event_id}.ics"
    assert events_path.exists()
    assert ics_path.exists()

    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert events[0]["movie"] == "流浪地球3"
    ics_content = ics_path.read_text(encoding="utf-8")
    assert "BEGIN:VCALENDAR" in ics_content
    assert "SUMMARY:流浪地球3 观影" in ics_content
    assert "5排7座" in ics_content
