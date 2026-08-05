import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent import agent_service
from app.schemas.agent import ChatRequest, DraftMergeRequest, StreamChatRequest


router = APIRouter(prefix="/api/agent", tags=["agent"])


CARD_TYPE_MAP = {
    "movie": "MOVIE_LIST",
    "cinema": "CINEMA_LIST",
    "showtime": "SHOWTIME_LIST",
    "seat_map": "SEAT_MAP",
    "confirm_order": "ORDER_CONFIRM",
    "payment": "PAYMENT",
    "ticket": "TICKET",
    "alternative": "ALTERNATIVE",
    "location_picker": "LOCATION_PICKER",
    "snack": "SNACK_LIST",
    "coupon": "COUPON_LIST",
}

NODE_THINKING = {
    "nlu": ("NLU_ANALYZING", "正在理解需求..."),
    "merge_context_initial": ("MEMORY_MERGING", "正在加载上下文..."),
    "reference": ("REFERENCE_RESOLVING", "正在处理指代和上下文..."),
    "merge_context_after_reference": ("MEMORY_MERGING", "正在同步最新条件..."),
    "planner": ("PLANNING", "正在规划下一步..."),
    "ask": ("ASKING", "正在确认缺少的信息..."),
    "tool": ("TOOL_EXECUTING", "正在查询和处理..."),
    "apply_result": ("RESULT_APPLYING", "正在整理处理结果..."),
}


def _event(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _message_deltas(trace_id: str, content: str) -> Iterator[str]:
    """Emit the assistant text in small SSE chunks so clients can render it incrementally."""
    if not content:
        yield _event(
            "message",
            {
                "traceId": trace_id,
                "type": "text",
                "content": "",
                "delta": True,
            },
        )
        return

    chunk_size = 4
    for index in range(0, len(content), chunk_size):
        if index:
            time.sleep(0.02)
        yield _event(
            "message",
            {
                "traceId": trace_id,
                "type": "text",
                "content": content[index : index + chunk_size],
                "delta": True,
            },
        )


def _stream_response(request: StreamChatRequest) -> Iterator[str]:
    trace_id = str(uuid.uuid4())

    try:
        payload: dict[str, Any] = {}
        if request.draftId is not None:
            payload["draftId"] = request.draftId
        if request.payload:
            payload.update(request.payload)
        chat_request = ChatRequest(
            sessionId=request.sessionId,
            memoryId=request.memoryId,
            userId=request.userId,
            jwt=request.jwt,
            text=request.message,
            event=request.event,
            payload=payload or None,
        )

        completed: list[str] = []
        for node_name, update in agent_service.stream_chat(chat_request):
            if node_name == "response":
                response = update["response"]
                yield from _message_deltas(trace_id, response.message)

                for card in response.cards:
                    card_payload = dict(card)
                    card_payload["type"] = CARD_TYPE_MAP.get(
                        str(card_payload.get("type", "")),
                        str(card_payload.get("type", "CARD")).upper(),
                    )
                    yield _event(
                        "card",
                        {
                            "traceId": trace_id,
                            "type": card_payload["type"],
                            "data": card_payload,
                        },
                    )

                yield _event(
                    "progress",
                    {
                        "traceId": trace_id,
                        "step": response.state,
                        "completed": completed,
                    },
                )
                yield _event(
                    "done",
                    {
                        "traceId": trace_id,
                        "state": response.state,
                        "memoryId": response.session.get("memoryId"),
                    },
                )
                continue

            if node_name == "greeting":
                response = update["response"]
                yield from _message_deltas(trace_id, response.message)
                yield _event(
                    "done",
                    {
                        "traceId": trace_id,
                        "state": response.state,
                        "memoryId": response.session.get("memoryId"),
                    },
                )
                continue

            completed.append(node_name)
            status_message = NODE_THINKING.get(
                node_name,
                ("PROCESSING", "正在处理..."),
            )
            yield _event(
                "thinking",
                {
                    "traceId": trace_id,
                    "status": status_message[0],
                    "message": status_message[1],
                    "node": node_name,
                    "completed": completed,
                },
            )
    except Exception as exc:
        yield _event(
            "error",
            {
                "traceId": trace_id,
                "code": 500,
                "message": "Agent 处理失败，请稍后重试。",
                "degraded": True,
                "detail": str(exc),
            },
        )


@router.post("/chat/stream")
def stream_chat(request: StreamChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _draft_merge_response(request: DraftMergeRequest) -> Iterator[str]:
    trace_id = str(uuid.uuid4())
    yield _event(
        "error",
        {
            "traceId": trace_id,
            "code": 501,
            "message": "草稿合并接口已预留，等待 Java 票务中台提供草稿查询接口。",
            "degraded": True,
            "draftId": request.draftId,
        },
    )


@router.post("/draft/merge")
def merge_draft(request: DraftMergeRequest) -> StreamingResponse:
    return StreamingResponse(
        _draft_merge_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
