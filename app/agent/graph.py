from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from app.agent.cards import card_builder
from app.agent.nlu import nlu_engine
from app.agent.planner import task_planner
from app.agent.reference import reference_resolver
from app.agent.state import merge_slots
from app.agent.tools import agent_toolbox
from app.schemas.agent import (
    AgentPlan,
    AgentResponse,
    AgentState,
    ChatRequest,
    NLUResult,
    ToolResult,
)


class AgentGraphState(TypedDict, total=False):
    request: ChatRequest
    user_text: str
    state: AgentState
    nlu: NLUResult
    plan: AgentPlan
    result: ToolResult
    response: AgentResponse


ASK_ACTIONS = {
    "ask_movie_or_genre",
    "ask_time",
    "ask_ticket_count",
    "smalltalk",
}


def nlu_node(data: AgentGraphState) -> dict[str, Any]:
    return {"nlu": nlu_engine.extract(data["request"])}


def merge_context_node(data: AgentGraphState) -> dict[str, Any]:
    state = data["state"]
    nlu = data["nlu"]
    return {
        "state": state.model_copy(
            update={
                "intent": nlu.intent,
                "slots": merge_slots(state.slots, nlu.slots),
            },
            deep=True,
        )
    }


def reference_node(data: AgentGraphState) -> dict[str, Any]:
    return {
        "nlu": reference_resolver.resolve(data["state"], data["nlu"]),
    }


def planner_node(data: AgentGraphState) -> dict[str, Any]:
    plan = task_planner.plan(data["state"], data["nlu"])
    jwt = data["request"].jwt
    if plan.action in {
        "search_nearby_cinemas",
        "search_movies",
        "search_showtimes",
        "get_seats",
        "recommend_snacks",
        "confirm_selection",
        "lock_seats",
        "create_order",
        "pay_order",
        "issue_ticket",
        "refund_order",
        "get_refund_status",
        "get_order",
        "list_orders",
    } and jwt:
        plan.params["jwt"] = jwt
    return {
        "plan": plan,
    }


def route_after_planner(data: AgentGraphState) -> str:
    return "ask" if data["plan"].action in ASK_ACTIONS else "tool"


def ask_node(data: AgentGraphState) -> dict[str, Any]:
    state = data["state"]
    messages = {
        "ask_movie_or_genre": "想看哪部电影，或者想看什么类型？",
        "ask_time": "想看什么时候的场次？",
        "ask_ticket_count": "需要买几张票？",
        "smalltalk": (
            "嗨～ 有什么想看的吗？\n\n"
            "你可以告诉我喜欢的类型，比如喜剧、动作、科幻，我帮你看看最近有什么好片。"
            "如果心里已经有想看的电影了，直接说片名就行，我来帮你查场次和座位。"
            "还没想好的话也没关系，带你逛逛最近热映的也不错～"
        ),
    }
    action = data["plan"].action
    message = messages.get(action, "需要再确认一下信息。")
    if action == "ask_movie_or_genre" and state.slots.get("cinemaName"):
        message = f"已选择{state.slots['cinemaName']}，想看哪部电影，或者想看什么类型？"
    return {
        "result": ToolResult(
            tool_name=action,
            message=message,
        )
    }


def tool_node(data: AgentGraphState) -> dict[str, Any]:
    return {
        "result": agent_toolbox.execute(data["plan"], data["state"]),
    }


def apply_result_node(data: AgentGraphState) -> dict[str, Any]:
    # Keep the existing side-effect logic while the graph is being introduced.
    from app.agent.service import agent_service

    state = data["state"]
    result = data["result"]
    agent_service._apply_tool_result(state, data["plan"], result)
    return {
        "state": state,
        "result": result,
    }


def response_node(data: AgentGraphState) -> dict[str, Any]:
    from app.agent.service import agent_service

    return {
        "response": agent_service.build_response(
            data["plan"],
            data["result"],
            data["state"],
        )
    }


checkpointer = MemorySaver(
    serde=JsonPlusSerializer(
        allowed_msgpack_modules=[
            ChatRequest,
            AgentState,
            NLUResult,
            AgentPlan,
            ToolResult,
            AgentResponse,
        ],
    )
)


def build_agent_graph():
    graph = StateGraph(AgentGraphState)
    graph.add_node("nlu", nlu_node)
    graph.add_node("merge_context_initial", merge_context_node)
    graph.add_node("reference", reference_node)
    graph.add_node("merge_context_after_reference", merge_context_node)
    graph.add_node("planner", planner_node)
    graph.add_node("ask", ask_node)
    graph.add_node("tool", tool_node)
    graph.add_node("apply_result", apply_result_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "nlu")
    graph.add_edge("nlu", "merge_context_initial")
    graph.add_edge("merge_context_initial", "reference")
    graph.add_edge("reference", "merge_context_after_reference")
    graph.add_edge("merge_context_after_reference", "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "ask": "ask",
            "tool": "tool",
        },
    )
    graph.add_edge("ask", "apply_result")
    graph.add_edge("tool", "apply_result")
    graph.add_edge("apply_result", "response")
    graph.add_edge("response", END)
    return graph.compile(checkpointer=checkpointer)


agent_graph = build_agent_graph()
