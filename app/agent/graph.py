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
    "ask",
    "smalltalk",
}


def nlu_node(data: AgentGraphState) -> dict[str, Any]:
    return {"nlu": nlu_engine.extract(data["request"], data.get("state"))}


def merge_context_node(data: AgentGraphState) -> dict[str, Any]:
    state = data["state"]
    nlu = data["nlu"]
    return {
        "state": state.model_copy(
            update={
                "intent": nlu.intent,
                "slots": merge_slots(state.slots, nlu.slots, nlu.intent),
            },
            deep=True,
        )
    }


def preserve_context_for_reference_node(
    data: AgentGraphState,
) -> dict[str, Any]:
    """Keep the persisted state intact until reference resolution finishes."""
    return {"state": data["state"]}


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
        "cancel_order",
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
    plan = data["plan"]
    action = plan.action

    # Generate a contextual ask message
    message = _build_ask_message(state, plan)

    # Context-aware smalltalk: user says "好的"/"行" while waiting for input
    if action == "smalltalk" and state.pending_action:
        ack_messages = {
            "ask_time": "那我们就看最近的场次吧？或者你说个偏好的时间～",
            "ask_ticket_count": "还需要确认票数，想买几张呢？",
            "get_seats": "想坐哪个位置？中间、前排还是后排？",
        }
        message = ack_messages.get(state.pending_action, message)

    return {
        "result": ToolResult(
            tool_name=action,
            message=message,
        )
    }


def _build_ask_message(state: "AgentState", plan: "AgentPlan") -> str:
    """Build a personalized ask message based on what's known and missing."""
    params = plan.params or {}
    if params.get("message"):
        return str(params["message"])
    missing = params.get("missing", [])
    filled = params.get("filled", {})
    next_ask = params.get("next_ask", "")

    # Defaults with personalisation
    movie_name = filled.get("movieName", "")
    ticket_count = filled.get("ticketCount", "")
    cinema_name = filled.get("cinemaName", "")

    ask_templates = {
        "movieName": lambda: (
            f"好的～想看哪部电影呢？"
            if not filled else
            f"想看哪部电影呢？"
        ),
        "genre": lambda: "想看什么类型的？喜剧、动作、科幻、动画都可以～",
        "ticketCount": lambda: (
            f"{movie_name}很不错！想买几张票呢？"
            if movie_name else
            "想买几张票呢？"
        ),
        "date": lambda: (
            f"《{movie_name}》～想看哪天的？比如今天、明天或者周末"
            if movie_name else
            "想看哪天的场次？今天、明天还是周末？"
        ),
        "timeRange": lambda: (
            f"《{movie_name}》～想看什么时间段的？比如上午、下午、晚上"
            if movie_name else
            "想看什么时间段的？上午、下午还是晚上？"
        ),
        "cinemaName": lambda: "有想去的影院吗？或者我帮你查查附近有哪些～",
        "seatPreference": lambda: "想坐什么位置？中间、前排、后排，或者普通座都可以～",
        "showtimeId": lambda: "我帮你查了场次，选一个合适的吧～",
        "seatIds": lambda: "座位图准备好了，选几个座位吧～",
    }

    if next_ask and next_ask in ask_templates:
        return ask_templates[next_ask]()

    # Generic fallback
    if missing:
        return f"还需要确认一下信息～"

    return "嗨～ 有什么想看的吗？我可以帮你查电影、选场次、选座。"



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
    graph.add_node(
        "merge_context_initial",
        preserve_context_for_reference_node,
    )
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
