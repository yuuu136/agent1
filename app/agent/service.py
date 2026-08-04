from collections.abc import Iterator
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.agent.cards import card_builder
from app.agent.graph import agent_graph
from app.agent.nlu import nlu_engine
from app.agent.planner import task_planner
from app.agent.reference import reference_resolver
from app.agent.state import merge_slots, session_store
from app.agent.tools import agent_toolbox
from app.prompts import prompt_manager
from app.schemas.agent import AgentPlan, AgentResponse, AgentState, ChatRequest, NLUResult, ToolResult
from app.utils.config_handler import agent_config


class AgentService:
    def chat(self, request: ChatRequest) -> AgentResponse:
        graph_config = self._graph_config(request.sessionId)
        state = self._load_state(request)
        user_text = request.text or self._event_text(request)
        if self._should_greet(request, state, user_text):
            response = self.build_greeting_response(state)
            state.history.append(
                {
                    "user": user_text,
                    "intent": "greeting",
                    "action": "greeting",
                    "success": True,
                }
            )
            session_store.save(state)
            self._save_graph_state(graph_config, request, user_text, state)
            return response

        state = state.model_copy(
            update={"last_user_text": user_text},
            deep=True,
        )
        graph_result = agent_graph.invoke(
            {
                "request": request,
                "user_text": user_text,
                "state": state,
            },
            config=graph_config,
        )
        state = graph_result["state"]
        nlu = graph_result["nlu"]
        plan = graph_result["plan"]
        result = graph_result["result"]
        response = graph_result["response"]
        state.history.append(
            {
                "user": user_text,
                "intent": nlu.intent,
                "action": plan.action,
                "success": result.success,
            }
        )
        session_store.save(state)
        self._save_graph_state(graph_config, request, user_text, state)
        return response

    def stream_chat(
        self,
        request: ChatRequest,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Stream LangGraph node updates while keeping the same session behavior."""
        graph_config = self._graph_config(request.sessionId)
        state = self._load_state(request)
        user_text = request.text or self._event_text(request)

        if self._should_greet(request, state, user_text):
            response = self.build_greeting_response(state)
            state.history.append(
                {
                    "user": user_text,
                    "intent": "greeting",
                    "action": "greeting",
                    "success": True,
                }
            )
            session_store.save(state)
            self._save_graph_state(graph_config, request, user_text, state)
            yield "greeting", {"response": response}
            return

        state = state.model_copy(
            update={"last_user_text": user_text},
            deep=True,
        )
        graph_state: dict[str, Any] = {
            "request": request,
            "user_text": user_text,
            "state": state,
        }

        for update in agent_graph.stream(
            graph_state,
            config=graph_config,
            stream_mode="updates",
        ):
            node_name, node_update = next(iter(update.items()))
            graph_state.update(node_update)
            yield node_name, node_update

        state = graph_state["state"]
        nlu = graph_state["nlu"]
        plan = graph_state["plan"]
        result = graph_state["result"]
        state.history.append(
            {
                "user": user_text,
                "intent": nlu.intent,
                "action": plan.action,
                "success": result.success,
            }
        )
        session_store.save(state)
        self._save_graph_state(graph_config, request, user_text, state)

    def extract_nlu(self, request: ChatRequest) -> NLUResult:
        return nlu_engine.extract(request)

    def _graph_config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def _load_state(self, request: ChatRequest) -> AgentState:
        graph_config = self._graph_config(request.sessionId)
        snapshot = agent_graph.get_state(graph_config)
        checkpoint_state = snapshot.values.get("state") if snapshot.values else None
        if isinstance(checkpoint_state, AgentState):
            if request.userId and not checkpoint_state.user_id:
                checkpoint_state = checkpoint_state.model_copy(
                    update={"user_id": request.userId},
                    deep=True,
                )
            return checkpoint_state.model_copy(deep=True)
        return session_store.get(request.sessionId, request.userId)

    def _save_graph_state(
        self,
        graph_config: dict[str, Any],
        request: ChatRequest,
        user_text: str,
        state: AgentState,
    ) -> None:
        agent_graph.update_state(
            graph_config,
            {
                "request": request,
                "user_text": user_text,
                "state": state,
            },
        )

    def build_greeting_response(self, state: AgentState) -> AgentResponse:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        time_greeting = self._time_greeting(now.hour)
        current_time = now.strftime("%Y年%m月%d日 %H:%M")
        try:
            message = prompt_manager.render(
                "greeting",
                {
                    "current_time": current_time,
                    "time_greeting": time_greeting,
                    "session_is_new": str(not state.history).lower(),
                },
            )
        except Exception:
            message = (
                f"{time_greeting}，你好，我是电影票智能体。我可以帮你查附近影院、订电影票、选座，"
                "也可以回答退改签规则。"
            )

        return AgentResponse(
            message=message.strip(),
            state="greeting",
            cards=[],
            suggestions=["附近有什么电影院", "帮我订两张明晚8点后的喜剧片", "查一下退票规则"],
            session={
                "sessionId": state.session_id,
                "intent": "greeting",
                "slots": state.slots,
                "pendingAction": None,
            },
        )

    def _time_greeting(self, hour: int) -> str:
        if 5 <= hour < 12:
            return "早上好"
        if 12 <= hour < 14:
            return "中午好"
        if 14 <= hour < 18:
            return "下午好"
        if 18 <= hour < 24:
            return "晚上好"
        return "夜深了"

    def merge_context(self, state: AgentState, nlu: NLUResult) -> AgentState:
        return state.model_copy(
            update={
                "intent": nlu.intent,
                "slots": merge_slots(state.slots, nlu.slots),
            },
            deep=True,
        )

    def resolve_reference(self, state: AgentState, nlu: NLUResult) -> NLUResult:
        return reference_resolver.resolve(state, nlu)

    def plan_next_action(self, state: AgentState, nlu: NLUResult) -> AgentPlan:
        return task_planner.plan(state, nlu)

    def execute_plan(self, plan: AgentPlan, state: AgentState) -> ToolResult:
        if plan.action in {"ask_movie_or_genre", "ask_time", "ask_ticket_count", "smalltalk"}:
            return ToolResult(tool_name=plan.action, message=self._ask_message(plan.action))
        return agent_toolbox.execute(plan, state)

    def build_response(
        self,
        plan: AgentPlan,
        result: ToolResult,
        state: AgentState,
    ) -> AgentResponse:
        cards = card_builder.build(plan, result)
        message = result.message or self._default_message(plan, result)
        suggestions = result.suggestions or self._suggestions(plan, state)

        return AgentResponse(
            message=message,
            state=plan.state,
            cards=cards,
            suggestions=suggestions,
            session={
                "sessionId": state.session_id,
                "intent": state.intent,
                "slots": state.slots,
                "pendingAction": state.pending_action,
            },
        )

    def _apply_tool_result(self, state: AgentState, plan: AgentPlan, result: ToolResult) -> None:
        if plan.action == "answer_price":
            return

        state.state = plan.state
        state.pending_action = plan.action
        data = result.data

        if plan.action == "cancel":
            settings = agent_config.get("agent", {})
            state.selected.clear()
            state.pending_action = None
            state.slots = {
                "city": settings.get("default_city", ""),
                "ticketCount": settings.get("default_ticket_count", 2),
                "seatPreference": settings.get("default_seat_preference", "middle"),
            }
            result.message = "已取消当前购票流程。"
            return

        if "showtimes" in data and data["showtimes"]:
            state.selected["showtime_candidates"] = data["showtimes"]
        if "seats" in data:
            state.selected["seat_map"] = data
        if "cinemas" in data:
            state.selected["cinema_candidates"] = data["cinemas"]
            state.slots.pop("changeCinema", None)
        if "snacks" in data:
            state.selected["snack_candidates"] = data["snacks"]
        if "coupons" in data:
            state.selected["coupon_candidates"] = data["coupons"]

        for key in ["showtimeId", "seatIds", "orderId", "lockId", "couponId", "snackIds"]:
            if key in data:
                state.slots[key] = data[key]

        if plan.action == "lock_seats" and result.success:
            order_plan = AgentPlan(
                action="create_order",
                params={
                    **state.slots,
                    "lockId": data.get("lockId"),
                    "showtimeId": data.get("showtimeId"),
                    "seatIds": data.get("seatIds", []),
                },
                state="creating_order",
            )
            order_result = agent_toolbox.execute(order_plan, state)
            if order_result.success:
                state.selected["order"] = order_result.data
                state.slots["orderId"] = order_result.data.get("orderId")
                result.data["order"] = order_result.data
                result.data["orderId"] = order_result.data.get("orderId")
                result.message = f"{result.message} {order_result.message}".strip()

        if plan.action == "pay_order" and str(data.get("status", "")).upper() in {
            "PAID",
            "paid",
        }:
            issue_plan = AgentPlan(
                action="issue_ticket",
                params={"orderId": data.get("orderId")},
                state="issuing_ticket",
            )
            issue_result = agent_toolbox.execute(issue_plan, state)
            if issue_result.success:
                state.selected["ticket"] = issue_result.data
                result.data.update(issue_result.data)
                result.data["ticketStatus"] = issue_result.data.get(
                    "ticketStatus",
                    "issued",
                )
                result.data.update(
                    self._after_issue_ticket(state, issue_result.data)
                )

    def _after_issue_ticket(
        self,
        state: AgentState,
        ticket_data: dict[str, Any],
    ) -> dict[str, Any]:
        calendar_plan = AgentPlan(
            action="create_calendar_event",
            params={"ticket": ticket_data, "slots": state.slots},
            state="calendar_created",
        )
        message_plan = AgentPlan(
            action="send_ticket_message",
            params={
                "ticket": ticket_data,
                "userId": state.user_id,
                "phone": state.slots.get("phone") or state.slots.get("phoneNumber"),
                "slots": state.slots,
            },
            state="ticket_notified",
        )
        calendar_result = agent_toolbox.execute(calendar_plan, state)
        notification_result = agent_toolbox.execute(message_plan, state)
        state.selected["calendar"] = calendar_result.data
        state.selected["notification"] = notification_result.data
        return {
            "calendar": calendar_result.data,
            "notification": notification_result.data,
        }

    def _event_text(self, request: ChatRequest) -> str:
        if request.event:
            return f"event:{request.event}"
        return ""

    def _should_greet(self, request: ChatRequest, state: AgentState, user_text: str) -> bool:
        if request.event:
            return False
        if request.payload:
            return False
        normalized = user_text.strip().lower()
        return not normalized or normalized in {"hi", "hello", "你好", "您好", "开始", "start"}

    def _ask_message(self, action: str) -> str:
        messages = {
            "ask_movie_or_genre": "想看哪部电影，或者想看什么类型？",
            "ask_time": "想看什么时候的场次？",
            "ask_ticket_count": "需要买几张票？",
            "smalltalk": "我可以帮你查电影、选场次、选座、搭配零食和优惠券。",
        }
        return messages.get(action, "需要再确认一下信息。")

    def _default_message(self, plan: AgentPlan, result: ToolResult) -> str:
        if not result.success:
            return result.message or "这个步骤暂时没有完成。"
        if plan.action == "confirm_selection" and "支付" in plan.reason:
            return "当前没有可支付订单，请先选择场次和座位。"
        if plan.action == "confirm_selection" and "零食" in plan.reason:
            return "零食已加入选择，可以继续选座或确认订单。"
        if plan.action == "confirm_selection" and "优惠券" in plan.reason:
            return "优惠券已加入选择，可以继续选座或确认订单。"
        return {
            "confirm_selection": "请确认当前选择。",
            "lock_seats": "座位已锁定，可以创建订单。",
            "create_order": "订单已创建。",
            "pay_order": "支付已完成。",
            "cancel": "已取消当前购票流程。",
        }.get(plan.action, "已完成当前步骤。")

    def _suggestions(self, plan: AgentPlan, state: AgentState) -> list[str]:
        suggestions: dict[str, list[str]] = {
            "ask_movie_or_genre": ["喜剧片", "动作片", "最近热映"],
            "ask_time": ["今晚", "明晚8点后", "周末下午"],
            "ask_ticket_count": ["1张", "2张", "3张"],
            "search_showtimes": ["换便宜点", "晚一点", "附近影院"],
            "get_seats": ["选中间", "选后排", "换一场"],
            "recommend_snacks": ["双人套餐", "不要零食", "看看优惠券"],
            "recommend_coupons": ["使用最优惠", "不用券", "继续下单"],
            "lock_seats": ["确认订单", "重新选座", "加零食"],
            "create_order": ["确认支付", "查看订单", "取消"],
        }
        return suggestions.get(plan.action, ["帮我订票", "查退票规则", "附近影院"])


agent_service = AgentService()
