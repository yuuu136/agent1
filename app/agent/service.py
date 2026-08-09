from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime
import json
import re
from typing import Any
from zoneinfo import ZoneInfo

from app.agent.cards import card_builder
from app.agent.graph import agent_graph
from app.agent.nlu import is_greeting_text, nlu_engine
from app.agent.planner import task_planner
from app.agent.reference import reference_resolver
from app.agent.state import merge_slots, session_store
from app.agent.tools import agent_toolbox
from app.clients.agent_memory import agent_memory_client
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
            memory_id = self._persist_turn(
                request=request,
                user_text=user_text,
                response=response,
                state=state,
                action="greeting",
                intent="greeting",
            )
            if memory_id:
                state.memory_id = memory_id
                response.session["memoryId"] = memory_id
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
        memory_id = self._persist_turn(
            request=request,
            user_text=user_text,
            response=response,
            state=state,
            action=plan.action,
            intent=nlu.intent,
        )
        if memory_id:
            state.memory_id = memory_id
            response.session["memoryId"] = memory_id
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
            memory_id = self._persist_turn(
                request=request,
                user_text=user_text,
                response=response,
                state=state,
                action="greeting",
                intent="greeting",
            )
            if memory_id:
                state.memory_id = memory_id
                response.session["memoryId"] = memory_id
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
            if node_name == "response":
                response = graph_state["response"]
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
                memory_id = self._persist_turn(
                    request=request,
                    user_text=user_text,
                    response=response,
                    state=state,
                    action=plan.action,
                    intent=nlu.intent,
                )
                if memory_id:
                    state.memory_id = memory_id
                    response.session["memoryId"] = memory_id
                    graph_state["state"] = state
                    graph_state["response"] = response
                session_store.save(state)
                self._save_graph_state(graph_config, request, user_text, state)
                node_update = {"response": response}
            yield node_name, node_update

    def extract_nlu(self, request: ChatRequest, state: AgentState | None = None) -> NLUResult:
        return nlu_engine.extract(request, state)

    def _graph_config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def _load_state(self, request: ChatRequest) -> AgentState:
        persistent_memory = self._load_persistent_memory(request)
        if persistent_memory:
            restored = self._state_from_persistent_memory(
                request,
                persistent_memory,
            )
            if restored:
                return restored

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

    def _load_persistent_memory(
        self,
        request: ChatRequest,
    ) -> dict[str, Any] | None:
        return agent_memory_client.current(
            jwt=request.jwt,
            session_id=request.sessionId,
            memory_id=request.memoryId,
        )

    def _state_from_persistent_memory(
        self,
        request: ChatRequest,
        memory: dict[str, Any],
    ) -> AgentState | None:
        raw_state = memory.get("stateJson") or memory.get("state_json")
        state_data: dict[str, Any] | None = None
        if isinstance(raw_state, str) and raw_state.strip():
            try:
                parsed = json.loads(raw_state)
                if isinstance(parsed, dict):
                    state_data = parsed
            except json.JSONDecodeError:
                state_data = None
        elif isinstance(raw_state, dict):
            state_data = raw_state

        try:
            state = AgentState.model_validate(state_data or {})
        except Exception:
            state = session_store.get(request.sessionId, request.userId)

        state.session_id = request.sessionId
        state.user_id = request.userId or state.user_id
        state.memory_id = memory.get("memoryId") or request.memoryId
        messages = memory.get("messages") or []
        if not state.history and isinstance(messages, list):
            state.history = [
                {
                    "user": item.get("content", ""),
                    "intent": item.get("intent", ""),
                    "action": item.get("action", ""),
                    "success": True,
                }
                for item in messages
                if isinstance(item, dict) and item.get("role") == "user"
            ]
        return state

    def _persist_turn(
        self,
        *,
        request: ChatRequest,
        user_text: str,
        response: AgentResponse,
        state: AgentState,
        action: str,
        intent: str,
    ) -> str | None:
        if not request.jwt:
            return None

        request_payload = {
            "sessionId": request.sessionId,
            "memoryId": request.memoryId or state.memory_id,
            "userId": request.userId,
            "type": request.type,
            "text": request.text,
            "event": request.event,
            "payload": request.payload,
        }
        saved = agent_memory_client.save_turn(
            jwt=request.jwt,
            session_id=request.sessionId,
            memory_id=request.memoryId or state.memory_id,
            user_message=user_text,
            assistant_message=response.message,
            event=request.event,
            intent=intent,
            action=action,
            state=response.state,
            state_json=json.dumps(
                state.model_dump(mode="json"),
                ensure_ascii=False,
            ),
            request_json=json.dumps(
                request_payload,
                ensure_ascii=False,
            ),
            response_json=json.dumps(
                response.model_dump(mode="json"),
                ensure_ascii=False,
            ),
        )
        if not saved:
            return None
        memory_id = saved.get("memoryId") or saved.get("memory_id")
        return str(memory_id) if memory_id else None

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
                f"{time_greeting}～ 现在是{current_time}。\n\n"
                "不管你是想随便逛逛看看最近有什么好片，还是心里已经有想看的电影了，"
                "直接跟我说就行，我来帮你搞定。\n\n"
                "你可以告诉我喜欢的类型、想去的影院、方便的时间，"
                "我帮你一步步找到最合适的那一场～"
            )

        state.last_bot_message = message.strip()

        return AgentResponse(
            message=message.strip(),
            state="greeting",
            cards=[],
            suggestions=["附近有什么电影院", "帮我订两张明晚8点后的喜剧片", "查一下退票规则"],
            session={
                "sessionId": state.session_id,
                "memoryId": state.memory_id,
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
                "slots": merge_slots(state.slots, nlu.slots, nlu.intent),
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
        state.last_bot_message = message

        return AgentResponse(
            message=message,
            state=state.state or plan.state,
            cards=cards,
            suggestions=suggestions,
            session={
                "sessionId": state.session_id,
                "memoryId": state.memory_id,
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
                "seatPreference": settings.get("default_seat_preference", "middle"),
            }
            result.message = "已取消当前购票流程。"
            return

        if plan.action == "cancel_order" and result.success:
            # Partial cleanup: release locked seats but keep booking context.
            state.slots.pop("orderId", None)
            state.slots.pop("lockId", None)
            state.slots.pop("seatIds", None)
            state.slots.pop("snackIds", None)
            state.slots.pop("snackItems", None)
            state.slots.pop("snackRequests", None)
            state.selected.pop("order", None)
            state.selected.pop("seat_map", None)
            state.selected.pop("snack_candidates", None)
            state.pending_action = None
            result.data["seatsReleased"] = True
            if plan.params.get("cancelFlow"):
                settings = agent_config.get("agent", {})
                state.selected.clear()
                state.slots = {
                    "city": settings.get("default_city", ""),
                    "seatPreference": settings.get(
                        "default_seat_preference",
                        "middle",
                    ),
                }
                state.state = "idle"
                result.message = "已取消订单并释放锁定的座位。"
                return
            self._resume_after_cancel_order(state, plan, result)
            return

        if plan.action == "confirm_selection" and plan.params.get("skipSnacks"):
            state.slots.pop("snackIds", None)
            state.slots.pop("snackItems", None)
            state.slots.pop("snackRequests", None)
            state.selected.pop("snack_candidates", None)
            state.pending_action = None
            result.data = {"skipped": "snacks"}
            result.message = "好的，已跳过零食。"
            self._resume_payment_after_optional_step(state, result)
            return

        if plan.action == "confirm_selection" and plan.params.get("skipCoupon"):
            state.slots.pop("couponId", None)
            state.selected.pop("coupon_candidates", None)
            state.pending_action = None
            result.data = {"skipped": "coupon"}
            result.message = "好的，已跳过优惠券。"
            self._resume_payment_after_optional_step(state, result)
            return

        if plan.action == "confirm_selection" and plan.params.get("snackIds"):
            snack_ids = plan.params.get("snackIds") or []
            snack_items = plan.params.get("snackItems") or []
            state.slots["snackIds"] = snack_ids
            if snack_items:
                state.slots["snackItems"] = snack_items
            state.selected.pop("snack_candidates", None)
            result.data = {"snackIds": snack_ids, "snackItems": snack_items}
            applied = self._apply_snacks_to_order(
                state,
                result,
                snack_ids,
                snack_items=snack_items,
                jwt=plan.params.get("jwt"),
            )
            if not result.success:
                return
            state.slots.pop("snackRequests", None)
            result.message = "零食已加入订单，可以继续支付。"
            if applied and result.data.get("totalAmount") not in [None, ""]:
                result.message = (
                    f"零食已加入订单，合计{result.data['totalAmount']}元，"
                    "可以继续支付。"
                )
            self._resume_payment_after_optional_step(state, result)
            return

        if plan.action == "search_movies":
            state.selected.pop("movie_candidates", None)
            state.selected.pop("showtime_candidates", None)
            # When search returns results with showtimes AND the user is asking
            # about a specific movie (not browsing), auto-fill date+time so
            # the user sees actual showtimes instead of being asked step by step.
            movies = data.get("movies") or []
            if isinstance(movies, list) and movies:
                want_specific = plan.params.get("movieName") or plan.params.get("keyword")
                if (want_specific or plan.params.get("movieLimit") == 1) and len(movies) == 1:
                    movie = movies[0]
                    showtimes = movie.get("upcomingShowtimes") or []
                    if showtimes:
                        data["directShowtimes"] = True
                        data["showtimes"] = showtimes
                        data["movieId"] = movie.get("movieId")
                        data["movieName"] = movie.get("movieName")
                        dates = sorted({
                            str(st.get("startAt", ""))[:10]
                            for st in showtimes if st.get("startAt")
                        })
                        if dates:
                            data["date"] = dates[0]
                            data["timePreference"] = "any"
                        state.state = "selecting_showtime"
                        state.pending_action = "search_showtimes"
                        result.message = (
                            f"已找到《{movie.get('movieName') or '这部电影'}》，"
                            "下面是可选场次，选择一场后就可以选座。"
                        )
                        result.suggestions = ["换个时间", "附近影院", "换一部电影"]
        if plan.action == "search_showtimes":
            state.selected.pop("showtime_candidates", None)
            state.selected.pop("order", None)
            state.selected.pop("ticket", None)
            state.selected.pop("calendar", None)
            state.selected.pop("notification", None)
        if plan.action == "search_nearby_cinemas":
            state.selected.pop("cinema_candidates", None)
        if plan.action in {
            "search_showtimes",
            "search_nearby_cinemas",
            "get_seats",
        }:
            state.selected.pop("seat_map", None)

        if "movies" in data and data["movies"]:
            state.selected["movie_candidates"] = data["movies"]
        if "showtimes" in data and data["showtimes"]:
            state.selected["showtime_candidates"] = data["showtimes"]
        if "seats" in data:
            state.selected["seat_map"] = data
        if "cinemas" in data:
            state.selected["cinema_candidates"] = data["cinemas"]
            state.slots.pop("changeCinema", None)
        if "snacks" in data:
            state.selected["snack_candidates"] = data["snacks"]
        if plan.action == "search_showtimes":
            state.slots.pop("changeShowtime", None)

        for key in [
            "showtimeId",
            "seatIds",
            "orderId",
            "lockId",
            "snackIds",
            "snackItems",
            "snackRequests",
            "movieId",
            "movieName",
            "cinemaId",
            "cinemaName",
            "hallName",
            "hallType",
            "language",
            "date",
            "time",
            "startAt",
            "endAt",
            "seatPositions",
            "amount",
            "status",
            "expiresAt",
        ]:
            if key in data:
                state.slots[key] = data[key]

        if plan.action == "search_showtimes" and result.success:
            self._auto_lock_explicit_seats(state, plan, result)
            if result.data.get("paymentReady"):
                return

        if (
            plan.action == "get_seats"
            and result.success
            and state.slots.get("seatPositions")
        ):
            self._auto_lock_explicit_seats(state, plan, result)
            if result.data.get("paymentReady"):
                return

        if plan.action == "lock_seats" and result.success and data.get("orderId"):
            state.selected["order"] = data.copy()
            state.slots["orderId"] = data.get("orderId")
            result.data["order"] = data.copy()
            self._offer_snacks_after_lock(
                state,
                result,
                data,
                jwt=plan.params.get("jwt"),
            )

        pay_status = str(data.get("status", "")).upper()
        if plan.action == "pay_order" and result.success and pay_status in {
            "PAID",
            "TICKETED",
        }:
            if pay_status == "TICKETED" or data.get("ticketStatus") == "issued":
                ticket_data = data.copy()
            else:
                issue_plan = AgentPlan(
                    action="issue_ticket",
                    params={"orderId": data.get("orderId")},
                    state="issuing_ticket",
                )
                issue_result = agent_toolbox.execute(issue_plan, state)
                if not issue_result.success:
                    return
                ticket_data = issue_result.data

            state.selected["ticket"] = ticket_data
            result.data.update(ticket_data)
            result.data["ticketStatus"] = ticket_data.get(
                "ticketStatus",
                "issued",
            )

    def _resume_after_cancel_order(
        self,
        state: AgentState,
        plan: AgentPlan,
        result: ToolResult,
    ) -> None:
        """Continue a seat/showtime replacement after the old seat lock is released."""
        action = plan.params.get("resumeAction")
        params = plan.params.get("resumeParams")
        if action not in {"get_seats", "search_showtimes"} or not isinstance(
            params,
            dict,
        ):
            return

        resume_params = dict(params)
        jwt = plan.params.get("jwt")
        if jwt:
            resume_params["jwt"] = jwt

        resume_plan = AgentPlan(
            action=action,
            reason=plan.reason,
            params=resume_params,
            state=(
                "selecting_seats"
                if action == "get_seats"
                else "selecting_showtime"
            ),
        )
        resume_result = agent_toolbox.execute(resume_plan, state)
        self._apply_tool_result(state, resume_plan, resume_result)

        if resume_result.success:
            prefix = "已释放原座位，"
            if action == "get_seats":
                resume_result.message = (
                    f"{prefix}{resume_result.message}"
                    if resume_result.message
                    else "已释放原座位，请重新选择座位。"
                )
            else:
                resume_result.message = (
                    f"{prefix}{resume_result.message}"
                    if resume_result.message
                    else "已释放原座位，以下是可选场次。"
                )
            resume_result.data["seatsReleased"] = True

        # The response must describe the resumed action, otherwise CardBuilder
        # would see cancel_order and omit the returned seat map/showtime cards.
        plan.action = resume_plan.action
        plan.reason = resume_plan.reason
        plan.params = resume_plan.params
        plan.state = resume_plan.state
        result.tool_name = resume_result.tool_name
        result.success = resume_result.success
        result.data = resume_result.data
        result.message = resume_result.message
        result.cards = resume_result.cards
        result.suggestions = resume_result.suggestions

    def _auto_lock_explicit_seats(
        self,
        state: AgentState,
        plan: AgentPlan,
        result: ToolResult,
    ) -> None:
        positions = state.slots.get("seatPositions")
        showtimes = result.data.get("showtimes") or []
        jwt = plan.params.get("jwt")
        if not isinstance(positions, list) or not positions or not jwt:
            return

        if plan.action == "get_seats":
            showtime_id = result.data.get("showtimeId") or state.slots.get("showtimeId")
            if not showtime_id:
                return
            showtime = {
                key: state.slots[key]
                for key in [
                    "showtimeId",
                    "movieId",
                    "movieName",
                    "cinemaId",
                    "cinemaName",
                    "hallName",
                    "hallType",
                    "language",
                    "date",
                    "time",
                    "startAt",
                    "endAt",
                    "price",
                ]
                if key in state.slots and state.slots[key] not in [None, ""]
            }
            showtime.update(
                {
                    key: result.data[key]
                    for key in [
                        "showtimeId",
                        "movieId",
                        "movieName",
                        "cinemaId",
                        "cinemaName",
                        "hallName",
                        "hallType",
                        "language",
                        "date",
                        "time",
                        "startAt",
                        "endAt",
                        "price",
                    ]
                    if key in result.data and result.data[key] not in [None, ""]
                }
            )
            showtime["showtimeId"] = showtime_id
            seat_result = result
        else:
            if len(showtimes) != 1 or not isinstance(showtimes[0], dict):
                return
            showtime = showtimes[0]
            showtime_id = showtime.get("showtimeId")
            if not showtime_id:
                return
            seat_result = agent_toolbox.execute(
                AgentPlan(
                    action="get_seats",
                    params={
                        **state.slots,
                        **showtime,
                        "jwt": jwt,
                        "showtimeId": showtime_id,
                    },
                    state="selecting_seats",
                ),
                state,
            )

        if not showtime_id:
            return

        self._remember_showtime_context(state, showtime)
        seat_map_data = deepcopy(seat_result.data)
        if not seat_result.success:
            result.data.clear()
            result.data.update(seat_map_data)
            result.message = seat_result.message
            state.state = "selecting_seats"
            state.pending_action = "get_seats"
            return

        seats = seat_result.data.get("seats") or []
        selected_ids: list[Any] = []
        missing: list[str] = []
        unavailable: list[str] = []
        for position in positions:
            if not isinstance(position, dict):
                continue
            row_no = self._as_int(position.get("rowNo"))
            seat_no = self._as_int(position.get("seatNo"))
            label = f"{row_no}排{seat_no}座"
            match = next(
                (
                    seat
                    for seat in seats
                    if isinstance(seat, dict)
                    and self._seat_row(seat) == row_no
                    and self._seat_number(seat) == seat_no
                ),
                None,
            )
            if not match or match.get("seatId") in [None, ""]:
                missing.append(label)
                continue
            if str(match.get("status") or "").lower() in {
                "locked",
                "sold",
                "unavailable",
                "couple",
            }:
                unavailable.append(label)
                continue
            selected_ids.append(match["seatId"])

        if missing:
            result.data.clear()
            result.data.update(seat_map_data)
            self._set_seat_retry_message(
                result,
                seats,
                positions,
                prefix=f"座位图中没有找到：{'、'.join(missing)}。",
            )
            state.state = "selecting_seats"
            state.pending_action = "get_seats"
            return
        if unavailable:
            result.data.clear()
            result.data.update(seat_map_data)
            self._set_seat_retry_message(
                result,
                seats,
                positions,
                prefix=f"指定座位已被占用：{'、'.join(unavailable)}。",
            )
            state.state = "selecting_seats"
            state.pending_action = "get_seats"
            return

        lock_result = agent_toolbox.execute(
            AgentPlan(
                action="lock_seats",
                params={
                    **state.slots,
                    **showtime,
                    "jwt": jwt,
                    "showtimeId": showtime_id,
                    "seatIds": selected_ids,
                    "ticketCount": len(selected_ids),
                },
                state="locking_seats",
            ),
            state,
        )
        if not lock_result.success:
            result.data.clear()
            result.data.update(seat_map_data)
            result.message = lock_result.message
            state.state = "selecting_seats"
            state.pending_action = "get_seats"
            return

        state.selected["seat_map"] = seat_map_data
        state.selected["order"] = lock_result.data.copy()
        state.state = "paying"
        state.pending_action = "pay_order"
        state.slots["showtimeId"] = showtime_id
        state.slots["seatIds"] = selected_ids
        state.slots["orderId"] = lock_result.data.get("orderId")
        result.data.clear()
        result.data.update(lock_result.data)
        result.data["seatPositions"] = positions
        result.message = (
            f"已按你的要求选择{'、'.join(self._seat_position_labels(positions))}，"
            "座位已锁定，订单已创建，可以直接支付。"
        )
        if self._apply_requested_snacks_after_lock(
            state,
            result,
            lock_result.data,
            jwt=jwt,
        ):
            result.data["paymentReady"] = True
            return

        if state.slots.get("seatPositions") and not state.slots.get("snackRequests"):
            result.data["paymentReady"] = True
            return

        offered = self._offer_snacks_after_lock(
            state,
            result,
            lock_result.data,
            jwt=jwt,
        )
        if not offered:
            result.data["paymentReady"] = True

    def _remember_showtime_context(
        self,
        state: AgentState,
        showtime: dict[str, Any],
    ) -> None:
        for key in [
            "showtimeId",
            "movieId",
            "movieName",
            "cinemaId",
            "cinemaName",
            "hallName",
            "hallType",
            "language",
            "date",
            "time",
            "startAt",
            "endAt",
            "price",
        ]:
            value = showtime.get(key)
            if value not in [None, ""]:
                state.slots[key] = value

    def _as_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _seat_row(self, seat: dict[str, Any]) -> int | None:
        return self._as_int(seat.get("row") or seat.get("rowNo"))

    def _seat_number(self, seat: dict[str, Any]) -> int | None:
        return self._as_int(seat.get("number") or seat.get("seatNo"))

    def _is_selectable_seat(self, seat: dict[str, Any]) -> bool:
        if seat.get("seatId") in [None, ""]:
            return False
        status = str(seat.get("status") or "available").lower()
        return status not in {"locked", "sold", "unavailable", "couple"}

    def _set_seat_retry_message(
        self,
        result: ToolResult,
        seats: list[Any],
        positions: list[Any],
        prefix: str,
    ) -> None:
        suggestions = self._suggest_available_seats(seats, positions)
        if suggestions:
            result.message = f"{prefix} 可选：{'、'.join(suggestions)}。"
            result.suggestions = suggestions
        else:
            result.message = f"{prefix} 请在座位图里重新选择座位。"
            result.suggestions = ["选中间", "选后排", "换一场"]

    def _suggest_available_seats(
        self,
        seats: list[Any],
        positions: list[Any],
        limit: int = 4,
    ) -> list[str]:
        requested = [
            (self._as_int(position.get("rowNo")), self._as_int(position.get("seatNo")))
            for position in positions
            if isinstance(position, dict)
        ]
        requested = [
            (row_no, seat_no)
            for row_no, seat_no in requested
            if row_no is not None and seat_no is not None
        ]
        available: list[tuple[int, int]] = []
        for seat in seats:
            if not isinstance(seat, dict) or not self._is_selectable_seat(seat):
                continue
            row_no = self._seat_row(seat)
            seat_no = self._seat_number(seat)
            if row_no is None or seat_no is None:
                continue
            available.append((row_no, seat_no))
        if not available:
            return []

        def score(candidate: tuple[int, int]) -> tuple[int, int, int, int]:
            row_no, seat_no = candidate
            if not requested:
                return (0, row_no, seat_no, 0)
            distances = []
            for requested_row, requested_seat in requested:
                same_row_penalty = 0 if row_no == requested_row else 1
                same_number_penalty = 0 if seat_no == requested_seat else 1
                distance = abs(row_no - requested_row) + abs(seat_no - requested_seat)
                distances.append((same_row_penalty + same_number_penalty, distance))
            best_penalty, best_distance = min(distances)
            return (best_penalty, best_distance, row_no, seat_no)

        labels: list[str] = []
        seen: set[str] = set()
        for row_no, seat_no in sorted(set(available), key=score):
            label = f"{row_no}排{seat_no}座"
            if label in seen:
                continue
            labels.append(label)
            seen.add(label)
            if len(labels) >= limit:
                break
        return labels

    def _seat_position_labels(self, positions: list[Any]) -> list[str]:
        labels = []
        for position in positions:
            if isinstance(position, dict):
                labels.append(
                    f"{position.get('rowNo')}排{position.get('seatNo')}座"
                )
        return labels

    def _offer_snacks_after_lock(
        self,
        state: AgentState,
        result: ToolResult,
        order_data: dict[str, Any],
        jwt: str | None = None,
    ) -> bool:
        if state.slots.get("snackIds") or result.data.get("snacks"):
            return False

        snack_result = agent_toolbox.execute(
            AgentPlan(
                action="recommend_snacks",
                params={
                    "orderId": order_data.get("orderId") or state.slots.get("orderId"),
                    "ticketCount": state.slots.get("ticketCount"),
                    "cinemaId": order_data.get("cinemaId") or state.slots.get("cinemaId"),
                    "cinemaName": order_data.get("cinemaName") or state.slots.get("cinemaName"),
                    "jwt": jwt,
                },
                state="selecting_snacks",
            ),
            state,
        )
        snacks = snack_result.data.get("snacks") if snack_result.success else None
        if not snacks:
            return False

        state.selected["snack_candidates"] = snacks
        state.state = "selecting_snacks"
        state.pending_action = "recommend_snacks"
        result.data["snacks"] = snacks
        result.data["showSnackRecommendations"] = True
        result.data["order"] = order_data.copy()
        result.data.pop("paymentReady", None)
        result.message = (
            "座位已锁定，订单已创建。这里有几款可选零食套餐，"
            "可以加入套餐，也可以说“不要零食”直接去支付。"
        )
        result.suggestions = ["不要零食", "直接支付"]
        return True

    def _apply_requested_snacks_after_lock(
        self,
        state: AgentState,
        result: ToolResult,
        order_data: dict[str, Any],
        jwt: str | None = None,
    ) -> bool:
        requests = state.slots.get("snackRequests")
        if not isinstance(requests, list) or not requests or not jwt:
            return False

        order_id = order_data.get("orderId") or state.slots.get("orderId")
        if not order_id:
            return False

        snack_result = agent_toolbox.execute(
            AgentPlan(
                action="recommend_snacks",
                params={
                    "orderId": order_id,
                    "ticketCount": state.slots.get("ticketCount"),
                    "cinemaId": order_data.get("cinemaId") or state.slots.get("cinemaId"),
                    "cinemaName": order_data.get("cinemaName") or state.slots.get("cinemaName"),
                    "jwt": jwt,
                },
                state="selecting_snacks",
            ),
            state,
        )
        snacks = snack_result.data.get("snacks") if snack_result.success else None
        if not snacks:
            return False

        snack_items: list[dict[str, int]] = []
        selected_labels: list[str] = []
        for request in requests:
            if not isinstance(request, dict):
                continue
            requested_name = str(request.get("name") or "").strip()
            match = self._match_snack_option(snacks, requested_name)
            if not match or match.get("snackId") in [None, ""]:
                continue
            quantity = self._as_int(request.get("quantity")) or 1
            if quantity <= 0:
                quantity = 1
            snack_id = match["snackId"]
            snack_items.append({"snackId": snack_id, "quantity": quantity})
            unit = str(request.get("unit") or "份")
            selected_labels.append(f"{quantity}{unit}{match.get('name') or requested_name}")

        if not snack_items:
            return False

        state.slots["snackIds"] = [item["snackId"] for item in snack_items]
        state.slots["snackItems"] = snack_items
        applied = self._apply_snacks_to_order(
            state,
            result,
            state.slots["snackIds"],
            snack_items=snack_items,
            jwt=jwt,
        )
        if not applied:
            return False

        state.slots.pop("snackRequests", None)
        state.state = "paying"
        state.pending_action = "pay_order"
        result.data["order"] = state.selected.get("order", order_data.copy())
        amount = result.data.get("totalAmount") or result.data.get("amount")
        amount_text = f"，合计{amount}元" if amount not in [None, ""] else ""
        result.message = (
            "座位已锁定，订单已创建，"
            f"已按你的要求加入{'、'.join(selected_labels)}{amount_text}，可以直接支付。"
        )
        result.suggestions = ["确认支付", "查看订单", "取消支付"]
        return True

    def _match_snack_option(
        self,
        snacks: list[Any],
        requested_name: str,
    ) -> dict[str, Any] | None:
        normalized_request = self._normalize_match_text(requested_name)
        if not normalized_request:
            return None
        for snack in snacks:
            if not isinstance(snack, dict):
                continue
            snack_name = self._normalize_match_text(snack.get("name"))
            if not snack_name:
                continue
            if normalized_request in snack_name or snack_name in normalized_request:
                return snack
        return None

    def _normalize_match_text(self, value: Any) -> str:
        return re.sub(r"[\s，。,.!?！？、:：;；]+", "", str(value or "")).casefold()

    def _resume_payment_after_optional_step(
        self,
        state: AgentState,
        result: ToolResult,
    ) -> bool:
        order = state.selected.get("order")
        if not isinstance(order, dict):
            order_id = state.slots.get("orderId")
            if not order_id:
                return False
            order = {"orderId": order_id}

        merged_data = {
            **order,
            **result.data,
            "order": order,
            "paymentReady": True,
        }
        result.data = merged_data
        state.state = "paying"
        state.pending_action = "pay_order"
        if result.message:
            result.message = f"{result.message} 现在可以支付。"
        else:
            result.message = "现在可以支付。"
        result.suggestions = ["确认支付", "查看订单", "取消支付"]
        return True

    def _apply_snacks_to_order(
        self,
        state: AgentState,
        result: ToolResult,
        snack_ids: list[Any],
        snack_items: list[Any] | None = None,
        jwt: str | None = None,
    ) -> bool:
        order_id = state.slots.get("orderId")
        if not order_id or not jwt:
            return False

        params = {
            "orderId": order_id,
            "snackIds": snack_ids,
            "jwt": jwt,
        }
        if snack_items:
            params["snackItems"] = snack_items

        snack_result = agent_toolbox.execute(
            AgentPlan(
                action="replace_order_snacks",
                params=params,
                state="selecting_snacks",
            ),
            state,
        )
        if not snack_result.success:
            result.success = False
            result.data = snack_result.data
            result.message = snack_result.message or "零食加入订单失败，请重新选择。"
            return False

        result.data.update(snack_result.data)
        order = state.selected.get("order")
        if isinstance(order, dict):
            order.update(
                {
                    key: value
                    for key, value in snack_result.data.items()
                    if value not in [None, ""]
                }
            )
            state.selected["order"] = order
        else:
            state.selected["order"] = snack_result.data.copy()

        for key in [
            "ticketAmount",
            "snackAmount",
            "totalAmount",
            "amount",
            "selectedSnacks",
        ]:
            if key in snack_result.data:
                state.slots[key] = snack_result.data[key]
        return True

    def _event_text(self, request: ChatRequest) -> str:
        if request.event:
            return f"event:{request.event}"
        return ""

    def _should_greet(self, request: ChatRequest, state: AgentState, user_text: str) -> bool:
        if request.event:
            return False
        # draftId is request context added by the frontend; it should not
        # turn a plain greeting into a booking step.
        payload = request.payload or {}
        if any(
            key not in {"draftId", "draft_id"}
            and value not in (None, "", [], {})
            for key, value in payload.items()
        ):
            return False
        return not user_text.strip() or is_greeting_text(user_text)

    def _ask_message(self, action: str) -> str:
        messages = {
            "ask_movie_or_genre": "想看哪部电影，或者想看什么类型？",
            "ask_time": "想看什么时候的？比如今晚、明天下午，或者周末～",
            "ask_ticket_count": "好的，想买几张票呢？",
            "smalltalk": (
            "嗨～ 有什么想看的吗？\n\n"
            "你可以告诉我喜欢的类型，比如喜剧、动作、科幻，我帮你看看最近有什么好片。"
            "如果心里已经有想看的电影了，直接说片名就行，我来帮你查场次和座位。"
            "还没想好的话也没关系，带你逛逛最近热映的也不错～"
        ),
        }
        return messages.get(action, "还需要确认一下信息。")

    def _default_message(self, plan: AgentPlan, result: ToolResult) -> str:
        if not result.success:
            return result.message or "这个步骤暂时没有完成。"
        if plan.action == "confirm_selection" and "支付" in plan.reason:
            return "当前没有可支付订单，请先选择场次和座位。"
        if plan.action == "confirm_selection" and "零食" in plan.reason:
            return "零食已加入选择，可以继续选座或确认订单。"
        if plan.action == "pay_order":
            if result.data.get("ticketStatus") == "issued" or str(result.data.get("status", "")).upper() == "TICKETED":
                return "支付成功，电子票已出票。"
            if result.data.get("qrCode"):
                return "支付二维码已生成，请扫码支付。"
            if str(result.data.get("paymentStatus", "")).upper() == "SUCCESS":
                return "支付成功。"
        return {
            "confirm_selection": "请确认当前选择。",
            "lock_seats": "座位已锁定，可以创建订单。",
            "create_order": "订单已创建。",
            "pay_order": "支付二维码已生成，请扫码支付。",
            "cancel": "已取消当前购票流程。",
        }.get(plan.action, "已完成当前步骤。")

    def _suggestions(self, plan: AgentPlan, state: AgentState) -> list[str]:
        suggestions: dict[str, list[str]] = {
            "ask_movie_or_genre": ["喜剧片", "动作片", "最近热映"],
            "ask_time": ["今晚", "明晚8点后", "周末下午"],
            "ask_ticket_count": ["1张", "2张", "3张"],
            "search_showtimes": ["换便宜点", "晚一点", "附近影院"],
            "get_seats": ["选中间", "选后排", "换一场"],
            "recommend_snacks": ["双人套餐", "不要零食", "直接支付"],
            "lock_seats": ["确认订单", "重新选座", "加零食"],
            "create_order": ["确认支付", "查看订单", "取消"],
        }
        return suggestions.get(plan.action, ["帮我订票", "查退票规则", "附近影院"])


agent_service = AgentService()
