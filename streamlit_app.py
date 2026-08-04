import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from streamlit_js_eval import get_geolocation


PROJECT_ROOT = Path(__file__).resolve().parent
OUTBOX_PATH = PROJECT_ROOT / "data" / "notifications" / "sms_outbox.json"

CARD_TYPE_MAP = {
    "MOVIE_LIST": "movie",
    "CINEMA_LIST": "cinema",
    "SHOWTIME_LIST": "showtime",
    "SEAT_MAP": "seat_map",
    "ORDER_CONFIRM": "confirm_order",
    "PAYMENT": "payment",
    "TICKET": "ticket",
    "ALTERNATIVE": "alternative",
    "LOCATION_PICKER": "location_picker",
    "SNACK_LIST": "snack",
    "COUPON_LIST": "coupon",
}


st.set_page_config(page_title="电影票智能体", layout="wide")


def init_state() -> None:
    defaults = {
        "session_id": f"web-{uuid.uuid4().hex[:8]}",
        "user_id": "demo-user",
        "phone": "",
        "location": "",
        "agent_messages": [],
        "agent_active_cards": [],
        "agent_pending_action": None,
        "selected_movie": None,
        "selected_cinema": None,
        "selected_showtime": None,
        "selected_seats": [],
        "current_order": None,
        "ticket": None,
        "mode": "AI 购票",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def api_base() -> str:
    return str(st.session_state.get("api_base", "http://127.0.0.1:8001")).rstrip("/")


def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{api_base()}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"请求失败：{exc}")
        return None


def api_post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        response = httpx.post(f"{api_base()}{path}", json=payload or {}, timeout=30)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", {})
        st.error(detail.get("message") or f"请求失败：{exc}")
        return None
    except httpx.HTTPError as exc:
        st.error(f"请求失败：{exc}")
        return None


def response_data(payload: dict[str, Any] | None) -> Any:
    if not payload:
        return None
    return payload.get("data", payload)


def stream_sse(
    path: str,
    payload: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    try:
        with httpx.stream(
            "POST",
            f"{api_base()}{path}",
            json=payload,
            timeout=None,
        ) as response:
            response.raise_for_status()
            event_name = ""
            data_lines: list[str] = []

            for line in response.iter_lines():
                if not line:
                    if event_name and data_lines:
                        yield event_name, parse_sse_data(data_lines)
                    event_name = ""
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())

            if event_name and data_lines:
                yield event_name, parse_sse_data(data_lines)
    except httpx.HTTPError as exc:
        yield "error", {"message": f"Agent 连接失败：{exc}", "degraded": True}


def parse_sse_data(lines: list[str]) -> dict[str, Any]:
    try:
        return json.loads("\n".join(lines))
    except json.JSONDecodeError:
        return {"raw": "\n".join(lines)}


def normalize_card(card: dict[str, Any], event_type: str = "") -> dict[str, Any]:
    normalized = dict(card)
    card_type = str(normalized.get("type") or event_type or "card")
    normalized["type"] = CARD_TYPE_MAP.get(card_type, card_type.lower())
    return normalized


def extract_location(geo_data: dict[str, Any] | None) -> str | None:
    if not geo_data:
        return None
    coords = geo_data.get("coords", geo_data)
    lat = coords.get("latitude")
    lng = coords.get("longitude")
    if lat is None or lng is None:
        return None
    return f"{lng},{lat}"


def is_nearby_query(text: str) -> bool:
    lowered = text.lower()
    return any(word in text for word in ["附近", "周边", "最近", "影院", "电影院"]) or any(
        word in lowered for word in ["nearby", "around", "cinema", "movie theater"]
    )


def build_agent_payload(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if is_nearby_query(text) and st.session_state.location:
        payload["location"] = st.session_state.location
    if st.session_state.phone:
        payload["phone"] = st.session_state.phone
    return payload


def run_agent(
    message: str,
    payload: dict[str, Any] | None = None,
    event: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    request_payload: dict[str, Any] = {
        "sessionId": st.session_state.session_id,
        "userId": st.session_state.user_id,
        "message": message,
    }
    if payload:
        request_payload["payload"] = payload
    if event:
        request_payload["event"] = event

    assistant_text = ""
    cards: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        status_line = st.empty()
        message_slot = st.empty()

        for event_name, event_data in stream_sse("/api/agent/chat/stream", request_payload):
            raw_events.append({"event": event_name, "data": event_data})
            if event_name == "thinking":
                status_line.caption(str(event_data.get("message") or "正在处理..."))
            elif event_name == "message":
                assistant_text = str(event_data.get("content") or "")
                status_line.empty()
                message_slot.write(assistant_text)
            elif event_name == "card":
                cards.append(
                    normalize_card(
                        event_data.get("data") or {},
                        str(event_data.get("type") or ""),
                    )
                )
            elif event_name == "error":
                status_line.empty()
                assistant_text = str(event_data.get("message") or "处理失败")
                message_slot.error(assistant_text)
            elif event_name == "done":
                status_line.empty()

        if cards:
            render_card_list(cards, interactive=False)

    return assistant_text, cards, raw_events


def queue_agent_action(event: str, message: str, payload: dict[str, Any]) -> None:
    st.session_state.agent_pending_action = {
        "event": event,
        "message": message,
        "payload": payload,
    }
    st.rerun()


def render_card_list(cards: list[dict[str, Any]], interactive: bool) -> None:
    for index, card in enumerate(cards):
        render_card(card, index, interactive=interactive)


def render_card(card: dict[str, Any], index: int, interactive: bool) -> None:
    card_type = str(card.get("type") or "card")
    card_id = str(card.get("id") or index)
    key_prefix = f"agent_{card_type}_{card_id}_{index}"

    with st.container(border=True):
        st.markdown(f"**{card.get('title') or card_id}**")
        if card.get("subtitle"):
            st.caption(str(card["subtitle"]))

        meta = card.get("meta") or {}
        if card_type == "seat_map":
            render_agent_seat_card(card, key_prefix, interactive)
        elif card_type == "ticket":
            st.success("出票成功")
            st.json(meta or card, expanded=False)
        elif meta:
            st.json(meta, expanded=False)

        if interactive and card_type != "seat_map":
            render_agent_card_action(card, card_type, card_id, key_prefix)


def render_agent_card_action(
    card: dict[str, Any],
    card_type: str,
    card_id: str,
    key_prefix: str,
) -> None:
    if card_type == "movie":
        payload = _action_payload(card) or {"movieId": card_id, "movieName": card.get("title")}
        if st.button("选择电影", key=f"{key_prefix}_select", icon=":material/movie:"):
            queue_agent_action("select_movie", "选择电影", payload)
    elif card_type == "cinema":
        payload = _action_payload(card) or {"cinemaId": card_id, "cinemaName": card.get("title")}
        if st.button("选择影院", key=f"{key_prefix}_select", icon=":material/location_on:"):
            queue_agent_action("select_cinema", "选择影院", payload)
    elif card_type == "showtime":
        payload = _action_payload(card) or {"showtimeId": card_id}
        if st.button("选择这场", key=f"{key_prefix}_select", icon=":material/event_seat:"):
            queue_agent_action("select_showtime", "选择场次", payload)
    elif card_type == "confirm_order":
        payload = card.get("payload") or {}
        order_id = payload.get("orderId") or card_id
        st.write(f"订单号：`{order_id}`")
        if st.button("确认支付", key=f"{key_prefix}_pay", icon=":material/payments:"):
            queue_agent_action("pay_order", "确认支付", {"orderId": order_id})
    elif card_type == "payment":
        if st.button("模拟支付", key=f"{key_prefix}_pay", icon=":material/payments:"):
            queue_agent_action("pay_order", "确认支付", {"orderId": card_id})


def render_agent_seat_card(
    card: dict[str, Any],
    key_prefix: str,
    interactive: bool,
) -> None:
    seats = card.get("seats") or []
    if not interactive:
        st.dataframe(seats, width="stretch")
        return

    selected: list[str] = []
    rows: dict[str, list[dict[str, Any]]] = {}
    for seat in seats:
        rows.setdefault(str(seat.get("row") or "-"), []).append(seat)

    for row, row_seats in rows.items():
        cols = st.columns(8)
        for col, seat in zip(cols, row_seats):
            seat_id = str(seat.get("seatId"))
            available = seat.get("status") == "available"
            with col:
                checked = st.checkbox(
                    seat_id,
                    key=f"{key_prefix}_{seat_id}",
                    disabled=not available,
                )
            if checked:
                selected.append(seat_id)

    if st.button("确认座位", key=f"{key_prefix}_confirm", icon=":material/check_circle:"):
        if not selected:
            st.warning("请先选择座位。")
        else:
            queue_agent_action(
                "confirm_order",
                "确认座位",
                {
                    "showtimeId": card.get("id"),
                    "seatIds": selected,
                    "ticketCount": len(selected),
                },
            )


def _action_payload(card: dict[str, Any]) -> dict[str, Any]:
    actions = card.get("actions") or []
    if actions and isinstance(actions[0], dict):
        return dict(actions[0].get("payload") or {})
    return {}


def render_ai_page() -> None:
    pending_action = st.session_state.agent_pending_action
    if pending_action:
        st.session_state.agent_pending_action = None
        st.session_state.agent_messages.append(
            {"role": "user", "content": pending_action["message"], "cards": []}
        )

    for message in st.session_state.agent_messages:
        with st.chat_message(
            message["role"],
            avatar=":material/person:" if message["role"] == "user" else ":material/smart_toy:",
        ):
            st.write(message["content"])
            if message.get("cards"):
                render_card_list(message["cards"], interactive=False)

    if pending_action:
        with st.chat_message("user", avatar=":material/person:"):
            st.write(pending_action["message"])
        text, cards, raw = run_agent(
            pending_action["message"],
            payload=pending_action["payload"],
            event=pending_action["event"],
        )
        st.session_state.agent_active_cards = cards
        st.session_state.agent_messages.append(
            {"role": "assistant", "content": text, "cards": cards, "raw": raw}
        )

    if st.session_state.agent_active_cards:
        st.subheader("当前可操作卡片")
        render_card_list(st.session_state.agent_active_cards, interactive=True)

    prompt = st.chat_input(
        "输入购票需求，例如：帮我订两张明晚8点后的喜剧片",
        submit_mode="disable",
    )
    if prompt:
        payload = build_agent_payload(prompt)
        if is_nearby_query(prompt) and not payload.get("location"):
            st.warning("查询附近影院需要定位。请允许浏览器定位，或在侧边栏手动填写经纬度。")
            return

        st.session_state.agent_messages.append({"role": "user", "content": prompt, "cards": []})
        with st.chat_message("user", avatar=":material/person:"):
            st.write(prompt)
        text, cards, raw = run_agent(prompt, payload=payload)
        st.session_state.agent_active_cards = cards
        st.session_state.agent_messages.append(
            {"role": "assistant", "content": text, "cards": cards, "raw": raw}
        )


def render_traditional_page() -> None:
    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        st.subheader("影片与场次")
        with st.form("movie_filter"):
            cols = st.columns([0.45, 0.35, 0.2], vertical_alignment="bottom")
            keyword = cols[0].text_input("影片关键词", placeholder="流浪地球 / 喜剧")
            genre = cols[1].selectbox("类型", ["", "喜剧", "科幻", "动作", "爱情"])
            submitted = cols[2].form_submit_button("查询", icon=":material/search:")

        movies_payload = api_get(
            "/api/v1/movies",
            {"keyword": keyword, "genre": genre} if submitted else None,
        )
        movies = (response_data(movies_payload) or {}).get("movies", [])
        render_movie_picker(movies)

        st.divider()
        render_showtime_picker()

    with right:
        st.subheader("选座与订单")
        render_seat_and_order_panel()


def render_movie_picker(movies: list[dict[str, Any]]) -> None:
    if not movies:
        st.info("暂无影片。")
        return
    cols = st.columns(3)
    for index, movie in enumerate(movies):
        with cols[index % 3].container(border=True, height="stretch"):
            st.markdown(f"**{movie.get('movieName')}**")
            st.caption(f"{movie.get('genre')} · {movie.get('durationMinutes')} 分钟")
            st.metric("评分", movie.get("score", "-"))
            if st.button(
                "选择影片",
                key=f"movie_{movie.get('movieId')}",
                icon=":material/movie:",
            ):
                st.session_state.selected_movie = movie
                st.session_state.selected_showtime = None
                st.session_state.selected_seats = []
                st.session_state.current_order = None
                st.rerun()


def render_showtime_picker() -> None:
    selected_movie = st.session_state.selected_movie or {}
    selected_cinema = st.session_state.selected_cinema or {}

    cols = st.columns(3)
    date = cols[0].selectbox("日期", ["today", "tomorrow", "weekend"])
    time_range = cols[1].selectbox("时间", ["", "18:00", "19:00", "20:00", "21:00"])
    ticket_count = cols[2].number_input("票数", min_value=1, max_value=6, value=2)

    showtime_params = {
        "movieId": selected_movie.get("movieId"),
        "cinemaId": selected_cinema.get("cinemaId"),
        "date": date,
        "timeRange": time_range or None,
        "ticketCount": int(ticket_count),
    }
    payload = api_get(
        "/api/v1/showtimes",
        {key: value for key, value in showtime_params.items() if value},
    )
    showtimes = (response_data(payload) or {}).get("showtimes", [])

    if not showtimes:
        st.info("没有匹配场次。")
        return

    for showtime in showtimes:
        with st.container(border=True):
            cols = st.columns([0.34, 0.26, 0.18, 0.22], vertical_alignment="center")
            cols[0].markdown(f"**{showtime.get('movieName')}**")
            cols[0].caption(f"{showtime.get('cinemaName')} · {showtime.get('hallName')}")
            cols[1].write(f"{showtime.get('date')} {showtime.get('time')}")
            cols[2].metric("票价", f"{showtime.get('price')} 元")
            cols[2].caption(f"余座 {showtime.get('remainingSeats')}")
            if cols[3].button(
                "选座",
                key=f"showtime_{showtime.get('showtimeId')}",
                icon=":material/event_seat:",
            ):
                st.session_state.selected_showtime = showtime
                st.session_state.selected_seats = []
                st.session_state.current_order = None
                st.session_state.ticket = None
                st.rerun()


def render_seat_and_order_panel() -> None:
    showtime = st.session_state.selected_showtime
    if not showtime:
        st.info("请先选择一个场次。")
        return

    with st.container(border=True):
        st.markdown(f"**{showtime.get('movieName')}**")
        st.caption(f"{showtime.get('cinemaName')} · {showtime.get('date')} {showtime.get('time')}")

    seats_payload = api_get(f"/api/v1/showtimes/{showtime.get('showtimeId')}/seats")
    seats = (response_data(seats_payload) or {}).get("seats", [])
    selected = render_traditional_seat_map(showtime.get("showtimeId"), seats)
    st.session_state.selected_seats = selected

    if selected and not st.session_state.current_order:
        if st.button("锁座并创建订单", type="primary", icon=":material/lock:"):
            payload = api_post(
                "/api/v1/orders",
                {
                    "showtimeId": showtime.get("showtimeId"),
                    "seatIds": selected,
                    "ticketCount": len(selected),
                    "userId": st.session_state.user_id,
                },
            )
            order = response_data(payload)
            if order:
                st.session_state.current_order = order
                st.rerun()

    if st.session_state.current_order:
        render_order_panel(st.session_state.current_order)


def render_traditional_seat_map(
    showtime_id: str,
    seats: list[dict[str, Any]],
) -> list[str]:
    selected: list[str] = []
    rows: dict[str, list[dict[str, Any]]] = {}
    for seat in seats:
        rows.setdefault(str(seat.get("row") or "-"), []).append(seat)

    for row, row_seats in rows.items():
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(row)
            cols = st.columns(8)
            for col, seat in zip(cols, row_seats):
                seat_id = str(seat.get("seatId"))
                available = seat.get("status") == "available"
                with col:
                    checked = st.checkbox(
                        seat_id,
                        key=f"trad_{showtime_id}_{seat_id}",
                        disabled=not available,
                    )
                if checked:
                    selected.append(seat_id)

    st.caption(f"已选择 {len(selected)} 个座位")
    return selected


def render_order_panel(order: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**订单确认**")
        cols = st.columns(2)
        cols[0].write(f"订单号：`{order.get('orderId')}`")
        cols[0].write(f"座位：{', '.join(order.get('seatIds') or [])}")
        cols[1].metric("应付", f"{order.get('amount', 0)} 元")
        cols[1].caption(f"状态：{order.get('status')}")
        if st.button("模拟支付并出票", type="primary", icon=":material/payments:"):
            payload = api_post(
                f"/api/v1/orders/{order.get('orderId')}/pay",
                {
                    "idempotencyKey": uuid.uuid4().hex,
                    "phone": st.session_state.phone,
                    "userId": st.session_state.user_id,
                },
            )
            ticket = response_data(payload)
            if ticket:
                st.session_state.ticket = ticket
                st.session_state.current_order = ticket
                st.rerun()

    if st.session_state.ticket:
        with st.container(border=True):
            st.success("电子票已生成")
            st.write(f"电影：{st.session_state.ticket.get('movieName')}")
            st.write(f"影院：{st.session_state.ticket.get('cinemaName')}")
            st.write(f"取票码：{', '.join(st.session_state.ticket.get('ticketCodes') or [])}")


def render_admin_page() -> None:
    payload = api_get("/api/v1/admin/overview")
    overview = response_data(payload) or {}
    cols = st.columns(4)
    cols[0].metric("影片", overview.get("movieCount", 0))
    cols[1].metric("场次", overview.get("showtimeCount", 0))
    cols[2].metric("订单", overview.get("orderCount", 0))
    cols[3].metric("模拟成交", f"{overview.get('revenue', 0)} 元")

    st.subheader("最近订单")
    orders = overview.get("orders") or []
    if orders:
        st.dataframe(orders, width="stretch")
    else:
        st.info("暂无订单。")


def render_outbox() -> None:
    with st.expander("本地短信发件箱"):
        if not OUTBOX_PATH.exists():
            st.info("暂无短信记录。")
            return
        try:
            records = json.loads(OUTBOX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.warning("短信发件箱文件不是有效 JSON。")
            return
        if records:
            st.dataframe(records, width="stretch")
        else:
            st.info("暂无短信记录。")


init_state()

with st.sidebar:
    st.header("运行设置")
    st.text_input("FastAPI 地址", value="http://127.0.0.1:8001", key="api_base")
    st.text_input("Session ID", key="session_id")
    st.text_input("用户 ID", key="user_id")
    st.text_input("手机号", key="phone")
    manual_location = st.text_input(
        "经纬度",
        value=st.session_state.location,
        placeholder="121.4737,31.2304",
    )
    if manual_location:
        st.session_state.location = manual_location

geo_data = get_geolocation(component_key="movie_agent_geolocation")
browser_location = extract_location(geo_data)
if browser_location:
    st.session_state.location = browser_location

st.title("电影票智能体")
mode = st.segmented_control(
    "购票模式",
    ["AI 购票", "传统购票", "运营看板"],
    key="mode",
)

if mode == "AI 购票":
    render_ai_page()
elif mode == "传统购票":
    render_traditional_page()
else:
    render_admin_page()

render_outbox()
