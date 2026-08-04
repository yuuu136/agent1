---
doc_id: kt-agent-dynamic-cards
title: 动态卡片协议与前端交互规范
category: frontend_cards
version: 1.0.0
last_updated: 2026-07-28
retrieval_keywords:
  - 动态卡片
  - Text Cards
  - 前端协议
  - React H5
  - 卡片事件
  - 座位图
---

# 动态卡片协议与前端交互规范

## 双模交互原则

系统采用“对话文本 + 动态卡片”的双模交互。文本负责解释、引导和情绪表达；卡片负责展示结构化信息和承接选择操作。

Agent 每次响应都应返回统一结构：

```json
{
  "message": "我找到了几场适合您的电影。",
  "state": "choosing_showtime",
  "progress": [
    {"key": "movie", "label": "选片", "status": "done"},
    {"key": "showtime", "label": "场次", "status": "current"},
    {"key": "seat", "label": "座位", "status": "pending"}
  ],
  "cards": [],
  "suggestions": ["选第一个", "换便宜点", "换近一点"]
}
```

## 卡片类型

### movie_list

影片列表卡片，用于推荐电影。

字段建议：

- `movieId`
- `movieName`
- `posterUrl`
- `genre`
- `duration`
- `rating`
- `tags`
- `reason`

按钮事件：

- `select_movie`
- `view_showtimes`

### cinema_list

影院列表卡片，用于推荐影院。

字段建议：

- `cinemaId`
- `cinemaName`
- `address`
- `distance`
- `minPrice`
- `features`
- `reason`

按钮事件：

- `select_cinema`
- `view_showtimes`

### showtime_list

场次列表卡片，用于推荐可购票场次。

字段建议：

- `showtimeId`
- `movieId`
- `movieName`
- `cinemaId`
- `cinemaName`
- `hallName`
- `hallType`
- `startTime`
- `endTime`
- `price`
- `remainingSeats`
- `distance`
- `recommendScore`
- `reason`

按钮事件：

- `select_showtime`
- `view_seats`

### seat_map

座位图卡片，用于展示座位状态和选择座位。

字段建议：

- `showtimeId`
- `hallName`
- `screenLabel`
- `rows`
- `cols`
- `selectedSeatIds`
- `recommendedSeatIds`
- `seats`

座位状态：

- `available`：可选。
- `sold`：已售。
- `locked`：已锁。
- `selected`：当前选中。
- `recommended`：Agent 推荐。
- `disabled`：不可选。

按钮事件：

- `select_seats`
- `auto_pick_seats`

### order_confirm

订单确认卡片，用于支付前确认。

字段建议：

- `orderId`
- `movieName`
- `cinemaName`
- `hallName`
- `startTime`
- `seatLabels`
- `ticketCount`
- `unitPrice`
- `totalPrice`
- `discountAmount`
- `payAmount`
- `expireAt`

按钮事件：

- `confirm_order`
- `cancel_order`
- `change_seats`

### payment

模拟支付卡片，用于点击支付。

字段建议：

- `orderId`
- `payAmount`
- `payMethods`
- `expireAt`

按钮事件：

- `pay_order`

### ticket

出票成功卡片，用于展示购票结果。

字段建议：

- `ticketId`
- `orderId`
- `movieName`
- `cinemaName`
- `hallName`
- `startTime`
- `seatLabels`
- `pickupCode`
- `qrCodeUrl`
- `tips`

按钮事件：

- `view_order`
- `add_calendar`

### recommendation

推荐或异常处理卡片，用于展示替代方案。

字段建议：

- `title`
- `description`
- `options`

常见用途：

- 当前场次售罄。
- 座位被抢。
- 价格过高。
- 无符合条件结果。
- 推荐相近时间或相近影院。

## 前端事件协议

卡片点击统一回传给 Agent：

```json
{
  "sessionId": "s001",
  "userId": "u001",
  "type": "event",
  "event": "select_showtime",
  "payload": {
    "showtimeId": "st001"
  }
}
```

Agent 应根据事件更新槽位，并继续任务规划。

## 快捷建议 Chips

快捷建议用于降低输入成本。建议每轮返回 2 到 4 个。

常见建议：

- 选第一个。
- 换便宜点。
- 换近一点。
- 晚一点。
- 看座位。
- 确认支付。
- 换一家影院。
- 还是老位置。

## 进度条规范

购票流程进度：

```text
选片 -> 选影院/场次 -> 选座 -> 确认 -> 支付 -> 出票
```

如果用户一句话信息完整，前面的步骤可直接标记为完成，这就是自动跳步在 UI 上的体现。

## 文本与卡片协作

Agent 文本不应重复卡片全部字段。文本应解释为什么推荐、下一步做什么、异常如何处理。

示例：

文本：“我按便宜优先帮您筛了 3 场，第一场最省钱，第二场距离最近。”

卡片：展示三条场次具体信息。

## 卡片为空的情况

如果没有卡片，Agent 也应该返回 `cards: []`，避免前端解析异常。

常见纯文本情况：

- 闲聊。
- FAQ 规则回答。
- 澄清追问。

