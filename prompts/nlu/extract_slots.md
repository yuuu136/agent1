从用户输入中识别意图并抽取电影票购票槽位。

需要识别的意图：
- book_ticket
- search_movie
- search_showtime
- select_or_modify
- order_query
- refund_policy
- faq
- admin_query
- smalltalk

需要抽取的槽位：
- movieName
- genre
- date
- timeRange
- city
- location
- cinemaName
- hallType
- ticketCount
- pricePreference
- seatPreference
- showtimeId
- seatIds
- orderId

输出 JSON，不要输出多余解释：
{
  "intent": "",
  "confidence": 0.0,
  "slots": {},
  "is_modification": false,
  "reference_text": "",
  "missing_slots": []
}
