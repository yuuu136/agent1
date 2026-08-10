# 电影票智能体意图目录

这个文件是意图识别的唯一语言词表来源：

- `[intent]`：用于 Chroma 意图检索的完整用户表达。
- `[lexicon]`：用于规则匹配、同义表达和槽位提取。
- `[map]`：用于把多个表达映射成统一业务值，格式为 `统一值 | 同义词 1 | 同义词 2`。

新增中文表达时优先修改本文件；Python 代码只保留流程、正则解析和业务状态判断。

## [intent] search_movies
- 我想看电影
- 想找点片子看
- 有什么电影可以看
- 最近有什么上映的电影
- 帮我推荐几部电影
- 今天有什么片子
- 想看部电影
- 想看点片子
- 有没有电影看
- 适合情侣看的
- 适合情侣约会看的电影
- 适合亲子看的电影
- 推荐一部轻松的电影
- 票房最高的电影
- 最近热映的电影
- 最近比较火的电影
- 想看评分高一点的电影
- 有没有蜘蛛侠电影

## [intent] book_ticket
- 给我买两张今晚八点的电影票
- 帮我订一张功夫女足影票
- 买两张蜘蛛侠电影票
- 今晚去看科幻片
- 明晚八点订票
- 我想买电影票
- 帮我订票
- 我想看今晚的电影
- 给我来两张票
- 帮我选一个 IMAX 场次

## [intent] search_showtimes
- 蜘蛛侠有哪些场次
- 查一下今晚蜘蛛侠的放映时间
- 今天有哪些场次
- 有什么场次可以看
- 看一下这家影院的排片
- 明天下午有什么电影场次
- 帮我查最近一场
- 我想看看蜘蛛侠什么时候能看

## [intent] select_showtime
- 第一个场次
- 就这场
- 选这个时间
- 我要这一场
- 选第二个

## [intent] seat_query
- 打开座位图
- 我要选座
- 帮我看看座位
- 随便选个座位
- 你帮我选个好位置
- 选中间的位置
- 选后排座位
- 换个座位

## [intent] confirm_order
- 确认当前订单
- 就按这个订单
- 确认座位
- 订单没问题

## [intent] pay_order
- 直接支付
- 去支付
- 确认付款
- 我要付款
- 帮我支付订单

## [intent] nearby_cinema
- 附近有什么影院
- 离我近的电影院
- 周边有哪些影院
- 找附近影城
- 帮我找离当前位置近的影院
- 哪家电影院离我最近

## [intent] location_query
- 我现在在哪里
- 我的当前位置
- 当前经纬度是多少
- 我的地理位置是什么
- 这里是哪里
- 帮我查一下当前位置
- 我现在的具体位置

## [intent] price_query
- 票价多少
- 这一场多少钱
- 电影票多少钱
- 这场电影的价格是多少
- 一张票要多少钱

## [intent] select_or_modify
- 有便宜一点的吗
- 换个便宜的
- 晚一点的场次
- 早点的场次
- 不要 IMAX
- 换普通厅
- 换一家影院
- 换个时间
- 重新选场次
- 不要这个座位

## [intent] snack
- 来点爆米花
- 我要加一瓶可乐
- 有没有零食
- 给我来一份套餐
- 加两杯可乐
- 想买点小吃
- 电影票加一份爆米花

## [intent] select_snacks
- 我要一瓶可乐
- 选两份爆米花
- 加一瓶可乐
- 套餐来一份

## [intent] skip_snacks
- 不要零食
- 不需要爆米花
- 直接去支付，不加小吃

## [intent] cancel
- 取消当前购票
- 算了先不买了
- 不看了
- 取消支付

## [intent] order_query
- 我的订单
- 查一下订单
- 查看订单详情
- 支付结果怎么样
- 付款了吗
- 我的购票记录

## [intent] refund_order
- 我要退票
- 帮我退掉这张票
- 申请退款
- 退了吧

## [intent] refund_status_query
- 退款成功了吗
- 查询退票状态
- 退款结果怎么样

## [intent] faq
- 退票规则是什么
- 退款多久能到
- 改签规则
- 电影票可以退吗
- IMAX 和普通厅有什么区别
- 座位为什么刚才又没有了
- 优惠券怎么使用
- 支付后怎么取票

## [lexicon] greeting
- 你好
- 您好
- 嗨
- 哈喽
- hi
- hello
- 你好呀
- 您好呀
- 你好啊
- 您好啊
- 嗨呀
- 哈喽呀
- 早上好
- 早安
- 上午好
- 中午好
- 下午好
- 晚上好
- 晚安
- 再见
- 拜拜
- 拜拜了
- 开始
- start

## [lexicon] ack
- 好
- 好的
- 好吧
- 就好
- 就行
- 可以的
- 行
- 行吧
- 可以
- 嗯
- 哦
- 知道了
- 谢谢
- 谢谢你
- 感谢
- 多谢
- 辛苦了
- 没问题
- 没事
- 不用谢
- 收到
- 明白了
- 我知道了
- 很好

## [lexicon] cancel
- 取消
- 取消订单
- 不用了
- 算了
- 先不买
- 不买了
- 别买了
- 不要了
- 先不要了
- 暂时不要了

## [lexicon] cancel_contains
- 先不支付
- 暂时不支付
- 不想支付
- 不要支付
- 取消支付
- 取消付款
- 不支付
- 先不付
- 暂时不付
- 不想付
- 不想付款
- 不付款了
- 不付钱了
- 先不付款
- 暂时不付款
- 不付了
- 暂时不要了
- 先不要了
- 不想要了
- 不想买了
- 不用买了
- 不想看了
- 不看了
- 我不要了
- 我不想要了
- 算了吧

## [lexicon] showtime_query
- 有什么场次
- 有哪些场次
- 有些什么场次
- 场次有哪些
- 查场次
- 查看场次
- 看看场次
- 查一下场次
- 有什么时间
- 有哪些时间

## [lexicon] any_time
- 都可以
- 都行
- 随便
- 不限
- 时间不限
- 什么时候都可以
- 哪个时间都可以
- 无所谓

## [lexicon] movie_search
- 我想看电影
- 想看电影
- 我要看电影
- 看电影
- 去看电影
- 帮我看电影
- 最近热映
- 正在上映
- 有什么电影
- 有啥电影
- 有哪些电影
- 有些什么电影
- 有什么影片
- 有啥影片
- 有哪些影片
- 有些什么影片
- 推荐电影
- 推荐影片
- 帮我推荐电影
- 帮我推荐影片
- 有什么片子推荐
- 想看部片子
- 想看一部片
- 想看点片子
- 想找点片子看
- 看看电影
- 查电影
- 有电影看吗
- 有什么电影看
- 有啥电影看

## [lexicon] price_query
- 多少钱
- 多少元
- 票价
- 价格
- 价位
- 什么价格
- 什么价
- 单价
- 费用
- 贵不贵

## [lexicon] order_query
- 查看订单
- 查订单
- 查询订单
- 看看订单
- 我的订单
- 订单详情
- 订单记录
- 历史订单
- 付款了吗
- 支付了吗
- 付钱了吗
- 支付状态
- 付款状态
- 支付结果
- 付款结果

## [lexicon] location_query
- 我的地理位置
- 我现在的具体位置
- 我的具体位置
- 我现在的位置
- 我的当前位置
- 当前位置
- 当前定位
- 定位信息
- 我的位置
- 现在在哪里
- 现在在哪
- 现在在哪儿
- 当前位置在哪里
- 当前位置在哪
- 当前位置在哪儿
- 这里是哪里
- 这里在哪
- 这里在哪儿
- 这是哪里
- 这是哪儿
- 我在哪里
- 我在哪
- 我在哪儿
- 我的坐标
- 当前坐标
- 经纬度

## [lexicon] price_preference
- 便宜
- 低价
- 价格低
- 价低
- 实惠
- 省钱
- 最低价
- 最省

## [lexicon] time_preference
- 早一点
- 早些
- 早点
- 早一些
- 早一点儿
- 更早
- 晚一点
- 晚些
- 晚点
- 晚一些
- 晚一点儿
- 更晚

## [lexicon] generic_booking
- book movie tickets
- buy movie tickets
- buy tickets
- book tickets
- movie tickets
- tickets

## [lexicon] faq
- 退票
- 改签
- 规则
- 政策
- 怎么处理
- FAQ

## [lexicon] nearby_cinema
- 附近
- 最近
- 周边
- 离我近
- 高德
- 地图

## [lexicon] nearby_cinema_english
- nearby
- around me
- map
- amap

## [lexicon] coupon
- 优惠
- 优惠券
- 券
- 折扣
- 便宜

## [lexicon] seat
- 座位
- 选座
- 靠中
- 中间
- 前排
- 后排
- 位置
- 坐席

## [lexicon] booking_with_seat
- 买
- 订
- 购票
- 影票
- 电影票
- 看电影

## [lexicon] payment
- 支付
- 付款
- 出票

## [lexicon] confirm
- 订单
- 确认
- 就这个
- 就这场
- 可以

## [lexicon] booking
- 买票
- 订票
- 购票
- 影票
- 电影票
- 电影
- 场次
- 影院
- 看
- 买

## [lexicon] booking_english
- book
- ticket
- movie
- showtime
- cinema

## [lexicon] movie_title_exclusion
- 附近
- 影院
- 座位
- 多少钱
- 价位
- 价格
- 优惠
- 规则

## [lexicon] non_movie
- 选择影院
- 选影院
- 选择电影
- 选电影
- 选择这场
- 这个
- 这场
- 这家
- 就这个
- 就这场
- 就这家
- 确认
- 确认一下
- 确认订单
- 确认座位
- 确认支付
- 都可以
- 都行
- 随便
- 不限
- 时间不限
- 什么时候都可以
- 哪个时间都可以
- 无所谓
- 换一场
- 换个场次
- 换到下一场
- 下一场
- 再来一场
- 下一个
- 换时间
- 换个时间
- 早一点
- 晚一点
- 便宜点
- 换便宜点
- 不要这个
- 不要这场
- 重新选座
- 重新选择座位
- 换个位置
- 换座位
- 更换座位
- 退了
- 退了吧
- 退掉
- 退掉吧
- 帮我退了
- 这张票退了
- 这个票退了

## [lexicon] modification
- 换
- 改
- 更
- 不要
- 不要这个
- 不要这场
- 不想要这场
- 便宜点
- 换便宜点
- 晚一点
- 早一点

## [lexicon] refund_request
- 退票
- 退款
- 申请退款
- 退了
- 退掉
- 退单
- 不要这张票
- 这张票不要
- 这张票退
- 这个票退

## [lexicon] refund_faq
- 规则
- 政策
- 说明
- 怎么
- 如何
- 能不能

## [lexicon] refund_status
- 退票状态
- 退款状态
- 退票结果
- 退款结果
- 退票成功了吗
- 退款成功了吗

## [lexicon] explicit_showtime_booking
- 选
- 选择
- 订
- 买
- 找

## [lexicon] recommendation_general
- 电影
- 影片
- 片子
- 片

## [lexicon] rag_markers
- 想
- 要
- 帮
- 找
- 查
- 看
- 推荐
- 有
- 哪些
- 什么
- 片
- 电影
- 影片
- 影院
- 影城
- 附近
- 周边
- 订单
- 票价
- 多少钱
- 零食
- 可乐
- 爆米花

## [lexicon] rag_title_markers
- 想
- 要
- 帮
- 找
- 查
- 看
- 推荐
- 有
- 哪些
- 什么
- 片
- 电影
- 影片
- 影院
- 影城
- 附近
- 周边

## [lexicon] booking_cues
- 买
- 订
- 购票
- 购买
- 预订
- 影票
- 电影票
- 张
- 买票
- 订票
- 座
- 今晚
- 明晚
- 今天
- 明天
- 上午
- 下午
- 晚上
- 几点

## [lexicon] movie_keyword_excluded
- 买
- 订
- 购票
- 购买
- 预订
- 影票
- 电影票
- 几张

## [lexicon] movie_keyword_generic
- 什么
- 啥
- 哪些
- 一些
- 些
- 推荐
- 热映
- 我想看
- 想看
- 我要看
- 帮我推荐
- 推荐一下
- 看看
- 查一下

## [lexicon] movie_keyword_prefix_excluded
- 什么
- 啥
- 哪些
- 有什么
- 有啥
- 有哪些
- 有些

## [lexicon] location_seat_exclusion
- 换个位置
- 换位置
- 换座位
- 选座
- 座位

## [lexicon] snack_negative
- 不要零食
- 不需要零食
- 不用零食
- 不吃零食
- 不加零食
- 不买零食
- 不买零食了
- 零食不要
- 零食不要了
- 不要爆米花
- 不需要爆米花
- 不加爆米花
- 不加爆米花了
- 不买爆米花
- 不要饮料
- 不买饮料
- 不要套餐
- 不加套餐
- 不需要小吃

## [lexicon] coupon_negative
- 不用券
- 不用优惠券
- 不使用优惠券
- 不要优惠券
- 优惠券不要
- 优惠券不要了
- 不需要优惠券
- 不想用券
- 不使用券
- 不用优惠
- 不使用优惠
- 不要优惠

## [lexicon] negative_hall
- 不要
- 不想要
- 不需要
- 不用
- 别要
- 不要看
- 不看

## [lexicon] plain_hall
- 普通厅
- 普通场
- 普通版
- 普通2D
- 2D就行
- 换普通
- 改普通

## [lexicon] time_evening
- 今晚
- 明晚
- 晚上

## [lexicon] time_afternoon
- 下午

## [lexicon] time_morning
- 上午

## [lexicon] date_today
- 今天
- 今晚

## [lexicon] date_tomorrow
- 明天
- 明晚

## [lexicon] date_after_tomorrow
- 后天

## [lexicon] date_weekend
- 周末

## [lexicon] movie_title_generic
- 电影
- 影片
- 片
- 片子
- 推荐

## [map] snack_alias
- 可乐 | 可乐 | 可口可乐 | cola | coke
- 爆米花 | 爆米花
- 雪碧 | 雪碧
- 饮料 | 饮料 | 汽水
- 套餐 | 套餐
- 小吃 | 小吃 | 零食

## [map] recommendation_criteria
- couple | 情侣 | 约会 | 恋人 | 对象 | 浪漫
- family | 亲子 | 带孩子 | 一家人 | 合家欢
- high_rating | 高分 | 高评分 | 评分高 | 评分最高 | 口碑
- box_office | 票房 | 最卖座
- hot | 热映 | 热门 | 最火 | 最近比较火 | 比较火 | 很火 | 比较热门 | 当下很火

## [map] genre
- 喜剧 | 喜剧
- 爱情 | 爱情
- 动作 | 动作
- 科幻 | 科幻
- 动画 | 动画
- 悬疑 | 悬疑
- 恐怖 | 恐怖

## [map] hall_type
- IMAX | IMAX
- 杜比 | 杜比
- 巨幕 | 巨幕
- 激光 | 激光

## [map] seat_preference
- middle | 中间 | 靠中 | 居中
- front | 前排
- back | 后排
- cheap | 便宜座 | 便宜位置
