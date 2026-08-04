# RAG 使用说明

## 作用

RAG 用于检索电影票智能体的稳定知识，例如业务规则、退改签说明、推荐策略、异常处理话术、动态卡片协议和演示案例。

实时座位、锁座、下单、支付、出票、真实订单状态不能靠 RAG 判断，必须调用业务接口。

## 目录

```text
data/knowledge/             原始知识文档
data/vector_store/chroma/   Chroma 持久化向量数据库
app/rag/                    RAG 代码
app/api/rag.py              RAG 测试接口
```

## 配置

模型和向量库配置在：

```text
app/config/rag.yml
app/config/chroma.yml
app/config/agent.yml
```

当前模型：

```yaml
chat_model_name: qwen-max
embedding_model_name: text-embedding-v4
```

本地 `.env` 需要：

```env
DASHSCOPE_API_KEY=你的通义千问 API Key
```

## 构建索引

启动服务后，可以调用：

```text
POST /agent/rag/build
```

也可以命令行构建：

```bash
python -m app.rag.build_index
```

构建完成后会写入 Chroma 持久化数据库：

```text
data/vector_store/chroma/
```

## 检索知识

```text
POST /agent/rag/search
```

请求：

```json
{
  "query": "座位被抢了怎么办？",
  "top_k": 5
}
```

## 基于 RAG 回答

```text
POST /agent/rag/answer
```

请求：

```json
{
  "query": "电影票可以退吗？",
  "top_k": 5
}
```

返回内容包含：

- `message`：模型基于检索片段生成的回答。
- `contexts`：本次使用的知识片段。

## Agent 接入边界

- 用户问规则、说明、推荐理由时，调用 `rag_service.answer()`。
- 用户要查场次、锁座、下单、支付时，调用业务工具。
- 构建索引会重建 Chroma collection，避免旧 chunk 残留。
