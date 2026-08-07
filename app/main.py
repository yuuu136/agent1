import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import agent_service
from app.api.agent import router as agent_router
from app.api.prompts import router as prompts_router
from app.api.rag import router as rag_router
from app.schemas.agent import ChatRequest
from app.utils.config_handler import agent_config, rag_config


app = FastAPI(
    title=agent_config.get("service", {}).get("name", "Movie Ticket Agent Service")
)

server_config = agent_config.get("server", {})
cors_allow_origins = server_config.get("cors_allow_origins") or []
cors_allow_origin_regex = server_config.get("cors_allow_origin_regex")
allow_all_origins = "*" in cors_allow_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins and not cors_allow_origin_regex else cors_allow_origins,
    allow_origin_regex=cors_allow_origin_regex,
    allow_credentials=not allow_all_origins or bool(cors_allow_origin_regex),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)
app.include_router(prompts_router)
app.include_router(agent_router)


@app.get("/agent/health")
def health():
    return {
        "status": "ok",
        "service": agent_config.get("service", {}).get("name", "movie-ticket-agent"),
        "chat_model": agent_config.get("llm", {}).get("chat_model_name"),
        "embedding_model": rag_config.get("embedding", {}).get("embedding_model_name"),
    }


@app.post("/agent/chat")
def chat(request: ChatRequest):
    return agent_service.chat(request)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=agent_config.get("server", {}).get("host", "127.0.0.1"),
        port=agent_config.get("server", {}).get("port", 8001),
        reload=False,
    )
