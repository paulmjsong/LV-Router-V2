from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .auth import UserContext, get_current_user
from .config import get_settings
from .db import Database
from .gist_regulations import GISTRegulationsRetriever
from .llm import LLMGateway
from .openai_compat import router as openai_router
from .repositories import ConversationRepository, RunRepository
from .routing import StageModelPolicy
from .runtime import WorkflowRuntime
from .web_search import WebSearchService
from .schemas import ChatRequest, ChatResponse, WorkflowInfo
from .workflows.builders import WorkflowServices

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("infonet")


@asynccontextmanager
async def lifespan(app: FastAPI):
    database = Database(settings)
    llm = LLMGateway(settings)
    try:
        await database.setup()
        runs = RunRepository(database)
        conversations = ConversationRepository(database)
        regulations = GISTRegulationsRetriever(settings, llm)
        await regulations.initialize()
        logger.info("Loaded GIST regulations FAISS vectorstore from %s", settings.gist_regulations_index_dir)
        web_search = WebSearchService(settings)

        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            await checkpointer.setup()
            runtime = WorkflowRuntime(
                services=WorkflowServices(
                    llm=llm,
                    regulations=regulations,
                    web_search=web_search,
                    policy=StageModelPolicy(),
                    settings=settings,
                ),
                checkpointer=checkpointer,
                runs=runs,
                conversations=conversations,
                settings=settings,
            )
            app.state.database = database
            app.state.runtime = runtime
            yield
    finally:
        await llm.close()
        await database.close()


app = FastAPI(title=settings.app_name, version="0.5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(openai_router)


def runtime(request: Request) -> WorkflowRuntime:
    return request.app.state.runtime


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/api/me")
async def me(user: UserContext = Depends(get_current_user)) -> UserContext:
    return user


@app.get("/api/workflows", response_model=list[WorkflowInfo])
async def workflows(request: Request, user: UserContext = Depends(get_current_user)) -> list[WorkflowInfo]:
    return runtime(request).list_workflows(user)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> ChatResponse:
    return await runtime(request).execute(payload, user)
