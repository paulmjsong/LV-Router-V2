from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .auth import UserContext, get_current_user
from .config import get_settings
from .db import Database
from .documents import DocumentService
from .github_publisher import GitHubPublisher
from .llm import LLMGateway
from .rag import RAGService
from .repositories import CollectionRepository, ConversationRepository, DocumentRepository, RunRepository
from .routing import StageModelPolicy
from .runtime import WorkflowRuntime
from .openai_compat import router as openai_router
from .schemas import (
    ChatRequest,
    ChatResponse,
    CollectionCreate,
    CollectionInfo,
    RunDecisionRequest,
    UploadResponse,
    WorkflowInfo,
)
from .storage import build_object_store
from .workflows.builders import WorkflowServices

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("saegyeol")


@asynccontextmanager
async def lifespan(app: FastAPI):
    database = Database(settings)
    llm = LLMGateway(settings)
    try:
        await database.setup()
        collections = CollectionRepository(database)
        documents = DocumentRepository(database)
        runs = RunRepository(database)
        conversations = ConversationRepository(database)
        rag = RAGService(settings, database, llm)
        document_service = DocumentService(
            settings=settings,
            collections=collections,
            documents=documents,
            llm=llm,
            object_store=build_object_store(settings),
        )
        github = GitHubPublisher(settings)

        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            await checkpointer.setup()
            runtime = WorkflowRuntime(
                services=WorkflowServices(
                    llm=llm,
                    rag=rag,
                    policy=StageModelPolicy(),
                    github=github,
                ),
                checkpointer=checkpointer,
                runs=runs,
                conversations=conversations,
                settings=settings,
            )
            app.state.database = database
            app.state.collections = collections
            app.state.document_service = document_service
            app.state.runtime = runtime
            yield
    finally:
        await llm.close()
        await database.close()


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
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


def collection_repository(request: Request) -> CollectionRepository:
    return request.app.state.collections


def document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/me")
async def me(user: UserContext = Depends(get_current_user)) -> UserContext:
    return user


@app.get("/api/workflows", response_model=list[WorkflowInfo])
async def workflows(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> list[WorkflowInfo]:
    return runtime(request).list_workflows(user)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> ChatResponse:
    return await runtime(request).execute(payload, user)


@app.post("/api/runs/{run_id}/decision", response_model=ChatResponse)
async def decide_run(
    run_id: UUID,
    payload: RunDecisionRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> ChatResponse:
    return await runtime(request).resume(
        run_id=run_id,
        decision=payload.decision,
        feedback=payload.feedback,
        user=user,
    )


@app.get("/api/collections", response_model=list[CollectionInfo])
async def list_collections(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> list[CollectionInfo]:
    return await collection_repository(request).list_accessible(user)


@app.post("/api/collections", response_model=CollectionInfo)
async def create_collection(
    payload: CollectionCreate,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> CollectionInfo:
    return await collection_repository(request).create(payload, user)


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(
    request: Request,
    collection_id: Annotated[UUID, Form()],
    files: Annotated[list[UploadFile], File()],
    user: UserContext = Depends(get_current_user),
) -> UploadResponse:
    service = document_service(request)
    indexed = []
    for upload in files:
        indexed.append(await service.ingest(upload, collection_id, user))
    return UploadResponse(documents=indexed)
