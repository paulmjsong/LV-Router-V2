from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APITimeoutError
from pydantic import BaseModel, ConfigDict, Field

from .auth import UserContext, get_current_user, verify_openai_backend_key
from .llm import LLMStreamChunk
from .runtime import ResolvedRoute, WorkflowRuntime
from .schemas import ChatRequest, ChatResponse, Quality, WorkflowId


logger = logging.getLogger("saegyeol.openai_compat")
router = APIRouter(prefix="/v1", tags=["OpenAI compatibility"])


MODEL_TO_WORKFLOW: dict[str, WorkflowId] = {
    "lab-auto": WorkflowId.AUTO,
    "lab-direct": WorkflowId.DIRECT,
    "lab-rag": WorkflowId.DOMAIN_RAG,
    "lab-paper": WorkflowId.PAPER,
    "lab-grant": WorkflowId.GRANT,
    "lab-website": WorkflowId.WEBSITE,
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "lab-auto": "Local semantic router",
    "lab-direct": "Explicit direct inference",
    "lab-rag": "Domain RAG over authorized collections",
    "lab-paper": "Research-paper workflow",
    "lab-grant": "Grant-proposal workflow",
    "lab-website": "Website change proposal with approval",
}


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(request: Request) -> WorkflowRuntime:
    return request.app.state.runtime


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "input_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _message_text(message.get("content")).strip()
            if text:
                return text
    raise HTTPException(status_code=400, detail="No user message was supplied")


def _conversation_id(request: Request, user: UserContext) -> UUID:
    chat_id = request.headers.get("X-OpenWebUI-Chat-Id", "").strip()
    if not chat_id:
        return uuid4()
    try:
        return UUID(chat_id)
    except ValueError:
        return uuid5(NAMESPACE_URL, f"openwebui:{user.user_id}:{chat_id}")


def _collection_ids(metadata: dict[str, Any]) -> list[UUID]:
    raw = metadata.get("collection_ids", [])
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list):
        return []
    result: list[UUID] = []
    for item in raw[:20]:
        try:
            result.append(UUID(str(item)))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid collection ID: {item}") from exc
    return result


def _quality(metadata: dict[str, Any]) -> Quality:
    raw = str(metadata.get("quality", Quality.BALANCED.value))
    try:
        return Quality(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported quality: {raw}") from exc


_APPROVAL = re.compile(
    r"^/(approve|reject)\s+([0-9a-fA-F-]{36})(?:\s+(.+))?$",
    re.DOTALL,
)


async def _execute_request(
    *,
    payload: OpenAIChatRequest,
    request: Request,
    user: UserContext,
    token_sink: Callable[[LLMStreamChunk], Awaitable[None]] | None = None,
    route_sink: Callable[[ResolvedRoute, str], Awaitable[None]] | None = None,
) -> ChatResponse:
    query = _latest_user_query(payload.messages)
    match = _APPROVAL.fullmatch(query)
    if match:
        decision, run_id_text, feedback = match.groups()
        try:
            run_id = UUID(run_id_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid approval run ID") from exc
        return await _runtime(request).resume(
            run_id=run_id,
            decision=decision,
            feedback=feedback,
            user=user,
        )

    workflow = MODEL_TO_WORKFLOW.get(payload.model)
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Unknown model/workflow: {payload.model}")

    collections = _collection_ids(payload.metadata)
    use_documents: bool | None = payload.metadata.get("use_documents")
    if workflow == WorkflowId.DOMAIN_RAG:
        use_documents = True

    return await _runtime(request).execute(
        ChatRequest(
            query=query,
            conversation_id=_conversation_id(request, user),
            workflow=workflow,
            quality=_quality(payload.metadata),
            collection_ids=collections,
            use_documents=use_documents,
        ),
        user,
        token_sink=token_sink,
        route_sink=route_sink,
    )


def _approval_notice(response: ChatResponse) -> str:
    if response.status != "awaiting_approval":
        return ""
    return (
        f"\n\nApproval required. Reply `/approve {response.run_id}` to continue, "
        f"or `/reject {response.run_id} <reason>` to reject it."
    )


def _display_answer(response: ChatResponse) -> str:
    return response.answer + _approval_notice(response)


def _completion_payload(
    *,
    completion_id: str,
    model: str,
    content: str,
    response: ChatResponse,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "saegyeol": {
            "run_id": str(response.run_id),
            "conversation_id": str(response.conversation_id),
            "workflow": response.workflow.value,
            "route_reason": response.route_reason,
            "model_tiers": response.model_tiers,
            "status": response.status,
            "sources": [source.model_dump(mode="json") for source in response.sources],
        },
    }


def _chunk_payload(
    *,
    completion_id: str,
    model: str,
    created: int,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> str:
    delta: dict[str, str] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    item = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(item, ensure_ascii=False)}\n\n"


def _error_message(exc: BaseException) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    if isinstance(exc, APITimeoutError):
        return "The selected model timed out before completing its response."
    logger.error(
        "Streaming request failed: %s",
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return "The request failed. Check the backend logs for details."


def _stream_request(
    *,
    payload: OpenAIChatRequest,
    request: Request,
    user: UserContext,
    completion_id: str,
) -> StreamingResponse:
    created = int(time.time())

    async def events():
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        answer_token_seen = False
        served_model_seen = False

        async def token_sink(chunk: LLMStreamChunk) -> None:
            await queue.put(("token", chunk))

        async def route_sink(route: ResolvedRoute, alias: str) -> None:
            label = "Auto route" if payload.model == "lab-auto" else "Selected route"
            content = f"> **{label}:** `{route.workflow.value}` → `{alias}`\n\n"
            await queue.put(("route", content))

        async def runner() -> None:
            try:
                response = await _execute_request(
                    payload=payload,
                    request=request,
                    user=user,
                    token_sink=token_sink,
                    route_sink=route_sink,
                )
                await queue.put(("done", response))
            except Exception as exc:
                await queue.put(("error", exc))

        task = asyncio.create_task(runner())
        # Send visible content immediately; a role-only chunk is often rendered
        # as an empty loading indicator by OpenAI-compatible chat clients.
        yield _chunk_payload(
            completion_id=completion_id,
            model=payload.model,
            created=created,
            role="assistant",
            content="> **Status:** routing request…\n\n",
        )

        try:
            while True:
                try:
                    kind, value = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if kind == "route":
                    yield _chunk_payload(
                        completion_id=completion_id,
                        model=payload.model,
                        created=created,
                        content=str(value),
                    )
                    continue

                if kind == "token":
                    chunk: LLMStreamChunk = value
                    if not served_model_seen:
                        served_model_seen = True
                        yield _chunk_payload(
                            completion_id=completion_id,
                            model=payload.model,
                            created=created,
                            content=f"> **Served model:** `{chunk.served_model}`\n\n",
                        )
                    if chunk.content:
                        answer_token_seen = True
                        yield _chunk_payload(
                            completion_id=completion_id,
                            model=payload.model,
                            created=created,
                            content=chunk.content,
                        )
                    continue

                if kind == "error":
                    yield _chunk_payload(
                        completion_id=completion_id,
                        model=payload.model,
                        created=created,
                        content=f"\n\n**Error:** {_error_message(value)}",
                    )
                    yield _chunk_payload(
                        completion_id=completion_id,
                        model=payload.model,
                        created=created,
                        finish_reason="stop",
                    )
                    yield "data: [DONE]\n\n"
                    break

                response: ChatResponse = value
                if not answer_token_seen:
                    yield _chunk_payload(
                        completion_id=completion_id,
                        model=payload.model,
                        created=created,
                        content=_display_answer(response),
                    )
                else:
                    notice = _approval_notice(response)
                    if notice:
                        yield _chunk_payload(
                            completion_id=completion_id,
                            model=payload.model,
                            created=created,
                            content=notice,
                        )

                yield _chunk_payload(
                    completion_id=completion_id,
                    model=payload.model,
                    created=created,
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"
                break
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/models")
async def list_models(_: None = Depends(verify_openai_backend_key)) -> dict[str, Any]:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "saegyeol-lab",
                "name": description,
            }
            for model_id, description in MODEL_DESCRIPTIONS.items()
        ],
    }


@router.post("/chat/completions")
async def chat_completions(
    payload: OpenAIChatRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    completion_id = f"chatcmpl-{uuid4().hex}"
    if payload.stream:
        return _stream_request(
            payload=payload,
            request=request,
            user=user,
            completion_id=completion_id,
        )

    response = await _execute_request(payload=payload, request=request, user=user)
    content = _display_answer(response)
    return JSONResponse(
        _completion_payload(
            completion_id=completion_id,
            model=payload.model,
            content=content,
            response=response,
        )
    )
