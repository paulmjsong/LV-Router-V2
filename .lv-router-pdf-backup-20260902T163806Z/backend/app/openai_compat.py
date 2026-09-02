from __future__ import annotations

import asyncio
import json
import logging
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
from .runtime import WorkflowRuntime
from .schemas import ChatRequest, ChatResponse, Quality, WorkflowId

logger = logging.getLogger("infonet.openai_compat")
router = APIRouter(prefix="/v1", tags=["OpenAI compatibility"])

MODEL_TO_WORKFLOW: dict[str, WorkflowId] = {
    "auto": WorkflowId.AUTO,
    "direct": WorkflowId.DIRECT,
    "web-search": WorkflowId.WEB_SEARCH,
    "gist-regulations": WorkflowId.REGULATIONS,
    "research-paper": WorkflowId.PAPER,
}
MODEL_DESCRIPTIONS: dict[str, str] = {
    "auto": "Automatic Routing",
    "direct": "Direct Response",
    "web-search": "Web Search",
    "gist-regulations": "GIST Regulations",
    "research-paper": "Research Paper Drafting",
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
    files: list[dict[str, Any]] = Field(default_factory=list)


def _runtime(request: Request) -> WorkflowRuntime:
    return request.app.state.runtime


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
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


def _quality(metadata: dict[str, Any]) -> Quality:
    raw = str(metadata.get("quality", Quality.BALANCED.value))
    try:
        return Quality(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported quality: {raw}") from exc


async def _execute_request(
    *,
    payload: OpenAIChatRequest,
    request: Request,
    user: UserContext,
    token_sink: Callable[[LLMStreamChunk], Awaitable[None]] | None = None,
) -> ChatResponse:
    workflow = MODEL_TO_WORKFLOW.get(payload.model)
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow: {payload.model}")
    if payload.files or payload.metadata.get("files"):
        raise HTTPException(status_code=400, detail="File uploads are disabled in this version")
    return await _runtime(request).execute(
        ChatRequest(
            query=_latest_user_query(payload.messages),
            conversation_id=_conversation_id(request, user),
            workflow=workflow,
            quality=_quality(payload.metadata),
        ),
        user,
        token_sink=token_sink,
    )


def _completion_payload(*, completion_id: str, model: str, response: ChatResponse) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "infonet": {
            "run_id": str(response.run_id),
            "conversation_id": str(response.conversation_id),
            "workflow": response.workflow.value,
            "route_reason": response.route_reason,
            "route_fallback": response.route_fallback,
            "route_difficulty": response.route_difficulty,
            "model_tiers": response.model_tiers,
            "sources": [source.model_dump(mode="json") for source in response.sources],
        },
    }


def _sse(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _chunk(completion_id: str, model: str, content: str, finish_reason: str | None = None) -> bytes:
    return _sse({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
    })


def _should_replay_completed_answer(
    answer: str,
    answer_body_seen: bool,
) -> bool:
    """Return True when a workflow completed without streaming its answer body."""
    return bool(answer.strip()) and not answer_body_seen


def _stream_request(
    *,
    payload: OpenAIChatRequest,
    request: Request,
    user: UserContext,
    completion_id: str,
):
    async def generate():
        queue: asyncio.Queue[LLMStreamChunk | Exception | None] = asyncio.Queue()
        answer_body_seen = False

        async def sink(chunk: LLMStreamChunk) -> None:
            nonlocal answer_body_seen
            if (
                chunk.content
                and chunk.event_type in {"token", "sources", "final-answer"}
            ):
                answer_body_seen = True
            await queue.put(chunk)

        async def run() -> None:
            try:
                response = await _execute_request(
                    payload=payload,
                    request=request,
                    user=user,
                    token_sink=sink,
                )

                if _should_replay_completed_answer(
                    response.answer,
                    answer_body_seen,
                ):
                    logger.info(
                        "Emitting non-streamed final answer for workflow=%s",
                        response.workflow.value,
                    )
                    await queue.put(
                        LLMStreamChunk(
                            content=response.answer,
                            requested_alias=(
                                response.model_tiers[-1]
                                if response.model_tiers
                                else payload.model
                            ),
                            served_model=payload.model,
                            event_type="final-answer",
                        )
                    )

                await queue.put(
                    LLMStreamChunk(
                        content="",
                        requested_alias=(
                            response.model_tiers[-1]
                            if response.model_tiers
                            else payload.model
                        ),
                        served_model=payload.model,
                        finish_reason="stop",
                        event_type="complete",
                    )
                )
            except Exception as exc:
                await queue.put(exc)
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            # Open the SSE stream immediately without adding a visible status line.
            yield _chunk(completion_id, payload.model, "")
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
                    continue

                if item is None:
                    break

                if isinstance(item, Exception):
                    logger.error(
                        "Streaming request failed: %s",
                        item,
                        exc_info=(type(item), item, item.__traceback__),
                    )
                    message = "The request failed. Check the backend logs."
                    if isinstance(item, APITimeoutError):
                        message = "The selected model timed out."
                    yield _chunk(
                        completion_id,
                        payload.model,
                        f"\n\n> **Error:** {message}\n",
                    )
                    yield _chunk(
                        completion_id,
                        payload.model,
                        "",
                        finish_reason="stop",
                    )
                    break

                if item.event_type == "complete":
                    yield _chunk(
                        completion_id,
                        payload.model,
                        "",
                        finish_reason="stop",
                    )
                    break

                if item.content:
                    yield _chunk(
                        completion_id,
                        item.served_model or payload.model,
                        item.content,
                    )

            yield b"data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        generate(),
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
            {"id": model_id, "object": "model", "created": created, "owned_by": "infonet", "name": name}
            for model_id, name in MODEL_DESCRIPTIONS.items()
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
        return _stream_request(payload=payload, request=request, user=user, completion_id=completion_id)
    try:
        response = await _execute_request(payload=payload, request=request, user=user)
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="The selected model timed out") from exc
    return JSONResponse(_completion_payload(completion_id=completion_id, model=payload.model, response=response))
