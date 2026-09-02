from types import SimpleNamespace
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.openai_compat as compat
from app.openai_compat import OpenAIChatRequest
from app.schemas import WorkflowId
from app.workflows.builders import (
    _looks_like_retrieval_query_artifact,
    _pdf_request_with_history,
)


def test_internal_openwebui_task_is_detached_from_pdf_routing() -> None:
    payload = OpenAIChatRequest(
        model="pdf-document",
        messages=[{"role": "user", "content": "Return JSON search queries"}],
        metadata={
            "task": "query_generation",
            "files": [{"id": "file-1", "name": "paper.pdf"}],
        },
    )
    assert compat._internal_task_name(payload) == "query_generation"
    assert compat._workflow_for_request(payload) == WorkflowId.DIRECT
    assert compat._document_inputs(payload, 5000) == ("", False)


def test_user_pdf_request_keeps_attachment_inputs() -> None:
    payload = OpenAIChatRequest(
        model="pdf-document",
        messages=[
            {
                "role": "system",
                "content": '<context><source id="1">paper evidence</source></context>',
            },
            {"role": "user", "content": "What is the main claim?"},
        ],
        metadata={"files": [{"id": "file-1", "name": "paper.pdf"}]},
    )
    context, attached = compat._document_inputs(payload, 5000)
    assert "paper evidence" in context
    assert attached is True
    assert compat._workflow_for_request(payload) == WorkflowId.PDF


@pytest.mark.asyncio
async def test_internal_task_uses_fresh_conversation_id(monkeypatch) -> None:
    captured = {}

    class RuntimeStub:
        settings = SimpleNamespace(pdf_document_context_chars=5000)

        async def execute(self, request, user, *, token_sink=None):
            captured["request"] = request
            captured["user"] = user
            captured["token_sink"] = token_sink
            return request

    runtime = RuntimeStub()
    monkeypatch.setattr(compat, "_runtime", lambda request: runtime)

    original = UUID("00000000-0000-0000-0000-000000000123")
    payload = OpenAIChatRequest(
        model="pdf-document",
        messages=[{"role": "user", "content": "Return a JSON queries object"}],
        metadata={
            "task": "query_generation",
            "files": [{"id": "file-1", "name": "paper.pdf"}],
        },
    )
    result = await compat._execute_request(
        payload=payload,
        request=SimpleNamespace(headers={"X-OpenWebUI-Chat-Id": str(original)}),
        user=SimpleNamespace(user_id="user-1"),
    )

    assert result.workflow == WorkflowId.DIRECT
    assert result.document_context == ""
    assert result.has_document_attachment is False
    assert result.conversation_id != original


def test_retrieval_query_artifacts_are_detected() -> None:
    assert _looks_like_retrieval_query_artifact(
        "assistant",
        '{"queries":["SemioticRAG minhwa"]}',
    )
    assert _looks_like_retrieval_query_artifact(
        "assistant",
        '```json\n{"queries":["SemioticRAG minhwa"]}\n```',
    )
    assert _looks_like_retrieval_query_artifact(
        "user",
        'Generate search queries and return a JSON object with a "queries" field.',
    )
    assert not _looks_like_retrieval_query_artifact(
        "user",
        "What are the paper's main claim and limitations?",
    )


def test_pdf_history_drops_stale_query_generation_turns() -> None:
    state = {
        "messages": [
            HumanMessage(
                content='Generate search queries and return JSON with a "queries" field.'
            ),
            AIMessage(content='{"queries":["SemioticRAG minhwa"]}'),
            HumanMessage(content="What is the paper's main claim?"),
        ],
        "query": "What is the paper's main claim?",
    }

    rendered = _pdf_request_with_history(state)
    assert rendered == "CURRENT REQUEST:\nWhat is the paper's main claim?"
