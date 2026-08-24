from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..auth import UserContext
from ..config import Settings
from ..gist_regulations import GISTRegulationsRetriever
from ..llm import LLMGateway
from ..routing import StageModelPolicy
from ..schemas import ModelTier, Quality, WorkflowId
from .prompts import (
    DIRECT_SYSTEM,
    PAPER_CONTENT_AGENT_SYSTEM,
    PAPER_DRAFTER_SYSTEM,
    PAPER_FINALIZER_SYSTEM,
    PAPER_ORCHESTRATOR_SYSTEM,
    PAPER_STRUCTURE_AGENT_SYSTEM,
    PAPER_VALIDATOR_SYSTEM,
    REGULATIONS_SYSTEM,
)
from .state import DirectState, PaperState, RegulationsState


@dataclass(slots=True)
class WorkflowServices:
    llm: LLMGateway
    regulations: GISTRegulationsRetriever
    policy: StageModelPolicy
    settings: Settings


@dataclass(frozen=True, slots=True)
class WorkflowSubgraphs:
    direct: Any
    regulations: Any
    paper: Any


class PaperPlan(BaseModel):
    objective: str
    target_section: str = ""
    content_tasks: list[str] = Field(default_factory=list)
    structure_tasks: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class PaperValidation(BaseModel):
    status: str = Field(pattern="^(pass|revise)$")
    issues: list[str] = Field(default_factory=list)
    revision_instructions: str = ""


def _user(state: dict[str, Any]) -> UserContext:
    return UserContext(
        user_id=state["user_id"],
        team_id=state.get("team_id"),
        roles=set(state.get("roles", [])),
    )


def _quality(state: dict[str, Any]) -> Quality:
    return Quality(state.get("quality", Quality.BALANCED.value))


def _workflow(state: dict[str, Any]) -> WorkflowId:
    return WorkflowId(state["workflow_id"])


def _recommended_tier(state: dict[str, Any]) -> ModelTier:
    return ModelTier(state.get("recommended_tier", ModelTier.CLOUD_SMALL.value))


def _stage_alias(state: dict[str, Any], services: WorkflowServices, stage: str) -> str:
    return services.policy.stage_alias(
        workflow=_workflow(state),
        recommended_tier=_recommended_tier(state),
        quality=_quality(state),
        stage=stage,
    )


def _message_text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _conversation(state: dict[str, Any], limit: int = 16, char_limit: int = 24000) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    remaining = char_limit
    for message in reversed(state.get("messages", [])[-limit:]):
        if remaining <= 0:
            break
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        content = _message_text(message).strip()
        if len(content) > remaining:
            content = content[:remaining]
        converted.append({"role": role, "content": content})
        remaining -= len(content)
    converted.reverse()
    return converted


def _request_with_history(state: dict[str, Any]) -> str:
    messages = _conversation(state, limit=8, char_limit=8000)
    blocks = [f"{item['role'].upper()}: {item['content']}" for item in messages[:-1]]
    history = "\n\n".join(blocks)
    request = state["query"][:20000]
    if not history:
        return f"CURRENT REQUEST:\n{request}"
    return f"PRIOR CONVERSATION:\n{history}\n\nCURRENT REQUEST:\n{request}"


def _event(state: dict[str, Any], alias: str, stage: str) -> list[dict[str, str]]:
    return [{"run_id": state["run_id"], "alias": alias, "stage": stage}]


async def _emit_workflow_step(
    state: dict[str, Any],
    services: WorkflowServices,
    *,
    step: int,
    total: int,
    title: str,
    detail: str,
    alias: str,
) -> None:
    """Emit a compact, user-visible workflow progress line during SSE streaming."""
    await services.llm.emit_control(
        run_id=state["run_id"],
        content=(
            f"> **Workflow step {step}/{total} — {title}:** "
            f"{detail}\n\n"
        ),
        event_type="workflow_step",
        requested_alias=alias,
    )


def _answer(answer: str) -> dict[str, Any]:
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


def build_direct_subgraph(services: WorkflowServices):
    async def answer(state: DirectState) -> DirectState:
        alias = _stage_alias(state, services, "answer")
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[{"role": "system", "content": DIRECT_SYSTEM}, *_conversation(state)],
            run_id=state["run_id"],
            workflow_id=WorkflowId.DIRECT.value,
            stage="answer",
            temperature=0.2,
            max_tokens=2400,
        )
        return {**_answer(result.content), "call_events": _event(state, alias, "answer")}

    graph = StateGraph(DirectState)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def build_regulations_subgraph(services: WorkflowServices):
    async def retrieve(state: RegulationsState) -> RegulationsState:
        await _emit_workflow_step(
            state,
            services,
            step=1,
            total=2,
            title="Regulation retrieval",
            detail="searching the GIST regulations vectorstore",
            alias=services.settings.embedding_model_alias,
        )
        sources = await services.regulations.retrieve(
            query=state["query"],
            user=_user(state),
            run_id=state["run_id"],
        )
        context = services.regulations.context_from_sources(sources)
        return {
            "context": context,
            "sources": [source.model_dump(mode="json") for source in sources],
            "retrieval_error": "" if context else "No relevant GIST regulation passages were found.",
            "call_events": _event(state, services.settings.embedding_model_alias, "gist_regulations_query_embedding"),
        }

    async def answer(state: RegulationsState) -> RegulationsState:
        alias = _stage_alias(state, services, "answer")
        if not state.get("context"):
            await _emit_workflow_step(
                state,
                services,
                step=2,
                total=2,
                title="Grounding check",
                detail="no relevant regulation passage was found",
                alias=alias,
            )
            return _answer(state.get("retrieval_error", "No GIST regulation evidence was found."))
        await _emit_workflow_step(
            state,
            services,
            step=2,
            total=2,
            title="Grounded answer",
            detail="writing the answer from the retrieved regulation passages",
            alias=alias,
        )
        prompt = f"{_request_with_history(state)}\n\nGIST REGULATION EVIDENCE:\n{state['context']}"
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": REGULATIONS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.REGULATIONS.value,
            stage="answer",
            temperature=0.0,
            max_tokens=2200,
        )
        return {**_answer(result.content), "call_events": _event(state, alias, "answer")}

    graph = StateGraph(RegulationsState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def build_paper_subgraph(services: WorkflowServices):
    async def orchestrator(state: PaperState) -> PaperState:
        alias = _stage_alias(state, services, "orchestrator")
        await _emit_workflow_step(
            state,
            services,
            step=1,
            total=5,
            title="Orchestrator",
            detail="turning the request into a drafting plan",
            alias=alias,
        )
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": PAPER_ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": _request_with_history(state)},
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.PAPER.value,
            stage="orchestrator",
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        try:
            plan = PaperPlan.model_validate(services.llm.extract_json_object(result.content))
            plan_text = plan.model_dump_json(indent=2)
        except Exception:
            plan_text = result.content
        await _emit_workflow_step(
            state,
            services,
            step=2,
            total=5,
            title="Specialist analysis",
            detail="running the content and structure agents in parallel",
            alias=alias,
        )
        return {"paper_plan": plan_text, "call_events": _event(state, alias, "orchestrator")}

    async def content_agent(state: PaperState) -> PaperState:
        alias = _stage_alias(state, services, "content_agent")
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": PAPER_CONTENT_AGENT_SYSTEM},
                {"role": "user", "content": f"REQUEST:\n{state['query']}\n\nPLAN:\n{state.get('paper_plan','')}"},
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.PAPER.value,
            stage="content_agent",
            temperature=0.1,
            max_tokens=1200,
        )
        return {
            "paper_agent_outputs": [{"run_id": state["run_id"], "agent": "content", "output": result.content}],
            "call_events": _event(state, alias, "content_agent"),
        }

    async def structure_agent(state: PaperState) -> PaperState:
        alias = _stage_alias(state, services, "structure_agent")
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": PAPER_STRUCTURE_AGENT_SYSTEM},
                {"role": "user", "content": f"REQUEST:\n{state['query']}\n\nPLAN:\n{state.get('paper_plan','')}"},
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.PAPER.value,
            stage="structure_agent",
            temperature=0.1,
            max_tokens=1000,
        )
        return {
            "paper_agent_outputs": [{"run_id": state["run_id"], "agent": "structure", "output": result.content}],
            "call_events": _event(state, alias, "structure_agent"),
        }

    async def drafter(state: PaperState) -> PaperState:
        alias = _stage_alias(state, services, "draft")
        await _emit_workflow_step(
            state,
            services,
            step=3,
            total=5,
            title="Drafter",
            detail="synthesizing the plan and both specialist outputs",
            alias=alias,
        )
        outputs = "\n\n".join(
            f"[{item['agent'].upper()} AGENT]\n{item['output']}"
            for item in state.get("paper_agent_outputs", [])
            if item.get("run_id") == state["run_id"]
        )
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": PAPER_DRAFTER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"REQUEST:\n{state['query']}\n\nPLAN:\n{state.get('paper_plan','')}"
                        f"\n\nSUBAGENT OUTPUTS:\n{outputs}"
                    ),
                },
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.PAPER.value,
            stage="draft",
            temperature=0.2,
            max_tokens=2600,
        )
        return {"paper_draft": result.content, "call_events": _event(state, alias, "draft")}

    async def validator(state: PaperState) -> PaperState:
        alias = _stage_alias(state, services, "validator")
        await _emit_workflow_step(
            state,
            services,
            step=4,
            total=5,
            title="Validator",
            detail="checking requested coverage and unsupported claims",
            alias=alias,
        )
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": PAPER_VALIDATOR_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"REQUEST:\n{state['query']}\n\nPLAN:\n{state.get('paper_plan','')}"
                        f"\n\nDRAFT:\n{state.get('paper_draft','')}"
                    ),
                },
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.PAPER.value,
            stage="validator",
            temperature=0.0,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        try:
            validation = PaperValidation.model_validate(services.llm.extract_json_object(result.content))
            validation_text = validation.model_dump_json(indent=2)
        except Exception:
            validation_text = result.content
        return {"paper_validation": validation_text, "call_events": _event(state, alias, "validator")}

    async def final(state: PaperState) -> PaperState:
        alias = _stage_alias(state, services, "final")
        await _emit_workflow_step(
            state,
            services,
            step=5,
            total=5,
            title="Finalizer",
            detail="revising the draft and preparing the final response",
            alias=alias,
        )
        result = await services.llm.chat(
            user=_user(state),
            model_alias=alias,
            messages=[
                {"role": "system", "content": PAPER_FINALIZER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"REQUEST:\n{state['query']}\n\nDRAFT:\n{state.get('paper_draft','')}"
                        f"\n\nVALIDATOR:\n{state.get('paper_validation','')}"
                    ),
                },
            ],
            run_id=state["run_id"],
            workflow_id=WorkflowId.PAPER.value,
            stage="final",
            temperature=0.2,
            max_tokens=2600,
        )
        return {**_answer(result.content), "call_events": _event(state, alias, "final")}

    graph = StateGraph(PaperState)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("content_agent", content_agent)
    graph.add_node("structure_agent", structure_agent)
    graph.add_node("draft", drafter)
    graph.add_node("validator", validator)
    graph.add_node("final", final)
    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "content_agent")
    graph.add_edge("orchestrator", "structure_agent")
    graph.add_edge(["content_agent", "structure_agent"], "draft")
    graph.add_edge("draft", "validator")
    graph.add_edge("validator", "final")
    graph.add_edge("final", END)
    return graph.compile()


def build_workflow_subgraphs(services: WorkflowServices) -> WorkflowSubgraphs:
    return WorkflowSubgraphs(
        direct=build_direct_subgraph(services),
        regulations=build_regulations_subgraph(services),
        paper=build_paper_subgraph(services),
    )
