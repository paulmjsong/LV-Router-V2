from __future__ import annotations

from uuid import UUID

from sqlalchemy import text

from .auth import UserContext
from .db import Database, commit
from .schemas import WorkflowId


class ConversationRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def list_recent(
        self,
        *,
        conversation_id: UUID,
        user_id: str,
        limit: int,
    ) -> list[dict[str, str]]:
        query = text(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM conversation_messages
                WHERE conversation_id=:conversation_id AND user_id=:user_id
                ORDER BY id DESC
                LIMIT :limit
            ) recent
            ORDER BY id ASC
            """
        )
        async with self.database.session() as session:
            rows = (
                await session.execute(
                    query,
                    {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "limit": limit,
                    },
                )
            ).mappings().all()
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in rows]

    async def append(
        self,
        *,
        conversation_id: UUID,
        user_id: str,
        role: str,
        content: str,
        workflow_id: WorkflowId,
        run_id: UUID,
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported conversation role: {role}")
        async with self.database.session() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO conversation_messages(
                        conversation_id, user_id, role, content, workflow_id, run_id
                    ) VALUES (
                        :conversation_id, :user_id, :role, :content, :workflow_id, :run_id
                    )
                    """
                ),
                {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "role": role,
                    "content": content,
                    "workflow_id": workflow_id.value,
                    "run_id": run_id,
                },
            )
            await commit(session)

    async def append_turn(
        self,
        *,
        conversation_id: UUID,
        user_id: str,
        query: str,
        answer: str,
        workflow_id: WorkflowId,
        run_id: UUID,
    ) -> None:
        async with self.database.session() as session:
            statement = text(
                """
                INSERT INTO conversation_messages(
                    conversation_id, user_id, role, content, workflow_id, run_id
                ) VALUES (
                    :conversation_id, :user_id, :role, :content, :workflow_id, :run_id
                )
                """
            )
            common = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "workflow_id": workflow_id.value,
                "run_id": run_id,
            }
            await session.execute(statement, {**common, "role": "user", "content": query})
            await session.execute(statement, {**common, "role": "assistant", "content": answer})
            await commit(session)


class RunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def create(
        self,
        *,
        run_id: UUID,
        conversation_id: UUID,
        thread_id: str,
        user: UserContext,
        workflow_id: WorkflowId,
        route_reason: str,
        quality: str,
    ) -> None:
        query = text(
            """
            INSERT INTO workflow_runs(
                id, conversation_id, thread_id, user_id, team_id, workflow_id,
                route_reason, quality, status
            ) VALUES (
                :id, :conversation_id, :thread_id, :user_id, :team_id, :workflow_id,
                :route_reason, :quality, 'running'
            )
            """
        )
        async with self.database.session() as session:
            await session.execute(
                query,
                {
                    "id": run_id,
                    "conversation_id": conversation_id,
                    "thread_id": thread_id,
                    "user_id": user.user_id,
                    "team_id": user.team_id,
                    "workflow_id": workflow_id.value,
                    "route_reason": route_reason,
                    "quality": quality,
                },
            )
            await commit(session)

    async def update_route(
        self,
        run_id: UUID,
        *,
        workflow_id: WorkflowId,
        route_reason: str,
    ) -> None:
        async with self.database.session() as session:
            await session.execute(
                text(
                    """
                    UPDATE workflow_runs
                    SET workflow_id=:workflow_id, route_reason=:route_reason, updated_at=now()
                    WHERE id=:id
                    """
                ),
                {
                    "id": run_id,
                    "workflow_id": workflow_id.value,
                    "route_reason": route_reason,
                },
            )
            await commit(session)

    async def update(
        self,
        run_id: UUID,
        *,
        status: str,
        answer: str | None = None,
        error_message: str | None = None,
    ) -> None:
        query = text(
            """
            UPDATE workflow_runs
            SET status=:status,
                answer=COALESCE(:answer, answer),
                error_message=:error_message,
                updated_at=now()
            WHERE id=:id
            """
        )
        async with self.database.session() as session:
            await session.execute(
                query,
                {
                    "id": run_id,
                    "status": status,
                    "answer": answer,
                    "error_message": error_message,
                },
            )
            await commit(session)
