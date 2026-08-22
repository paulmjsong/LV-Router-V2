from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import bindparam, text

from .auth import UserContext
from .db import Database, commit
from .schemas import CollectionCreate, CollectionInfo, DocumentInfo, Visibility, WorkflowId


class CollectionRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _select_columns() -> str:
        return (
            "id, name, description, visibility, owner_user_id, team_id, "
            "system_key, created_at"
        )

    async def list_accessible(self, user: UserContext) -> list[CollectionInfo]:
        query = text(
            f"""
            SELECT {self._select_columns()}
            FROM collections
            WHERE owner_user_id = :user_id
               OR visibility = 'public'
               OR (visibility = 'team' AND team_id IS NOT DISTINCT FROM :team_id)
            ORDER BY system_key NULLS LAST, name
            """
        )
        async with self.database.session() as session:
            rows = (
                await session.execute(
                    query,
                    {"user_id": user.user_id, "team_id": user.team_id},
                )
            ).mappings().all()
        return [CollectionInfo(**dict(row)) for row in rows]

    async def create(self, payload: CollectionCreate, user: UserContext) -> CollectionInfo:
        if payload.visibility == Visibility.TEAM and not user.team_id:
            raise HTTPException(status_code=400, detail="A team collection requires a team ID")
        if payload.visibility == Visibility.PUBLIC and "admin" not in user.roles:
            raise HTTPException(status_code=403, detail="Only administrators can create public collections")
        collection_id = uuid4()
        query = text(
            f"""
            INSERT INTO collections(
                id, name, description, visibility, owner_user_id, team_id, system_key
            ) VALUES (
                :id, :name, :description, :visibility, :owner_user_id, :team_id, NULL
            )
            RETURNING {self._select_columns()}
            """
        )
        async with self.database.session() as session:
            row = (
                await session.execute(
                    query,
                    {
                        "id": collection_id,
                        "name": payload.name.strip(),
                        "description": payload.description.strip(),
                        "visibility": payload.visibility.value,
                        "owner_user_id": user.user_id,
                        "team_id": user.team_id if payload.visibility == Visibility.TEAM else None,
                    },
                )
            ).mappings().one()
            await commit(session)
        return CollectionInfo(**dict(row))

    async def ensure_system_collection(
        self,
        *,
        system_key: str,
        name: str,
        description: str,
    ) -> CollectionInfo:
        """Create or update a reserved public collection and return it."""
        query = text(
            f"""
            INSERT INTO collections(
                id, name, description, visibility, owner_user_id, team_id, system_key
            ) VALUES (
                :id, :name, :description, 'public', 'system', NULL, :system_key
            )
            ON CONFLICT (system_key) WHERE system_key IS NOT NULL
            DO UPDATE SET name=EXCLUDED.name, description=EXCLUDED.description
            RETURNING {self._select_columns()}
            """
        )
        async with self.database.session() as session:
            row = (
                await session.execute(
                    query,
                    {
                        "id": uuid4(),
                        "system_key": system_key,
                        "name": name,
                        "description": description,
                    },
                )
            ).mappings().one()
            await commit(session)
        return CollectionInfo(**dict(row))

    async def get_system_collection(self, system_key: str) -> CollectionInfo | None:
        query = text(
            f"""
            SELECT {self._select_columns()}
            FROM collections
            WHERE system_key=:system_key
            """
        )
        async with self.database.session() as session:
            row = (
                await session.execute(query, {"system_key": system_key})
            ).mappings().first()
        return CollectionInfo(**dict(row)) if row is not None else None

    async def get_accessible(self, collection_id: UUID, user: UserContext) -> CollectionInfo:
        query = text(
            f"""
            SELECT {self._select_columns()}
            FROM collections
            WHERE id = :id
              AND (
                    owner_user_id = :user_id
                 OR visibility = 'public'
                 OR (visibility = 'team' AND team_id IS NOT DISTINCT FROM :team_id)
              )
            """
        )
        async with self.database.session() as session:
            row = (
                await session.execute(
                    query,
                    {"id": collection_id, "user_id": user.user_id, "team_id": user.team_id},
                )
            ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail="Collection not found or not accessible")
        return CollectionInfo(**dict(row))

    async def require_write(self, collection_id: UUID, user: UserContext) -> CollectionInfo:
        collection = await self.get_accessible(collection_id, user)
        if collection.system_key is not None:
            if "admin" not in user.roles:
                raise HTTPException(
                    status_code=403,
                    detail="Only administrators can modify a system collection",
                )
            return collection
        is_owner = collection.owner_user_id == user.user_id
        is_team_editor = (
            collection.visibility == Visibility.TEAM
            and collection.team_id == user.team_id
            and user.has_any_role({"editor", "admin"})
        )
        if not is_owner and not is_team_editor:
            raise HTTPException(status_code=403, detail="You cannot modify this collection")
        return collection


class DocumentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def create(
        self,
        *,
        collection_id: UUID,
        filename: str,
        mime_type: str,
        storage_uri: str,
        sha256: str,
        user_id: str,
    ) -> DocumentInfo:
        document_id = uuid4()
        query = text(
            """
            INSERT INTO documents(
                id, collection_id, filename, mime_type, storage_uri, sha256, uploaded_by, status
            ) VALUES (
                :id, :collection_id, :filename, :mime_type, :storage_uri, :sha256, :uploaded_by, 'processing'
            )
            RETURNING id, collection_id, filename, mime_type, status, chunk_count, created_at
            """
        )
        async with self.database.session() as session:
            row = (
                await session.execute(
                    query,
                    {
                        "id": document_id,
                        "collection_id": collection_id,
                        "filename": filename,
                        "mime_type": mime_type,
                        "storage_uri": storage_uri,
                        "sha256": sha256,
                        "uploaded_by": user_id,
                    },
                )
            ).mappings().one()
            await commit(session)
            return DocumentInfo(**dict(row))

    async def insert_chunks(self, document_id: UUID, collection_id: UUID, chunks: list[dict[str, Any]]) -> None:
        statement = text(
            """
            INSERT INTO document_chunks(
                document_id, collection_id, chunk_index, page_number, heading, content, embedding
            ) VALUES (
                :document_id, :collection_id, :chunk_index, :page_number, :heading, :content,
                CAST(:embedding AS vector)
            )
            """
        )
        async with self.database.session() as session:
            for chunk in chunks:
                await session.execute(
                    statement,
                    {
                        "document_id": document_id,
                        "collection_id": collection_id,
                        **chunk,
                    },
                )
            await session.execute(
                text(
                    "UPDATE documents SET status='ready', chunk_count=:count, error_message=NULL WHERE id=:id"
                ),
                {"count": len(chunks), "id": document_id},
            )
            await commit(session)

    async def mark_failed(self, document_id: UUID, message: str) -> None:
        async with self.database.session() as session:
            await session.execute(
                text("UPDATE documents SET status='failed', error_message=:message WHERE id=:id"),
                {"message": message[:2000], "id": document_id},
            )
            await commit(session)

    async def get(self, document_id: UUID) -> DocumentInfo:
        query = text(
            """
            SELECT id, collection_id, filename, mime_type, status, chunk_count, created_at
            FROM documents WHERE id=:id
            """
        )
        async with self.database.session() as session:
            row = (await session.execute(query, {"id": document_id})).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentInfo(**dict(row))


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
