from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from .config import Settings

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: AsyncEngine = create_async_engine(
            settings.sqlalchemy_database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self):
        async with self.session_factory() as session:
            yield session

    async def setup(self) -> None:
        dimension = int(self.settings.embedding_dimensions)
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            """
            CREATE TABLE IF NOT EXISTS collections (
                id UUID PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                visibility TEXT NOT NULL CHECK (visibility IN ('private', 'team', 'public')),
                owner_user_id TEXT NOT NULL,
                team_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                storage_uri TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                uploaded_by TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'processing',
                error_message TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER,
                heading TEXT,
                content TEXT NOT NULL,
                embedding VECTOR({dimension}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(document_id, chunk_index)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS workflow_runs (
                id UUID PRIMARY KEY,
                conversation_id UUID NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                team_id TEXT,
                workflow_id TEXT NOT NULL,
                route_reason TEXT NOT NULL,
                quality TEXT NOT NULL,
                status TEXT NOT NULL,
                answer TEXT,
                pending_action JSONB,
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id UUID NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                run_id UUID NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            "CREATE INDEX IF NOT EXISTS ix_collections_team ON collections(team_id)",
            "CREATE INDEX IF NOT EXISTS ix_documents_collection ON documents(collection_id)",
            "CREATE INDEX IF NOT EXISTS ix_chunks_collection ON document_chunks(collection_id)",
            "CREATE INDEX IF NOT EXISTS ix_chunks_text ON document_chunks USING gin(to_tsvector('simple', content))",
            "CREATE INDEX IF NOT EXISTS ix_runs_user_created ON workflow_runs(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS ix_messages_conversation ON conversation_messages(user_id, conversation_id, id DESC)",
        ]
        async with self.engine.begin() as connection:
            for statement in statements:
                await connection.execute(text(statement))

        # HNSW may be unavailable on older pgvector builds. Use a separate transaction so
        # an unsupported index method cannot roll back the schema creation above.
        try:
            async with self.engine.begin() as connection:
                await connection.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
                    )
                )
        except Exception:
            logger.warning("HNSW index creation failed; pgvector search will use an exact scan", exc_info=True)

    async def close(self) -> None:
        await self.engine.dispose()


async def commit(session: AsyncSession) -> None:
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
