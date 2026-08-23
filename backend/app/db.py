from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from .config import Settings


class Database:
    """Application persistence only; GIST retrieval stays in the read-only FAISS store."""

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
        statements = [
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
            "CREATE INDEX IF NOT EXISTS ix_runs_user_created ON workflow_runs(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS ix_messages_conversation ON conversation_messages(user_id, conversation_id, id DESC)",
        ]
        async with self.engine.begin() as connection:
            for statement in statements:
                await connection.execute(text(statement))

    async def close(self) -> None:
        await self.engine.dispose()


async def commit(session: AsyncSession) -> None:
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
