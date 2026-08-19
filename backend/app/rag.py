from __future__ import annotations

from uuid import UUID

from sqlalchemy import bindparam, text

from .auth import UserContext
from .config import Settings
from .db import Database
from .llm import LLMGateway
from .schemas import SourceCitation


class RAGService:
    def __init__(self, settings: Settings, database: Database, llm: LLMGateway) -> None:
        self.settings = settings
        self.database = database
        self.llm = llm

    async def retrieve(
        self,
        *,
        query: str,
        collection_ids: list[UUID],
        user: UserContext,
        run_id: str,
    ) -> list[SourceCitation]:
        retrieval_query = query[: self.settings.rag_query_chars]
        vector = (
            await self.llm.embed_texts(
                user=user,
                texts=[retrieval_query],
                run_id=run_id,
                stage="query_embedding",
            )
        )[0]
        vector_literal = "[" + ",".join(format(value, ".10g") for value in vector) + "]"

        collection_clause = ""
        params: dict = {
            "user_id": user.user_id,
            "team_id": user.team_id,
            "query": retrieval_query,
            "embedding": vector_literal,
            "candidate_k": max(self.settings.rag_top_k * 4, 20),
            "top_k": self.settings.rag_top_k,
        }
        if collection_ids:
            collection_clause = "AND c.collection_id IN :collection_ids"
            params["collection_ids"] = list(collection_ids)

        statement = text(
            f"""
            WITH allowed AS (
                SELECT c.id, c.document_id, c.collection_id, c.page_number, c.content,
                       d.filename,
                       c.embedding
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id AND d.status = 'ready'
                JOIN collections col ON col.id = c.collection_id
                WHERE (
                       col.owner_user_id = :user_id
                    OR col.visibility = 'public'
                    OR (col.visibility = 'team' AND col.team_id IS NOT DISTINCT FROM :team_id)
                )
                {collection_clause}
            ),
            vector_hits AS (
                SELECT id,
                       row_number() OVER (ORDER BY embedding <=> CAST(:embedding AS vector)) AS rank
                FROM allowed
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :candidate_k
            ),
            text_hits AS (
                SELECT id,
                       row_number() OVER (
                           ORDER BY ts_rank_cd(
                               to_tsvector('simple', content),
                               websearch_to_tsquery('simple', :query)
                           ) DESC
                       ) AS rank
                FROM allowed
                WHERE to_tsvector('simple', content) @@ websearch_to_tsquery('simple', :query)
                ORDER BY ts_rank_cd(
                    to_tsvector('simple', content),
                    websearch_to_tsquery('simple', :query)
                ) DESC
                LIMIT :candidate_k
            ),
            fused AS (
                SELECT COALESCE(v.id, t.id) AS id,
                       COALESCE(1.0 / (60 + v.rank), 0.0)
                     + COALESCE(1.0 / (60 + t.rank), 0.0) AS score
                FROM vector_hits v
                FULL OUTER JOIN text_hits t ON t.id = v.id
            )
            SELECT a.id AS chunk_id, a.document_id, a.filename AS title,
                   a.page_number AS page, f.score, a.content
            FROM fused f
            JOIN allowed a ON a.id = f.id
            ORDER BY f.score DESC
            LIMIT :top_k
            """
        )
        if collection_ids:
            statement = statement.bindparams(bindparam("collection_ids", expanding=True))

        async with self.database.session() as session:
            rows = (await session.execute(statement, params)).mappings().all()

        return [
            SourceCitation(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                title=row["title"],
                page=row["page"],
                score=float(row["score"]),
                excerpt=row["content"],
            )
            for row in rows
        ]

    def context_from_sources(self, sources: list[SourceCitation]) -> str:
        parts: list[str] = []
        used = 0
        for index, source in enumerate(sources, start=1):
            label = source.title
            if source.page is not None:
                label += f", page {source.page}"
            block = f"[SOURCE {index}: {label}]\n{source.excerpt}"
            if used + len(block) > self.settings.rag_context_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)
