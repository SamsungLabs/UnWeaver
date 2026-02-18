import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from itertools import chain

import pandas as pd

from ..base import (
    LOGGER,
    BaseKVStorage,
    RetrievalResult,
    Retriever,
    Tokenizer,
    VectorStorage,
)
from ..configurations import NaiveQueryConfig, UnweaverQueryConfig
from ..utils import stringify_table_list, truncate_table_list_by_token_size

L = logging.getLogger(LOGGER)


class BuildContextFromChunks:

    def __init__(self, *, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    async def __call__(
        self,
        chunks: list[dict],
        params,
    ) -> tuple[str, list[dict]]:
        context_parts: list[list[str]] = [
            ["id", "content"],
        ]
        for i, chunk in enumerate(chunks):
            context_parts.append(
                [
                    str(i),
                    chunk["content"],
                ]
            )
        context_parts = truncate_table_list_by_token_size(
            data=context_parts,
            tokenizer=self.tokenizer,
            collate_func=", ".join,
            max_token_size=params.chunks_max_token_size,
        )
        context = stringify_table_list(
            data=context_parts,
            table_format=params.chunks_table_format,
            mark_format=True,
        )
        return context, chunks[: len(context_parts) - 1]


class RetrieveRelevantChunks:

    def __init__(self, *, chunks_vdb: VectorStorage, chunks_db: BaseKVStorage):
        self.chunks_vdb = chunks_vdb
        self.chunks_db = chunks_db

    async def __call__(self, query: list[str], params) -> list[dict]:
        """

        Args:
            query: In enriched format, so a list with query and optionally
                   relevant entities.

        """
        chunk_ids_dists_ = await asyncio.gather(
            *[
                self.chunks_vdb.query(
                    query=q,
                    top_k=params.retrieve_top_k,
                    meta_filter=None,
                )
                for q in query
            ]
        )
        L.debug("Retrieved %d chunks (before deduplication)", len(chunk_ids_dists_))
        chunk_ids_dists = self._deduplicate_retrieved_chunks(
            chunks=list(chain(*chunk_ids_dists_)),
            top_k=params.retrieve_top_k,
        )
        L.debug("Retrieved %d chunks (after deduplication)", len(chunk_ids_dists))

        # Retrieve chunks from key-value storage
        chunks = await asyncio.gather(
            *[self.chunks_db.get_by_id(c["id"]) for c in chunk_ids_dists]
        )
        for chunk, chunk_id_dist in zip(chunks, chunk_ids_dists, strict=True):
            assert chunk is not None
            if "id" in chunk:
                assert chunk["id"] == chunk_id_dist["id"]
            else:
                chunk["id"] = chunk_id_dist["id"]
            chunk["distance"] = chunk_id_dist["distance"]

        return chunks  # type: ignore[return-value]

    def _deduplicate_retrieved_chunks(
        self, chunks: list[dict], top_k: int
    ) -> list[dict]:
        # Handle deduplication after using multiple alternative queries
        df = pd.DataFrame.from_records(chunks)
        df = df.groupby(["id"], as_index=False).min()
        df = df.sort_values(by="distance", ascending=True)  # type: ignore
        return df.to_dict(orient="records")[:top_k]


@dataclass
class NaiveRetriever(Retriever):

    tokenizer: Tokenizer
    chunks_vdb: VectorStorage
    text_chunks: BaseKVStorage

    async def __call__(
        self, query: str, params: NaiveQueryConfig, **kwargs
    ) -> RetrievalResult:
        context = {}

        chunks = await RetrieveRelevantChunks(
            chunks_vdb=self.chunks_vdb, chunks_db=self.text_chunks
        )([query], params)

        context["Sources"], used_chunks = await BuildContextFromChunks(
            tokenizer=self.tokenizer
        )(chunks, params)

        return RetrievalResult(
            context=[
                "\n".join(f"-----{k}-----\n{v}\n" for k, v in context.items() if v)
            ],
            metadata={
                "anchor_chunks": [c["id"] for c in used_chunks],
            },
        )


@dataclass
class UnweaverRetriever(Retriever):
    tokenizer: Tokenizer
    entities_vdb: VectorStorage
    chunks_vdb: VectorStorage
    text_chunks: BaseKVStorage
    entity_base: pd.DataFrame

    async def __call__(
        self, query: str, params: UnweaverQueryConfig, **kwargs
    ) -> RetrievalResult:
        entity_ids = [
            e["entity_id"]
            for e in await self.entities_vdb.query(
                query=query,
                top_k=params.retrieve_top_k_ents,
            )
        ]
        entities = self.entity_base.loc[entity_ids].to_dict(orient="records")

        chunk_ids: list[str] = self._select_chunks(
            entities,
            chunks_top_k=params.retrieve_top_k_chunks,
            entities_top_k=params.retrieve_top_k_ents,
        )
        chunks: list[dict] = []
        for pos, chunk in enumerate(await self.text_chunks.get_by_ids(chunk_ids)):
            assert isinstance(chunk, dict)
            chunk["normalized_order"] = pos / len(chunk_ids)
            chunks.append(chunk)

        context, used_chunks = await BuildContextFromChunks(tokenizer=self.tokenizer)(
            chunks,
            params,
        )

        anchor_chunks = [c["id"] for c in used_chunks]
        anchor_nodes = [
            e["name"]
            for e in entities
            if any(x in anchor_chunks for x in e["chunk_id"])
        ]

        return RetrievalResult(
            context=[context],
            metadata={
                "anchor_chunks": anchor_chunks,
                "anchor_nodes": anchor_nodes,
            },
        )

    def _select_chunks(
        self, entities: list[dict], chunks_top_k: int, entities_top_k: int
    ) -> list[str]:
        chunk_counter = Counter(
            [chunk_id for e in entities for chunk_id in e["chunk_id"]]
        )
        chunk_scores = {}
        for entity_pos, entity in enumerate(entities):
            for chunk_id in entity["chunk_id"]:
                if chunk_id in chunk_scores:
                    continue
                chunk_scores[chunk_id] = (
                    chunk_counter[chunk_id]
                    / sum(chunk_counter.values())
                    * (1 - (entity_pos / entities_top_k))
                )
        return [
            c for c, _ in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        ][:chunks_top_k]
