# pylint: disable=no-member

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tqdm.asyncio import tqdm

from .base import (
    LLM,
    LOGGER,
    BaseKVStorage,
    Chunk,
    Chunker,
    ContentHandler,
    Embedder,
    QueryResult,
    StorageNameSpace,
    Tokenizer,
    VectorStorage,
)
from .chunkers import init_chunker
from .configurations import QueryConfig, RAGConfig
from .embedders import init_embedder
from .llms import init_llm
from .retrievers import NaiveRetriever, UnweaverRetriever
from .storage import JsonKVStorage, MongoKVStorage, get_vector_storage
from .tokenizers import init_tokenizer
from .usage_monitors import UsageMonitor
from .utils import (
    TimeLogger,
    build_conversation,
    compute_mdhash_id,
    get_run_id,
    get_unique_id,
    is_retrieval_result_empty,
    limit_async_func_call_asyncio,
    merge_retrieval_results,
    parse_json_list,
    remove_file_if_exists,
)

L = logging.getLogger(LOGGER)

MAX_ASYNC_QUERY = 16

KEY_SET = set(["entity_name", "entity_description"])


class ExtractedEntity(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    chunk_id: list[str]
    model_config = ConfigDict(extra="forbid")


@dataclass
class Unweaver:
    # pylint: disable=too-many-instance-attributes

    id: str
    working_dir: Path
    config: RAGConfig

    prompts: dict[str, str]

    usage_monitor: UsageMonitor | None

    tokenizer: Tokenizer
    chunker: Chunker
    embedder: Embedder
    llm: LLM

    entities_vdb: VectorStorage
    chunks_vdb: VectorStorage

    text_chunks: BaseKVStorage

    entity_base: pd.DataFrame

    unweaver_retriever: UnweaverRetriever
    naive_retriever: NaiveRetriever

    @classmethod
    async def in_working_dir(
        cls,
        working_dir: Path,
        config: RAGConfig,
        reset: bool = False,
        run_name: str | None = None,
    ):
        # pylint: disable=too-many-locals
        if reset:
            remove_file_if_exists(working_dir / "kv_store_text_chunks.json")
            remove_file_if_exists(working_dir / "entities.pqt")

        stage = os.environ.get("GRAPHRAG_STAGE", "_")
        run_id = get_run_id(working_dir, run_name, run_mode=stage)
        usage_monitor = UsageMonitor()

        prompts = {}
        prompt_paths = config.general.prompts.model_dump()
        for key, prompt_path in prompt_paths.items():
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts[key] = f.read()

        tokenizer = await init_tokenizer(config.general.tokenizer)
        chunker = await init_chunker(
            config.index.chunker,
            tokenizer=tokenizer,
        )

        llm_response_cache = (
            await MongoKVStorage.from_environ_credentials(
                namespace="llm_response_cache", run_id=run_id
            )
            if config.general.use_cache_llm
            else None
        )
        llm = await init_llm(
            config.general.llm,
            cache=llm_response_cache,
            usage_monitor=usage_monitor,
        )

        embedder_response_cache = (
            await MongoKVStorage.from_environ_credentials(
                namespace="embedder_response_cache", run_id=run_id
            )
            if config.general.use_cache_embedder
            else None
        )
        embedder = await init_embedder(
            config.general.embedder,
            cache=embedder_response_cache,
        )

        entities_vdb = await get_vector_storage(
            config.general.vector_storage_name
        ).in_working_dir(
            working_dir=working_dir,
            namespace="entities",
            embedder=embedder,
            meta_fields={
                "entity_id": "string",
            },
            reset=reset,
        )
        chunks_vdb = await get_vector_storage(
            config.general.vector_storage_name
        ).in_working_dir(
            working_dir=working_dir,
            namespace="chunks",
            embedder=embedder,
            reset=reset,
        )
        text_chunks = JsonKVStorage(
            namespace="text_chunks",
            global_config={"working_dir": working_dir.absolute()},
        )

        entity_base = (
            pd.read_parquet(working_dir / "entities.pqt")
            if (working_dir / "entities.pqt").exists()
            else pd.DataFrame()
        )

        naive_retriever = NaiveRetriever(
            tokenizer=tokenizer, chunks_vdb=chunks_vdb, text_chunks=text_chunks
        )

        unweaver_retriever = UnweaverRetriever(
            tokenizer=tokenizer,
            entities_vdb=entities_vdb,
            chunks_vdb=chunks_vdb,
            text_chunks=text_chunks,
            entity_base=entity_base,
        )

        return cls(
            id=run_id,
            working_dir=working_dir,
            config=config,
            prompts=prompts,
            usage_monitor=usage_monitor,
            tokenizer=tokenizer,
            chunker=chunker,
            embedder=embedder,
            llm=llm,
            entities_vdb=entities_vdb,
            chunks_vdb=chunks_vdb,
            text_chunks=text_chunks,
            entity_base=entity_base,
            naive_retriever=naive_retriever,
            unweaver_retriever=unweaver_retriever,
        )

    @TimeLogger.async_time_log(log_category="block")
    async def insert(self, input_: Iterable[str], handler: ContentHandler):
        # pylint: disable=too-many-statements

        docs = []
        for item_id in input_:
            for metadata in handler.get_metadatas(item_id):
                docs.append(handler.get_content(metadata))
        L.info("Loaded content for %d documents", len(docs))

        chunks: list[Chunk] = [
            Chunk(
                id=get_unique_id(prefix="chunk-"),
                content=c,
                doc_id=str(i),
            )
            for i, cs in enumerate(await self.chunker(docs))
            for c in cs
        ]
        L.info("Chunked documents into %d chunks", len(chunks))

        inserting_chunks = {
            c.id: {"content": c.content, "id": c.id, "doc_id": c.doc_id} for c in chunks
        }
        L.info("Inserting chunks to KV Storage...")
        await TimeLogger.async_time_log(log_category="block", tag="chunk_upsert")(
            self.text_chunks.upsert
        )(inserting_chunks)
        inserting_chunks_vdb = [
            {"item": c.id, "content": c.content, "id": c.id, "doc_id": c.doc_id}
            for c in chunks
        ]
        L.info("Inserting chunks for VectorRAG...")
        await TimeLogger.async_time_log(log_category="block", tag="vdb_chunk_upsert")(
            self.chunks_vdb.upsert
        )(inserting_chunks_vdb)

        L.info("Entity extraction...")
        results = await tqdm.gather(
            *[self._index_chunk(c) for c in chunks], desc="Entity extraction"
        )
        output_df = pd.DataFrame(
            [e.model_dump() for entities in results for e in entities]
        )
        L.info("Extracted %d entities", len(output_df))

        def join_descriptions(description_list: Iterable[str]) -> str:
            # pylint: disable=unsubscriptable-object
            joined = " ".join((desc for desc in description_list))
            # Trim too long texts
            tokens = self.tokenizer.encode([joined])[0]
            limit = int(0.95 * self.config.general.llm.max_token_size)
            if len(tokens) > limit:
                joined = self.tokenizer.decode([tokens[:limit]])[0]
            return joined

        self.entity_base = output_df.groupby("name").agg(
            {"description": join_descriptions, "chunk_id": "sum"}
        )
        self.entity_base["name"] = self.entity_base.index
        self.entity_base = await self._summarize_descriptions(self.entity_base)
        L.info("Extracted %d unique entities", len(self.entity_base))

        L.info("Upserting entities into VDB...")
        data_for_vdb = [
            {
                "item": compute_mdhash_id(entity_id),
                "content": row["description"],
                "entity_id": entity_id,
            }
            for entity_id, row in self.entity_base.iterrows()
        ]

        await TimeLogger.async_time_log(
            log_category="block", tag="entity_vdb_upsertion"
        )(self.entities_vdb.upsert)(data_for_vdb)

        await self._insert_done()

    async def _insert_done(self):
        self.entity_base.to_parquet(self.working_dir / "entities.pqt")

        storages = [
            self.entities_vdb,
            self.chunks_vdb,
            self.text_chunks,
        ]
        tasks = [
            cast(StorageNameSpace, s).index_done_callback()
            for s in storages
            if s is not None
        ]
        await asyncio.gather(*tasks)

    async def _index_chunk(self, chunk: Chunk) -> list[ExtractedEntity]:
        extraction_prompt = self.prompts["extraction_prompt_template"]
        filled_prompt = extraction_prompt.format(input_text=chunk.content)
        out, _ = await self.llm(
            build_conversation(prompt=filled_prompt),
            log_data={"phase": "entity extraction"},
        )
        if out is None:
            out = ""
        parsed_out = parse_json_list(out)
        output_entities = []
        for dict_ in parsed_out:
            try:
                entity = ExtractedEntity(
                    name=dict_["entity_name"],
                    description=dict_["entity_description"],
                    chunk_id=[chunk.id],
                )
                output_entities.append(entity)
            except (ValidationError, KeyError, TypeError):
                pass
        return output_entities

    @TimeLogger.async_time_log(log_category="block")
    async def _summarize_descriptions(self, entity_base: pd.DataFrame) -> pd.DataFrame:
        entities_to_summarize = entity_base[
            entity_base["description"].apply(len)
            > self.config.index.extraction.summarization_threshold
        ]

        results = await tqdm.gather(
            *[
                self._summarize_single_description(
                    str(entity_id), row["description"], row["chunk_id"]
                )
                for entity_id, row in entities_to_summarize.iterrows()
            ],
            desc="Description summarization",
        )
        summarized_entities = [e.model_dump() for e in results]
        for entity_dict in summarized_entities:
            entity_name = entity_dict["name"]
            description = entity_dict["description"]
            entity_base.loc[entity_name, "description"] = description
        return entity_base

    async def _summarize_single_description(
        self, entity_name: str, entity_description: str, chunk_id: list[str]
    ) -> ExtractedEntity:
        summarization_prompt = self.prompts["summarization_prompt_template"]
        formatted_prompt = summarization_prompt.format(
            entity=entity_name,
            description=entity_description,
        )
        conversation_history = build_conversation(prompt=formatted_prompt)
        out, _ = await self.llm(
            conversation_history, log_data={"phase": "description summarization"}
        )
        if out is None:
            out = ""
        parsed_out = parse_json_list(out)
        try:
            return ExtractedEntity(
                name=entity_name,
                description=parsed_out[0] if len(parsed_out) > 0 else "",
                chunk_id=chunk_id,
            )
        except (ValidationError, IndexError):
            L.warning(
                "Failed to summarize entity description, fallback onto truncation"
            )
            return ExtractedEntity(
                name=entity_name,
                description=entity_description[
                    : self.config.index.extraction.summarization_threshold
                ],
                chunk_id=chunk_id,
            )

    @TimeLogger.async_time_log(log_category="block")
    async def query(
        self,
        queries: list[str],
        params: QueryConfig,
        contexts_gt: list[list[str]] | None = None,
    ) -> list[QueryResult]:
        _ = contexts_gt
        return await tqdm.gather(
            *[self._query_single(q, params) for q in queries], desc="Querying"
        )

    # pylint: disable=too-many-branches
    @limit_async_func_call_asyncio(max_calls=MAX_ASYNC_QUERY)
    @TimeLogger.async_time_log(log_category="block")
    async def _query_single(
        self,
        query: str,
        params: QueryConfig,
    ) -> QueryResult:

        retrieval_methods = params.retrieval_methods

        L.debug("retrieval_methods: %s", retrieval_methods)
        retrieval_results = []
        for retrieval_method in retrieval_methods:
            match retrieval_method:
                case "unweaver":
                    _retrieval_result = await self.unweaver_retriever(
                        query, params.unweaver
                    )
                case "naive":
                    _retrieval_result = await self.naive_retriever(query, params.naive)
                case _:
                    raise RuntimeError(
                        f"Unsupported retrieval method: {retrieval_methods[0]}"
                    )
            _retrieval_result.metadata["retrieval_method"] = retrieval_method
            retrieval_results.append(_retrieval_result)

        retrieval_result = merge_retrieval_results(retrieval_results)

        if is_retrieval_result_empty(retrieval_result) or params.mock_retrieval:
            if not params.mock_retrieval:
                L.warning("Could not retrieve context for query: %s", query)
            return QueryResult(
                answer=self.prompts["no_data_response"],
                retrieval_result=retrieval_result,
                metadata={
                    "retrieval_methods": retrieval_methods,
                },
            )

        querying_prompt = self.prompts["querying_prompt_template"]
        filled_prompt = querying_prompt.format(
            question=query, context="\n".join(retrieval_result.context)
        )

        answer, _ = await self.llm(
            build_conversation(prompt=filled_prompt),
            log_data={"phase": "response generation"},
        )

        return QueryResult(
            answer=answer,
            retrieval_result=retrieval_result,
            metadata={
                "retrieval_methods": retrieval_methods,
            },
        )
