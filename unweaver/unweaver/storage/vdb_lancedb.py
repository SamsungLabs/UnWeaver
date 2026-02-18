import logging
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from more_itertools import chunked

from ..base import LOGGER, Embedder

L = logging.getLogger(LOGGER)

_TBL_NAME = "table"
_PA_TYPES_MAP = {
    "string": pa.string,
    "int_list": lambda: pa.list_(pa.int64()),
    "int": pa.int32,
    "bool": pa.bool8,
}

VECTOR_STORAGE_BATCH_SIZE = 1024


async def _init_lancedb(uri, schema):
    # pylint: disable=unexpected-keyword-arg
    db_connection = lancedb.connect(uri)
    db_table = db_connection.create_table(
        name=_TBL_NAME,
        schema=schema,
        exist_ok=True,
    )
    return db_connection, db_table


@dataclass
class LanceDBStorage:
    db_connection: Any
    db_table: Any

    namespace: str
    meta_fields: set[str]

    embedder: Embedder
    batch_size: int

    @classmethod
    async def in_working_dir(
        cls,
        *,
        working_dir: Path,
        namespace: str,
        embedder: Embedder,
        batch_size: int = VECTOR_STORAGE_BATCH_SIZE,
        meta_fields: dict[str, str] | None = None,
        reset: bool = False,
    ) -> "LanceDBStorage":
        uri = working_dir / f"vdb_{namespace}"
        if reset and uri.exists():
            shutil.rmtree(uri)
            L.info("Cleared vdb %s", namespace)

        schema = [
            pa.field("vector", pa.list_(pa.float32(), embedder.embedding_dim)),
            pa.field("item", pa.string()),
        ]
        if meta_fields is not None:
            schema.extend(
                [pa.field(k, _PA_TYPES_MAP[t]()) for k, t in meta_fields.items()]
            )
        db_connection, db_table = await _init_lancedb(uri, pa.schema(schema))
        return cls(
            db_connection=db_connection,
            db_table=db_table,
            namespace=namespace,
            meta_fields=set() if meta_fields is None else set(meta_fields.keys()),
            embedder=embedder,
            batch_size=batch_size,
        )

    async def upsert(self, data: list[dict]):
        if not data:
            L.warning("No data to upsert to %s", self.namespace)
            return
        L.info("Started upsertion of %d vectors to %s", len(data), self.namespace)

        assert all("item" in dp and "content" in dp for dp in data)

        embeddings = await self.embedder(
            [dp["content"] for dp in data],
            show_progress_bar=True,
            log_data={"phase": f"index_{self.namespace}"},
        )

        data_batches = list(
            chunked(zip(data, embeddings, strict=True), self.batch_size)
        )
        for data_batch in data_batches:
            self._upsert_batch(data_batch)

    def _upsert_batch(self, data: list[tuple[dict, np.ndarray]]):
        data_ = [
            {
                "item": dp["item"],
                "vector": embedding,
                **{k: v for k, v in dp.items() if k in self.meta_fields},
            }
            for dp, embedding in data
        ]
        self.db_table.add(data=data_)

        # removing versioning artifacts
        self.db_table.optimize(cleanup_older_than=timedelta(days=0))

    async def delete(self, ids: set[str]):
        if not ids:
            L.warning("No ids to delete from %s", self.namespace)
            return
        for id_ in ids:
            self.db_table.delete(where=f"item = {repr(id_)}")
        L.info("Deleted %d vectors from %s", len(ids), self.namespace)

    async def query(
        self,
        query: str,
        top_k: int = 5,
        cosine_smaller_than: float = float("inf"),
        meta_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        embedding = (
            await self.embedder(
                [query],
                show_progress_bar=False,
                log_data={"phase": f"query_{self.namespace}"},
            )
        )[0]

        db_query = self.db_table.search(embedding).distance_type("cosine")
        if meta_filter is not None:
            for field, value in meta_filter.items():
                db_query = db_query.where(f"{field} = {repr(value)}")
        results = db_query.limit(top_k).to_pandas()

        results.drop(
            results[results["_distance"] > cosine_smaller_than].index, inplace=True
        )
        results.rename(columns={"item": "id", "_distance": "distance"}, inplace=True)
        results.drop(
            results.columns.difference(["id", "distance", *self.meta_fields]),
            axis="columns",
            inplace=True,
        )
        return results.to_dict(orient="records")

    async def index_start_callback(self):
        pass

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass
