import logging
import os
from dataclasses import dataclass
from typing import Any

import tqdm.asyncio
from pymongo import AsyncMongoClient, ReplaceOne
from pymongo.errors import ServerSelectionTimeoutError

from ..base import LOGGER, BaseKVStorage
from ..utils import limit_async_func_call_asyncio

L = logging.getLogger(LOGGER)

MONGO_TIMEOUT_MS = 10000
MONGO_MAX_ASYNC_CALLS = 16


@dataclass
class MongoKVStorage(BaseKVStorage):
    run_id: str

    _client: AsyncMongoClient
    _database: Any
    _cache: Any

    @classmethod
    async def from_environ_credentials(cls, namespace, run_id) -> "MongoKVStorage":
        try:
            mongo_user = os.environ["MONGO_USER"]
            mongo_password = os.environ["MONGO_PASSWORD"]
            mongo_url = os.environ["MONGO_URL"]
        except KeyError as exc:
            raise RuntimeError(
                "Mongo credentials were not defined in the environment variables."
            ) from exc

        connection_string = f"mongodb://{mongo_user}:{mongo_password}@{mongo_url}"
        client: AsyncMongoClient = AsyncMongoClient(
            connection_string,
            timeoutms=MONGO_TIMEOUT_MS,
        )
        L.info("Attempting to connect to the Mongo instance...")
        try:
            await client.server_info()
            L.info("Mongo Connection succeeded.")
        except ServerSelectionTimeoutError as exc:
            raise RuntimeError("Failed to connect to the Mongo instance.") from exc

        database = client[run_id]  # use the output_dir hash instead
        cache = database[namespace]
        doc_count = await cache.count_documents({})
        L.info("Load KV %s with %d data", namespace, doc_count)

        return cls(
            namespace=namespace,
            run_id=run_id,
            _client=client,
            _database=database,
            _cache=cache,
            global_config={},  # not used in the case of Mongo
        )

    async def all_keys(self) -> list[str]:
        return await self._cache.distinct("_id")

    async def index_done_callback(self):
        pass

    @limit_async_func_call_asyncio(MONGO_MAX_ASYNC_CALLS)
    async def get_by_id(self, id):  # pylint: disable=redefined-builtin
        return await self._cache.find_one({"_id": id})

    async def get_by_ids(self, ids, fields=None):
        tasks = [self.get_by_id(id) for id in ids]
        results = await tqdm.asyncio.tqdm.gather(*tasks, desc="Querying mongo cache")
        if fields is None:
            return results
        return [
            ({k: v for k, v in result if k in fields} if result else None)
            for result in results
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        cursor = self._cache.find({"_id": {"$in": data}})
        found = [found_doc["_id"] async for found_doc in cursor]
        return {s for s in data if s not in found}

    @limit_async_func_call_asyncio(MONGO_MAX_ASYNC_CALLS)
    async def upsert(self, data: dict[str, dict]):
        # is it actually an upsert or a replacement?
        upserts = [
            ReplaceOne({"_id": key}, val, upsert=True) for key, val in data.items()
        ]
        await self._cache.bulk_write(upserts)

    async def drop(self):
        await self._cache.drop()
