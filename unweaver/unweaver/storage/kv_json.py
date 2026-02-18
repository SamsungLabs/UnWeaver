import logging
import os
from dataclasses import dataclass

from ..base import LOGGER, BaseKVStorage
from ..utils import load_json, write_json

L = logging.getLogger(LOGGER)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        L.debug("Loading KV %s from %s", self.namespace, self._file_name)
        self._data = load_json(self._file_name) or {}
        L.info("Load KV %s with %d data", self.namespace, len(self._data))

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):  # pylint: disable=redefined-builtin
        L.debug("Get by ID %s", id)
        result = self._data.get(id, None)
        if result is None:
            L.warning("No data found for ID %s", id)
            L.warning("Length of _data: %d", len(self._data))
            L.debug("_data: %s", list(self._data.keys()))
        return result

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return {s for s in data if s not in self._data}

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    async def drop(self):
        self._data = {}
