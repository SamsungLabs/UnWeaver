from typing import Type

from ..base import VectorStorage
from ..configurations.literals import VectorStorageName
from .kv_json import JsonKVStorage
from .kv_mongo import MongoKVStorage
from .vdb_lancedb import LanceDBStorage

__all__ = [
    "JsonKVStorage",
    "LanceDBStorage",
    "MongoKVStorage",
]


def get_vector_storage(vector_storage_name: VectorStorageName) -> Type[VectorStorage]:
    return {
        "lancedb": LanceDBStorage,
    }[vector_storage_name]
