from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

LOGGER = "unweaver"


class Chunk(BaseModel):
    id: str
    content: str
    doc_id: str
    model_config = ConfigDict(extra="forbid")


class RetrievalResult(BaseModel):
    context: list[str]
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class QueryResult(BaseModel):
    answer: str
    retrieval_result: RetrievalResult
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ContentHandler(Protocol):

    def get_content(self, item_metadata: dict[str, Any]) -> str: ...
    def get_metadatas(self, item_id: str) -> list[dict[str, Any]]: ...


class Tokenizer(Protocol):
    model: str

    def encode(self, texts: list[str]) -> list[list[int]]: ...
    def decode(self, tokens: list[list[int]]) -> list[str]: ...


class Chunker(Protocol):

    async def __call__(self, docs: list[str]) -> list[list[str]]: ...


class Embedder(Protocol):
    model: str
    embedding_dim: int
    max_token_size: int

    async def __call__(
        self,
        texts: list[str],
        *,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray: ...


class LLM(Protocol):
    model: str

    async def __call__(
        self,
        conversation: list[dict[str, str]],
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[str, dict]:
        outs, log_data = await self.get_n_completions(
            conversation=conversation,
            n=1,
            log_data=log_data,
            **kwargs,
        )
        return outs[0], log_data

    async def get_n_completions(
        self,
        conversation: list[dict],
        n: int = 1,
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]: ...


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """commit the storage operations after indexing"""

    async def index_done_callback(self):
        """commit the storage operations after indexing"""

    async def query_done_callback(self):
        """commit the storage operations after querying"""


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> T | None:  # pylint: disable=redefined-builtin
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: set[str] | None = None
    ) -> list[T | None]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


class VectorStorage(Protocol):
    namespace: str
    meta_fields: set[str]

    embedder: Embedder

    @classmethod
    async def in_working_dir(
        cls,
        *,
        working_dir: Path,
        namespace: str,
        embedder: Embedder,
        batch_size: int = 1024,
        meta_fields: dict[str, str] | None = None,
        reset: bool = False,
    ) -> "VectorStorage": ...

    async def query(
        self,
        query: str,
        top_k: int = 5,
        cosine_smaller_than: float = 0.8,
        meta_filter: dict[str, Any] | None = None,
    ) -> list[dict]: ...

    async def upsert(self, data: list[dict]): ...
    async def delete(self, ids: set[str]): ...
    async def index_start_callback(self): ...
    async def index_done_callback(self): ...
    async def query_done_callback(self): ...


class Retriever(Protocol):

    async def __call__(self, query: str, params: Any, **kwargs) -> RetrievalResult: ...
