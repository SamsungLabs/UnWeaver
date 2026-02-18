import asyncio
import logging
from datetime import datetime
from itertools import compress

import numpy as np
from more_itertools import chunked
from tqdm.asyncio import tqdm

from ..base import LOGGER, BaseKVStorage, Embedder
from ..utils import compute_args_hash

L = logging.getLogger(LOGGER)


class EmbedderWrapper(Embedder):

    def __init__(self, embedder: Embedder, **kwargs):
        # pylint: disable=unused-argument
        self._embedder = embedder

        self.model = embedder.model
        self.embedding_dim = embedder.embedding_dim
        self.max_token_size = embedder.max_token_size

    async def __call__(
        self,
        texts: list[str],
        *,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError


class BatchingEmbedderWrapper(EmbedderWrapper):

    def __init__(self, embedder: Embedder, *, batch_size: int):
        super().__init__(embedder)
        self._batch_size = batch_size

    async def __call__(
        self,
        texts: list[str],
        *,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if not texts:
            return np.array([])

        gather_fn = (
            (
                lambda *tasks: tqdm.gather(
                    *tasks,
                    desc=(
                        "embedding: " + kwargs["log_data"]["phase"]
                        if "log_data" in kwargs
                        else ""
                    ),
                )
            )
            if show_progress_bar
            else asyncio.gather
        )
        return np.concatenate(
            await gather_fn(
                *[
                    self._embedder(ts, **kwargs)
                    for ts in chunked(texts, self._batch_size)
                ]
            )
        )


class CacheLoadingEmbedderWrapper(EmbedderWrapper):

    def __init__(self, embedder: Embedder, *, cache: BaseKVStorage):
        super().__init__(embedder)
        self._cache = cache

    async def __call__(
        self,
        texts: list[str],
        *,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if not texts:
            return np.array([])

        embeddings = np.zeros((len(texts), self.embedding_dim))

        hashes = [compute_args_hash(self.model, t) for t in texts]
        if_caches_return = await self._cache.get_by_ids(hashes)

        is_cached = np.array([e is not None for e in if_caches_return])
        if is_cached.any():
            embeddings[is_cached] = [
                e["return"] for e in compress(if_caches_return, is_cached)  # type: ignore
            ]
            L.info("Loaded %i instances from cache", is_cached.sum())

        uncached_texts = list(compress(texts, ~is_cached))

        if not uncached_texts:
            return embeddings

        response_embeddings = await self._embedder(
            uncached_texts,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )
        embeddings[~is_cached] = response_embeddings

        return embeddings


class CacheWritingEmbedderWrapper(EmbedderWrapper):

    def __init__(self, embedder: Embedder, *, cache: BaseKVStorage):
        super().__init__(embedder)
        self._cache = cache

    async def __call__(
        self,
        texts: list[str],
        *,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if not texts:
            return np.array([])

        hashes = [compute_args_hash(self.model, t) for t in texts]

        log_data = {}
        if "log_data" in kwargs:
            log_data.update(kwargs["log_data"])

        response_embeddings = await self._embedder(
            texts,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )

        things_to_upsert = {
            hash: {
                "return": embedding.tolist(),
                "model": self.model,
                "input": text,
                "timestamp": datetime.timestamp(datetime.now()),
                "log_data": log_data,
            }
            for hash, text, embedding in zip(hashes, texts, response_embeddings)
        }
        if len(things_to_upsert) != 0:
            await self._cache.upsert(things_to_upsert)
            await self._cache.index_done_callback()
        return response_embeddings
