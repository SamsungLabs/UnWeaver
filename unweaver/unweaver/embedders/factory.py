from ..base import BaseKVStorage, Embedder
from ..configurations import EmbedderConfig
from ..utils import TimeLogger, limit_async_func_call_asyncio
from .embedders import OpenAIEmbedder
from .wrappers import (
    BatchingEmbedderWrapper,
    CacheLoadingEmbedderWrapper,
    CacheWritingEmbedderWrapper,
)


async def init_embedder(
    config: EmbedderConfig,
    *,
    cache: BaseKVStorage | None = None,
) -> Embedder:
    embedder: Embedder = OpenAIEmbedder(
        model=config.model,
        embedding_dim=config.embedding_dim,
        max_token_size=config.max_token_size,
        urls=config.base_urls,
        keys=config.api_keys,
        timeout=config.timeout,
    )

    embedder = TimeLogger.async_time_log(log_category="embedding")(embedder)
    embedder = limit_async_func_call_asyncio(config.max_async)(embedder)

    if cache is not None:
        embedder = CacheWritingEmbedderWrapper(
            embedder=embedder,
            cache=cache,
        )

    embedder = BatchingEmbedderWrapper(
        embedder=embedder,
        batch_size=config.batch_size,
    )

    if cache is not None:
        embedder = CacheLoadingEmbedderWrapper(
            embedder=embedder,
            cache=cache,
        )

    return embedder
