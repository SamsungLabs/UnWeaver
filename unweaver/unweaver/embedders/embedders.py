import asyncio
import logging
import random

import numpy as np
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..base import LOGGER, Embedder

L = logging.getLogger(LOGGER)


class OpenAIEmbedder(Embedder):

    def __init__(
        self,
        *,
        model: str,
        embedding_dim: int,
        max_token_size: int,
        urls: list[str],
        keys: list[list[str]],
        timeout: int,
    ):
        self._clients = []
        for url, keys_ in zip(urls, keys, strict=True):
            for key in keys_:
                self._clients.append(
                    AsyncOpenAI(
                        base_url=url,
                        api_key=key,
                        timeout=timeout,
                        max_retries=0,
                    )
                )

        self._lock = asyncio.Lock()
        self._client_failures = [0] * len(self._clients)

        self.model = model

        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, RuntimeError)
        ),
    )
    async def __call__(
        self,
        texts: list[str],
        *,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        _ = show_progress_bar
        _ = kwargs

        if not texts:
            return np.array([])

        client_index = await self._choose_client_based_on_failures()
        client = self._clients[client_index]

        try:
            response = await client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float",
            )
            if response is None:
                raise RuntimeError

            return np.array([x.embedding for x in response.data])
        except APIConnectionError as e:
            async with self._lock:
                self._client_failures[client_index] += 1
            raise e

    async def _choose_client_based_on_failures(self) -> int:
        async with self._lock:
            weights = [1 / (1 + count) for count in self._client_failures]
            client_index = random.choices(
                range(len(self._clients)), weights=weights, k=1
            )[0]
            return client_index
