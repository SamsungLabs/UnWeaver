import asyncio
import logging
import random

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from rich.pretty import pretty_repr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..base import LLM, LOGGER
from ..utils import normalize_whitespace

L = logging.getLogger(LOGGER)


class OpenAILLM(LLM):

    def __init__(
        self,
        *,
        model: str,
        urls: list[str],
        keys: list[list[str]],
        timeout: int,
        enable_thinking: bool = False,
    ):
        self._clients = []
        for url, keys_ in zip(urls, keys, strict=True):
            for key in keys_:
                default_headers = None

                self._clients.append(
                    AsyncOpenAI(
                        base_url=url,
                        api_key=key,
                        default_headers=default_headers,
                        timeout=timeout,
                        max_retries=0,
                    )
                )

        self._lock = asyncio.Lock()
        self._client_failures = [0] * len(self._clients)

        self.model = model
        self.enable_thinking = enable_thinking

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, RuntimeError, APIStatusError)
        ),
    )
    async def get_n_completions(
        self,
        conversation: list[dict[str, str]],
        n: int = 1,
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]:
        log_data = {} if log_data is None else log_data

        _ = kwargs.pop("call_id", None)
        extra_body = kwargs.pop("extra_body", {})
        extra_body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        client_index = await self._choose_client_based_on_failures()
        client = self._clients[client_index]
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=conversation,  # type: ignore[arg-type]
                extra_body=extra_body,
                n=n,
                **kwargs,
            )
            if response is None:
                L.error("LLM response is None")
                raise RuntimeError

            if response.usage is not None:
                log_data.update(
                    {
                        "total_tokens": response.usage.total_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                )

            results: list[str] = [
                normalize_whitespace(c.message.content)
                for c in response.choices
                if c.message.content is not None
            ]

            if len(results) == 0:
                L.error("All LLM response contents are None")
                L.error("Response:")
                L.error(pretty_repr(response))
                raise RuntimeError

            return results, log_data
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
