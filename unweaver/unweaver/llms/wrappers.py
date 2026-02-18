import re
from datetime import datetime
from re import Pattern

from ..base import LLM, BaseKVStorage
from ..usage_monitors import UsageMonitor
from ..utils import TimeLogger, compute_args_hash, limit_async_func_call_asyncio


class LLMWrapper(LLM):

    def __init__(self, llm: LLM, **kwargs):
        _ = kwargs
        self._llm = llm
        self.model = llm.model

    async def get_n_completions(
        self,
        conversation: list[dict[str, str]],
        n: int = 1,
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]:
        raise NotImplementedError


class CachingLLMWrapper(LLMWrapper):

    def __init__(self, llm: LLM, *, cache: BaseKVStorage):
        super().__init__(llm)
        self._cache = cache

    async def get_n_completions(
        self,
        conversation: list[dict[str, str]],
        n: int = 1,
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]:
        log_data = {} if log_data is None else log_data
        call_id = kwargs.pop("call_id", None)
        hash_ = compute_args_hash(self.model, conversation, n, call_id)
        cached = await self._cache.get_by_id(hash_)
        if cached is not None:
            log_data.update(cached["log_data"])
            return cached["return"], log_data

        responses, log_data = await self._llm.get_n_completions(
            conversation, n, log_data=log_data, **kwargs
        )

        await self._cache.upsert(
            {
                hash_: {
                    "return": responses,
                    "model": self.model,
                    "input": conversation,
                    "timestamp": datetime.timestamp(datetime.now()),
                    "log_data": log_data,
                }
            }
        )
        await self._cache.index_done_callback()

        return responses, log_data


class UsageMonitoringLLMWrapper(LLMWrapper):

    def __init__(
        self,
        llm: LLM,
        *,
        usage_monitor: UsageMonitor,
    ):
        super().__init__(llm)
        self._usage_monitor = usage_monitor

    async def get_n_completions(
        self,
        conversation: list[dict[str, str]],
        n: int = 1,
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]:
        log_data = {} if log_data is None else log_data
        response, log_data = await self._llm.get_n_completions(
            conversation, n, log_data=log_data, **kwargs
        )
        self._usage_monitor.log_call_llm(log_data)
        return response, log_data


class TimeLoggingLLMWrapper(LLMWrapper):

    def __init__(self, llm: LLM, *, log_category: str):
        super().__init__(llm)
        self._get_n_completions = TimeLogger.async_time_log(log_category=log_category)(
            self._llm.get_n_completions
        )

    async def get_n_completions(
        self,
        conversation: list[dict[str, str]],
        n: int = 1,
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]:
        return await self._get_n_completions(
            conversation=conversation, n=n, log_data=log_data, **kwargs
        )


class LimitingLLMWrapper(LLMWrapper):

    def __init__(self, llm: LLM, *, n_max_calls: int):
        super().__init__(llm)
        self._n_max_calls = n_max_calls
        self._get_n_completions = limit_async_func_call_asyncio(n_max_calls)(
            self._llm.get_n_completions
        )

    async def get_n_completions(
        self,
        conversation: list[dict[str, str]],
        n: int = 1,
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]:
        return await self._get_n_completions(
            conversation=conversation, n=n, log_data=log_data, **kwargs
        )


class RegexPostprocessLLMWrapper(LLMWrapper):

    def __init__(self, llm: LLM, *, patterns: dict[Pattern, str]):
        super().__init__(llm)
        self.patterns = patterns

    async def get_n_completions(
        self,
        conversation: list[dict[str, str]],
        n: int = 1,
        *,
        log_data: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]:
        log_data = {} if log_data is None else log_data
        _responses, log_data = await self._llm.get_n_completions(
            conversation, n, log_data=log_data, **kwargs
        )
        responses = []
        for pattern, replacement in self.patterns.items():
            for _response in _responses:
                responses.append(pattern.sub(replacement, _response))
        return responses, log_data


class ThinkingLLMWrapper(
    RegexPostprocessLLMWrapper
):  # pylint: disable=too-few-public-methods
    def __init__(self, llm: LLM):
        super().__init__(
            llm, patterns={re.compile(r"<think>.*</think>", re.DOTALL): ""}
        )
