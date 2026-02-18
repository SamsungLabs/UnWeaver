from ..base import LLM, BaseKVStorage
from ..configurations import LLMConfig
from ..usage_monitors import UsageMonitor
from .llms import OpenAILLM
from .wrappers import (
    CachingLLMWrapper,
    LimitingLLMWrapper,
    ThinkingLLMWrapper,
    TimeLoggingLLMWrapper,
    UsageMonitoringLLMWrapper,
)


async def init_llm(
    config: LLMConfig,
    *,
    cache: BaseKVStorage | None = None,
    usage_monitor: UsageMonitor | None = None,
) -> LLM:
    llm: LLM = OpenAILLM(
        model=config.model,
        urls=config.base_urls,
        keys=config.api_keys,
        timeout=config.timeout,
        enable_thinking=config.enable_thinking,
    )

    llm = TimeLoggingLLMWrapper(llm=llm, log_category="llm")
    llm = LimitingLLMWrapper(llm=llm, n_max_calls=config.max_async)

    if cache is not None:
        llm = CachingLLMWrapper(
            llm=llm,
            cache=cache,
        )

    if usage_monitor is not None:
        llm = UsageMonitoringLLMWrapper(llm=llm, usage_monitor=usage_monitor)

    if config.postprocess_thinking:
        llm = ThinkingLLMWrapper(llm=llm)

    return llm
