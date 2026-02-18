import logging

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ragas.llms import LangchainLLMWrapper

from .configurations import LLMConfig
from .literals import LOGGER

L = logging.getLogger(LOGGER)


def init_llm(llm_config: LLMConfig):
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=llm_config.base_urls[0],
            api_key=SecretStr(llm_config.api_keys[0][0]),
            model=llm_config.model,
            timeout=llm_config.timeout,
        )
    )
    L.info(
        "Initialized LLM %s with url %s",
        llm_config.model,
        llm_config.base_urls[0],
    )
    return llm
