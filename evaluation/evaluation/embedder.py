import logging

from langchain_community.embeddings import InfinityEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from ragas.embeddings import HuggingfaceEmbeddings, LangchainEmbeddingsWrapper

from .configurations import EmbedderConfig
from .literals import LOGGER

L = logging.getLogger(LOGGER)


def init_embedder(embedder_config: EmbedderConfig):
    if embedder_config.mode == "local":
        embedder = HuggingfaceEmbeddings(  # pylint: disable=abstract-class-instantiated
            model_name=embedder_config.model, encode_kwargs={"show_progress_bar": False}
        )
        L.info("Initialized local embeddings %s ", embedder_config.model)
        return embedder
    assert embedder_config.mode == "remote"

    base_url = embedder_config.base_urls[0]
    api_key = SecretStr(embedder_config.api_keys[0][0])

    if base_url.endswith("v1"):
        embedder = OpenAIEmbeddings(
            model=embedder_config.model,
            base_url=base_url,
            api_key=api_key,
        )
        L.info(
            "Initialized remote OpenAI embeddings %s with url %s",
            embedder_config.model,
            base_url,
        )
    else:
        embedder = LangchainEmbeddingsWrapper(
            InfinityEmbeddings(model=embedder_config.model, infinity_api_url=base_url),
        )
        L.info(
            "Initialized remote infinity embeddings %s with url %s",
            embedder_config.model,
            base_url,
        )
    return embedder
