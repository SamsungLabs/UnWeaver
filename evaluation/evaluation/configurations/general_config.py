from . import defaults
from .checkable_config import CheckableConfig
from .literals import EmbedderMode


class LLMConfig(CheckableConfig):
    # Lists are used for compatibility with dualgraphrag
    # Only the first elements are used, rest is ignored
    base_urls: list[str] = defaults.LLM_BASE_URLS
    api_keys: list[list[str]] = defaults.LLM_API_KEYS

    model: str = defaults.LLM_MODEL
    timeout: int = defaults.LLM_TIMEOUT
    max_retries: int = defaults.LLM_MAX_RETRIES
    max_workers: int = defaults.LLM_MAX_WORKERS

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = ["model"]
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class EmbedderConfig(CheckableConfig):
    max_async: int = defaults.EMBEDDER_MAX_ASYNC
    mode: EmbedderMode = defaults.EMBEDDER_MODE

    # Lists are used for compatibility with
    # https://github.sec.samsung.net/SAIC-Warsaw/nano-graphrag
    # Only the first elements are used, rest is ignored
    base_urls: list[str] = defaults.EMBEDDER_BASE_URLS
    api_keys: list[list[str]] = defaults.EMBEDDER_API_KEYS

    model: str = defaults.EMBEDDER_MODEL
    embedding_dim: int = defaults.EMBEDDER_DIM
    max_token_size: int = defaults.EMBEDDER_MAX_TOKEN_SIZE
    batch_size: int = defaults.EMBEDDER_BATCH_SIZE
    timeout: int = defaults.EMBEDDER_TIMEOUT

    @staticmethod
    def get_checks():
        conflicts: list[str] = ["model", "embedding_dim"]
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs
