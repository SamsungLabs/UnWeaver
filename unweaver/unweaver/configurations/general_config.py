from pydantic import Field

from . import defaults, sections
from .checkable_config import CheckableConfig
from .literals import SystemName, VectorStorageName


class LLMConfig(CheckableConfig):
    max_async: int = defaults.LLM_MAX_ASYNC
    base_urls: list[str] = defaults.LLM_BASE_URLS
    api_keys: list[list[str]] = defaults.LLM_API_KEYS
    model: str = defaults.LLM_MODEL
    timeout: int = defaults.LLM_TIMEOUT
    max_token_size: int = defaults.LLM_MAX_TOKEN_SIZE
    postprocess_thinking: bool = defaults.LLM_POSTPROCESS_THINKING
    enable_thinking: bool = defaults.LLM_ENABLE_THINKING

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = ["model"]
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class TokenizerConfig(CheckableConfig):
    model: str = defaults.TOKENIZER_MODEL

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = ["model"]
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class EmbedderConfig(CheckableConfig):
    max_async: int = defaults.EMBEDDER_MAX_ASYNC
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


class PromptPathsConfig(CheckableConfig):
    extraction_prompt_template: str = defaults.EXTRACTION_PROMPT
    querying_prompt_template: str = defaults.QUERY_PROMPT
    summarization_prompt_template: str = defaults.SUMMARIZATION_PROMPT
    no_data_response: str = defaults.NO_DATA_RESPONSE

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class GeneralConfig(CheckableConfig):
    use_cache_llm: bool = defaults.USE_CACHE_LLM
    use_cache_embedder: bool = defaults.USE_CACHE_EMBEDDER
    use_which_system: SystemName = defaults.USE_WHICH_SYSTEM

    vector_storage_name: VectorStorageName = defaults.VECTOR_STORAGE_NAME

    remote_mlflow_uri: str = defaults.DEFAULT_REMOTE_MLFLOW_URI
    experiment_name_index: str = defaults.DEFAULT_EXPERIMENT_NAME_INDEX
    experiment_name_query: str = defaults.DEFAULT_EXPERIMENT_NAME_QUERY

    # Always keep subconfig field names consistent with sections.py!
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    prompts: PromptPathsConfig = Field(default_factory=PromptPathsConfig)

    @staticmethod
    def get_checks():
        conflicts: list[str] = [
            "vector_storage_name",
        ]
        warnings: list[str] = []
        subconfigs: list[str] = [
            sections.LLM,
            sections.TOKENIZER,
            sections.EMBEDDER,
            sections.PROMPTS,
        ]
        return conflicts, warnings, subconfigs
