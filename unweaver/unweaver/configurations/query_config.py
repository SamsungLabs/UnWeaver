from pydantic import Field

from . import defaults, sections
from .checkable_config import CheckableConfig
from .literals import RetrievalMethod


class NaiveQueryConfig(CheckableConfig):
    retrieve_top_k: int = defaults.NAIVE_RETRIEVE_TOP_K
    chunks_max_token_size: int = defaults.NAIVE_CHUNKS_MAX_TOKEN_SIZE
    chunks_table_format: str = defaults.NAIVE_CHUNKS_TABLE_FORMAT

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class UnweaverQueryConfig(CheckableConfig):
    retrieve_top_k_ents: int = defaults.UNWEAVER_RETRIEVE_TOP_K_ENTS
    retrieve_top_k_chunks: int = defaults.UNWEAVER_RETRIEVE_TOP_K_CHUNKS
    chunks_max_token_size: int = defaults.UNWEAVER_CHUNKS_MAX_TOKEN_SIZE
    chunks_table_format: str = defaults.UNWEAVER_CHUNKS_TABLE_FORMAT

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class QueryConfig(CheckableConfig):
    mock_retrieval: bool = defaults.MOCK_RETRIEVAL
    retrieval_methods: list[RetrievalMethod] = Field(
        default_factory=defaults.RETRIEVAL_METHODS.copy
    )

    # Always keep subconfig field names consistent with sections.py!
    naive: NaiveQueryConfig = Field(default_factory=NaiveQueryConfig)
    unweaver: UnweaverQueryConfig = Field(default_factory=UnweaverQueryConfig)

    def __post_init__(self):
        assert len(self.retrieval_methods) == len(set(self.retrieval_methods))

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = [
            sections.NAIVE,
            sections.UNWEAVER,
        ]
        return conflicts, warnings, subconfigs
