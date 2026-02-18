from pydantic import Field

from . import defaults, sections
from .checkable_config import CheckableConfig
from .literals import ChunkingStrategy


class ChunkerConfig(CheckableConfig):
    strategy: ChunkingStrategy = defaults.CHUNK_STRATEGY
    max_token_size: int = defaults.CHUNK_MAX_TOKEN_SIZE
    overlap_token_size: int = defaults.CHUNK_OVERLAP_TOKEN_SIZE

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class ExtractionConfig(CheckableConfig):
    summarization_threshold: int = defaults.SUMMARIZATION_THRESHOLD

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = [
            "summarization_threshold",
        ]
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class IndexConfig(CheckableConfig):

    # Always keep subconfig field names consistent with sections.py!
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = [
            sections.CHUNKER,
            sections.EXTRACTION,
        ]
        return conflicts, warnings, subconfigs
