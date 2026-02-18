from ..base import Chunker, Tokenizer
from ..configurations import ChunkerConfig
from .chunkers import MarkerChunker, NaiveChunker, NoopChunker


async def init_chunker(
    config: ChunkerConfig,
    *,
    tokenizer: Tokenizer,
) -> Chunker:
    chunker: Chunker
    if config.strategy == "by_token_size":
        chunker = NaiveChunker(
            tokenizer=tokenizer,
            max_token_size=config.max_token_size,
            overlap_token_size=config.overlap_token_size,
        )
    elif config.strategy == "noop":
        chunker = NoopChunker()
    elif config.strategy == "by_marker":
        chunker = MarkerChunker()
    else:
        raise RuntimeError

    return chunker
