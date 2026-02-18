from ..base import Tokenizer
from ..configurations import TokenizerConfig
from .tokenizers import HFTokenizer


async def init_tokenizer(config: TokenizerConfig) -> Tokenizer:
    return HFTokenizer(model=config.model)
