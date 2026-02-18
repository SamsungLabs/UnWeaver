from more_itertools import windowed

from ..base import Tokenizer


class NoopChunker:
    def __init__(self):
        pass

    async def __call__(self, docs: list[str]) -> list[list[str]]:
        return [[doc] for doc in docs]


class NaiveChunker:

    def __init__(
        self, *, tokenizer: Tokenizer, max_token_size: int, overlap_token_size: int
    ):
        self._tokenizer = tokenizer
        self._max_token_size = max_token_size
        self._overlap_token_size = overlap_token_size

    async def __call__(self, docs: list[str]) -> list[list[str]]:
        return [
            self._tokenizer.decode(
                [
                    list(filter(None, window))
                    for window in windowed(
                        doc_tokens,
                        n=self._max_token_size,
                        step=self._max_token_size - self._overlap_token_size,
                    )
                ]
            )
            for doc_tokens in self._tokenizer.encode(docs)
        ]


class MarkerChunker:
    MARKER: str = "[CHUNK_BREAK]"

    def __init__(self):
        pass

    async def __call__(self, docs: list[str]) -> list[list[str]]:
        return [
            [s for s in doc.split(self.MARKER) if len(s) > 0 and not s.isspace()]
            for doc in docs
        ]
