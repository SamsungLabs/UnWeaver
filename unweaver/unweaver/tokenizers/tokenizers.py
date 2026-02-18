from transformers import AutoTokenizer

from ..base import Tokenizer


class ChrTokenizer(Tokenizer):
    def __init__(self):
        self.model = "chr"

    def encode(self, texts: list[str]) -> list[list[int]]:
        return [[ord(c) for c in t] for t in texts]

    def decode(self, tokens: list[list[int]]) -> list[str]:
        return ["".join(chr(t) for t in ts) for ts in tokens]


class HFTokenizer(Tokenizer):
    def __init__(self, *, model):
        self.model = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)

    def encode(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        return self._tokenizer(texts, add_special_tokens=False)["input_ids"]

    def decode(self, tokens: list[list[int]]) -> list[str]:
        if not tokens:
            return []
        return self._tokenizer.batch_decode(tokens)
