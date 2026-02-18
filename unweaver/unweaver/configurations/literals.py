from typing import Literal

# ALL LITERALS HERE ARE ASSUMED TO BE STRING-TYPE!
# If you introduce a different one, correct robust_isinstance
# in checkable_config.py

ChunkingStrategy = Literal["by_token_size", "by_marker", "noop"]

VectorStorageName = Literal["lancedb"]

SystemName = Literal["unweaver"]

RetrievalMethod = Literal[
    "naive",
    "unweaver",
]
