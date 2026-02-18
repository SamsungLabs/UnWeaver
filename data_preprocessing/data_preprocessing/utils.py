import logging
from pathlib import Path
from uuid import uuid4

L = logging.getLogger("rag_preprocessor")


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_unique_id(prefix: str = "") -> str:
    if prefix:
        return f"{prefix}{str(uuid4())}"
    return str(uuid4())


def get_extension(file_path: Path) -> str:
    return file_path.suffix.lower()
