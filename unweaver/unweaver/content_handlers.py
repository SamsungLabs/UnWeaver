import json
import logging
from pathlib import Path
from typing import Any

from .base import LOGGER, ContentHandler

L = logging.getLogger(LOGGER)


class DatasetContentHandler(ContentHandler):

    def __init__(self, root_dir: Path):
        self._root_dir = root_dir

    def get_content(self, item_metadata: dict[str, Any]) -> str:
        if item_metadata["type"] == "fs:directory":
            return ""

        if item_metadata["type"] in {"fs:file"}:
            item_path = (
                self._root_dir / item_metadata["id"] / f"{item_metadata['id']}.json"
            )
            assert item_path.exists()

            with open(item_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._read_merge_parts(data["content"])

        if item_metadata["type"] == "ignore":
            # just to be safe
            return ""

        raise RuntimeError

    def get_metadatas(self, item_id: str) -> list[dict[str, Any]]:
        item_path = self._root_dir / item_id / f"{item_id}.json"
        assert item_path.exists()

        with open(item_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert item_id == data["id"]

        metadatas = []

        if data["type"] in {"fs:directory", "fs:file"}:
            metadata = {
                "id": item_id,
                "type": data["type"],
                "path": str(
                    (self._root_dir / data["path"] / data["name"]).relative_to(
                        self._root_dir
                    )
                ),
            }
            metadatas.append(metadata)
            return metadatas

        if data["type"] == "ignore":
            # Fallback for entries with only prescience
            return metadatas

        raise RuntimeError

    def _read_merge_parts(self, parts: list[dict[str, str]]) -> str:
        content = []
        for part in parts:
            with open(
                self._root_dir / part["path_textualized"], "r", encoding="utf-8"
            ) as f:
                content.append(f.read())
        return "\n\n".join(content)
