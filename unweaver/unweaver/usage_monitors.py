import json
import logging
from pathlib import Path

from .base import LOGGER

L = logging.getLogger(LOGGER)


class UsageMonitor:

    def __init__(self):
        self._calls_llm: list[dict] = []

    def dump_usage(self, output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._calls_llm, f, indent=2)
        L.info("Saved usage stats to %s", output_path.absolute())

    def log_call_llm(self, log_data: dict):
        self._calls_llm.append(log_data)
