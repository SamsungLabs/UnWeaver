import json
import logging
import logging.config
import os
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import json_repair

from .literals import LOGGER

L = logging.getLogger(LOGGER)

LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(levelname)s|%(name)s] %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s|%(name)s] %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": None,
            "maxBytes": 10_000_000,
            "backupCount": 3,
        },
    },
    "loggers": {
        "root": {
            "level": "DEBUG",
            "handlers": [
                "stdout",
                "file",
            ],
        },
        "httpx": {
            "level": "WARNING",
        },
    },
}


def init_logging(log_path: Path | None = None):
    if log_path is None:
        log_path_maybe = os.getenv("KGEVAL_LOG_PATH")
        if log_path_maybe is not None:
            log_path = Path(log_path_maybe)
    if log_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        uid = str(uuid4())[:8]
        log_path = Path(f"/tmp/kg_evaluation-{timestamp}-{uid}.log")

    assert log_path is not None
    log_path.parent.mkdir(exist_ok=True, parents=True)
    LOGGING_CONFIG["handlers"]["file"]["filename"] = str(log_path.absolute())
    logging.config.dictConfig(config=LOGGING_CONFIG)


def load_json(filepath: Path) -> list | dict:
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    else:
        L.warning(
            "Attempted to load data from %s but no such file was found, returning empty {}.",
            filepath,
        )
        data = {}
    return data


def dict_recursive_join(base: dict, update: dict) -> dict:
    """
    Note that this is not deepcopy, subdicts may be directly assigned
    """
    keyset = set(base.keys())
    keyset.update(update.keys())
    result = {}
    for key in keyset:
        if key in update:
            if isinstance(update[key], dict) and key in base:
                assert isinstance(base[key], dict)
                result[key] = dict_recursive_join(base[key], update[key])
            else:
                result[key] = update[key]
        else:
            result[key] = base[key]
    return result


def load_dict_from_json_file(filepath: Path) -> dict:
    with open(filepath, encoding="utf-8") as f:
        ret_dict = json.load(f)
    assert isinstance(ret_dict, dict)
    return ret_dict


def parse_json_list(response: str) -> list:
    json_data_maybe = _convert_response_to_json(response)
    if not isinstance(json_data_maybe, list):
        L.warning("Failed to parse JSON from response to list")
        return []
    return json_data_maybe


def parse_json_dict(response: str) -> dict:
    json_data_maybe = _convert_response_to_json(response)
    if not isinstance(json_data_maybe, dict):
        L.warning("Failed to parse JSON from response to dict")
        return {}
    return json_data_maybe


def _convert_response_to_json(response: str):
    """Parse JSON data from response string."""
    L.debug("Started extracting JSON data from response:\n%s", response)
    json_data_maybe = json_repair.loads(response)
    L.debug("Parsed JSON data:\n%s", json_data_maybe)
    return json_data_maybe


def try_eval(_input: str):
    """
    Used in parsing values provided in commandline with --extra parameter
    """
    try:
        return literal_eval(_input)
    except (ValueError, SyntaxError):
        return _input
