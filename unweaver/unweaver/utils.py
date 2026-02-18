import asyncio
import json
import logging.config
import os
import re
import time
from ast import literal_eval
from functools import wraps
from hashlib import md5
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import json_repair
import pandas

from .base import LOGGER, RetrievalResult, Tokenizer

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
        "file_rotating": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": None,
            "maxBytes": 256_000_000,
            "backupCount": 4,
        },
        "file_last": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": None,
            "mode": "w",
        },
    },
    "loggers": {
        "root": {
            "level": "DEBUG",
            "handlers": [
                "stdout",
                "file_rotating",
                "file_last",
            ],
        },
    },
}


RUN_ID_FILE = "RUN_ID"


def get_unique_id(prefix: str = "") -> str:
    if prefix:
        return f"{prefix}{str(uuid4())}"
    return str(uuid4())


def get_run_id(working_dir: Path, run_name: str | None, run_mode: str = "") -> str:
    run_id_file = working_dir / RUN_ID_FILE
    if run_id_file.exists():
        with open(run_id_file, encoding="utf-8") as f:
            run_id = f.read()
        L.info("Restarting previous %s run with id %s", run_mode, run_id)
    else:
        run_id = get_unique_id()
        with open(run_id_file, "w", encoding="utf-8") as f:
            f.write(run_id)
        L.info("Starting new %s run with id %s", run_mode, run_id)
    if run_name is not None:
        run_id = f"{run_id}-{run_name}"
    run_id = md5(run_id.encode("utf-8")).hexdigest()[:16]
    return run_id


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def init_logging(log_path_rotating: Path, log_path_last: Path, log_level: str = "INFO"):
    log_path_rotating.parent.mkdir(exist_ok=True, parents=True)
    LOGGING_CONFIG["handlers"]["file_rotating"]["filename"] = str(
        log_path_rotating.absolute()
    )

    log_path_last.parent.mkdir(exist_ok=True, parents=True)
    LOGGING_CONFIG["handlers"]["file_last"]["filename"] = str(log_path_last.absolute())

    LOGGING_CONFIG["handlers"]["stdout"]["level"] = log_level.upper()

    logging.config.dictConfig(config=LOGGING_CONFIG)

    # Set the logging level to WARNING to ignore INFO and DEBUG logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if getattr(logging, log_level.upper(), logging.INFO) <= logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("pymongo").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.DEBUG)


def truncate_table_list_by_token_size(
    data: list[Any],
    tokenizer: Tokenizer,
    collate_func: Callable[[Any], str],
    max_token_size: int,
) -> list[Any]:
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, sample in enumerate(data):
        tokens += len(tokenizer.encode([collate_func(sample)])[0])
        if tokens > max_token_size:
            return data[:i]
    return data


def stringify_table_list(
    data: list[list[str]], table_format: str = "csv", mark_format: bool = True
) -> str:
    # Only the header for was given
    if len(data) <= 1:
        return ""

    df = pandas.DataFrame(data[1:], columns=data[0]).set_index("id")
    if table_format == "csv":
        try:
            output = df.to_csv()
        except BaseException:  # pylint: disable=broad-exception-caught
            output = df.to_csv(escapechar="\\")  # csv-incompliant characters
    elif table_format == "html":
        output = df.to_html()
    elif table_format == "latex":
        output = df.to_latex()
    else:
        raise ValueError(f"selected table format '{table_format}' is not available")

    output = output.strip()
    if mark_format:
        output = f"```{table_format}\n{output}\n```"
    return output


def split_by_markers(text: str, markers: list[str]) -> list[str]:
    if not markers:
        return [text]
    results = re.split("|".join(re.escape(marker) for marker in markers), text)
    return [r.strip() for r in results if r.strip()]


_UNICODE_WHITESPACES = [
    "\u0009",  # character tabulation
    # "\u000a", # line feed
    "\u000b",  # line tabulation
    "\u000c",  # form feed
    # "\u000d", # carriage return
    "\u0020",  # space
    # "\u0085", # next line
    "\u00a0",  # no-break space
    "\u1680",  # ogham space mark
    "\u2000",  # en quad
    "\u2001",  # em quad
    "\u2002",  # en space
    "\u2003",  # em space
    "\u2004",  # three-per-em space
    "\u2005",  # four-per-em space
    "\u2006",  # six-per-em space
    "\u2007",  # figure space
    "\u2008",  # punctuation space
    "\u2009",  # thin space
    "\u200A",  # hair space
    # "\u2028", # line separator
    # "\u2029", # paragraph separator
    "\u202f",  # narrow no-break space
    "\u205f",  # medium mathematical space
    "\u3000",  # ideographic space
]
_WHITESPACES_RE = re.compile(r"[" + "".join(_UNICODE_WHITESPACES) + "]")


def normalize_whitespace(string: str) -> str:
    return _WHITESPACES_RE.sub(" ", string)


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


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    if not os.path.exists(file_name):
        L.warning("Tried to load json from %s but it doesn't exist!", file_name)
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def parse_json_list(response: str) -> list:
    json_data_maybe = _convert_response_to_json(response)
    if not isinstance(json_data_maybe, list):
        L.warning(
            "Failed to parse JSON from response to list. Response was:\n%s", response
        )
        return []
    return json_data_maybe


def parse_json_dict(response: str) -> dict:
    json_data_maybe = _convert_response_to_json(response)
    if not isinstance(json_data_maybe, dict):
        L.warning(
            "Failed to parse JSON from response to dict, response was:\n%s", response
        )
        return {}
    return json_data_maybe


def _convert_response_to_json(response: str):
    """Parse JSON data from response string."""
    L.debug("Started extracting JSON data from response:\n%s", response)
    json_data_maybe = json_repair.loads(response)
    L.debug("Parsed JSON data:\n%s", json_data_maybe)
    return json_data_maybe


def build_conversation(
    *,
    prompt: str,
    system_prompt: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    answer_prepend: str | None = None,
) -> list[dict[str, str]]:
    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})
    if conversation_history is not None:
        conversation.extend(conversation_history)
    conversation.append({"role": "user", "content": prompt})
    if answer_prepend is not None:
        conversation.append({"role": "assistant", "content": answer_prepend})
    return conversation


def try_eval(_input: str):
    """
    Used in parsing values provided in commandline with --extra parameter
    """
    try:
        return literal_eval(_input)
    except (ValueError, SyntaxError):
        return _input


def limit_async_func_call_asyncio(max_calls: int):
    semaphore = asyncio.Semaphore(max_calls)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class TimeLogger:
    profiling_on: bool = True
    timing_logs: dict[str, list[dict]] = {}

    @classmethod
    def dump_logs(cls, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls.timing_logs, f, indent=2)

    @classmethod
    def async_time_log(cls, *, log_category: str, tag: str | None = None):
        if log_category not in cls.timing_logs:
            cls.timing_logs[log_category] = []

        def wrapper(func):
            if not cls.profiling_on:
                return func

            @wraps(func)
            async def wrapped(*args, **kwargs):
                start_time = time.perf_counter()
                value = await func(*args, **kwargs)
                end_time = time.perf_counter()
                time_spent = end_time - start_time
                func_name = getattr(func, "__name__", func.__class__.__name__)
                log_data = kwargs.get("log_data", {})
                log_entry = {
                    "func": func_name,
                    "time": time_spent,
                    "tag": tag,
                    "log_data": log_data,
                    "start": start_time,
                    "end": end_time,
                }
                cls.timing_logs[log_category].append(log_entry)
                return value

            return wrapped

        return wrapper

    @classmethod
    def time_log(cls, *, log_category: str, tag: str | None = None):
        def wrapper(func):
            if not cls.profiling_on:
                return func

            @wraps(func)
            def wrapped(*args, **kwargs):
                start_time = time.perf_counter()
                value = func(*args, **kwargs)
                end_time = time.perf_counter()
                time_spent = end_time - start_time
                func_name = func.__name__
                log_data = kwargs.get("log_data", {})
                log_entry = {
                    "func": func_name,
                    "time": time_spent,
                    "tag": tag,
                    "log_data": log_data,
                    "start": start_time,
                    "end": end_time,
                }
                cls.timing_logs[log_category].append(log_entry)
                return value

            return wrapped

        return wrapper


def remove_file_if_exists(file: Path):
    if file.exists():
        os.remove(file)
        L.info("Removed file %s", file.absolute())


def is_retrieval_result_empty(retrieval_result: RetrievalResult) -> bool:
    return (
        not retrieval_result
        or not retrieval_result.context
        or len(retrieval_result.context) == 0
        or all(len(c.strip()) == 0 for c in retrieval_result.context)
    )


def merge_retrieval_results(
    retrieval_results: list[RetrievalResult],
) -> RetrievalResult:
    retrieval_result: RetrievalResult = RetrievalResult(
        context=[], metadata={"retrieval_methods": []}
    )
    for rr in retrieval_results:
        retrieval_result.context.extend(rr.context)
        if "retrieval_method" in rr.metadata:
            retrieval_result.metadata["retrieval_methods"].append(
                rr.metadata["retrieval_method"]
            )
        retrieval_result.metadata.update(rr.metadata)  # pylint: disable=no-member
    return retrieval_result
