import json
import logging
import os
from pathlib import Path
from pprint import pformat

from pydantic import Field

from ..literals import LOGGER
from ..utils import dict_recursive_join, load_dict_from_json_file, try_eval
from . import sections
from .checkable_config import CheckableConfig
from .defaults import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_KEY,
    DEFAULT_METRICS,
    DEFAULT_NUM_REPEATS,
    DEFAULT_REMOTE_MLFLOW_URI,
)
from .general_config import EmbedderConfig, LLMConfig

__all__ = [
    "EmbedderConfig",
    "LLMConfig",
    "EvalConfig",
    "get_main_section",
]

L = logging.getLogger(LOGGER)
L.setLevel(logging.DEBUG)


def get_main_section(word: str) -> str:
    if not word:
        return ""
    for keyword in (sections.EMBEDDER, sections.LLM):
        if keyword.startswith(word):
            return keyword
    return ""


class EvalConfig(CheckableConfig):

    remote_mlflow_uri: str = DEFAULT_REMOTE_MLFLOW_URI
    experiment_name: str = DEFAULT_EXPERIMENT_NAME

    num_repeats: int = DEFAULT_NUM_REPEATS
    metrics: list[str] = DEFAULT_METRICS

    # Always keep subconfig field names consistent with sections.py!
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = [
            sections.LLM,
            sections.EMBEDDER,
        ]
        return conflicts, warnings, subconfigs


def parse_extra(base_dict: dict, extra_config: list[str]) -> dict:
    for ec_field in extra_config:
        # colon can be part of the value - only the first one is a delimeter between keys and value
        keys_, value_ = ec_field.strip().split(":", 1)
        keys = keys_.split(".")
        value = try_eval(value_)
        section = get_main_section(keys[0])
        if not section:
            raise RuntimeError
        if section not in base_dict:
            base_dict[section] = {}
        match len(keys):
            case 3:
                if keys[1] not in base_dict[section]:
                    base_dict[section][keys[1]] = {}
                base_dict[section][keys[1]][keys[2]] = value
            case 2:
                base_dict[section][keys[1]] = value
            case _:
                raise RuntimeError
    return base_dict


def get_envvar_config() -> dict:
    config_dict: dict = {}
    env_llm_api_key = os.environ.get("LLM_API_KEY", None)
    if env_llm_api_key:
        config_dict = dict_recursive_join(
            config_dict, {"general": {"llm": {"api_key": env_llm_api_key}}}
        )

    env_embedder_api_key = os.environ.get("EMBEDDER_API_KEY", None)
    if env_embedder_api_key:
        config_dict = dict_recursive_join(
            config_dict, {"general": {"embedder": {"api_key": env_llm_api_key}}}
        )
    return config_dict


def get_sanitized_config(config: EvalConfig) -> EvalConfig:
    # cleans the config of keys we are not willing to dump
    config = config.copy(deep=True)
    config.general.llm.api_keys = [[DEFAULT_KEY] for _ in config.general.llm.base_urls]
    config.general.embedder.api_keys = [
        [DEFAULT_KEY] for _ in config.general.embedder.base_urls
    ]
    return config


def prepare_config(
    *,
    base_config_path: Path | None,
    reconfig_paths: list[Path] | None,
    extra_config: list[str] | None,
    output_path: Path | None,
    strict_config_check: bool = False,
) -> EvalConfig:
    # Note that output_path is NOT written to until base_config_path
    # is loaded, so the configuration will not be overwritten.
    # Both parameters being the same is the typical use case when indexing.
    base_dict: dict = {}
    if base_config_path is not None:
        if not base_config_path.exists():
            raise ValueError(f"Invalid base configuration path {base_config_path}")
        base_dict = load_dict_from_json_file(base_config_path)
        # Add defaults if base_config is incomplete:
        config = EvalConfig.model_validate(base_dict)

        base_dict = config.model_dump()
        L.debug(
            "Loaded config from %s, config:\n%s",
            base_config_path.absolute(),
            pformat(base_dict),
        )
    else:
        L.debug("No base config, proceeding with default")

    reconfig_dict: dict = (
        get_envvar_config()
    )  # envvars have priority over defaults, but yield to user config
    for reconfig_path in reconfig_paths or []:
        reconfig_dict = dict_recursive_join(
            reconfig_dict, load_dict_from_json_file(reconfig_path)
        )
        L.info(
            "Loaded reconfig options from %s, updated reconfig:\n%s",
            reconfig_path.absolute(),
            pformat(reconfig_dict),
        )

    if extra_config is not None:
        reconfig_dict = parse_extra(reconfig_dict, extra_config)
        L.debug("Extended reconfig with extra parameters:\n%s", pformat(reconfig_dict))

    configured, problems = EvalConfig.check_override(
        reference=base_dict,
        override=reconfig_dict,
        strict_override_check=strict_config_check,
    )

    if not configured:
        L.error(
            "Attempted reconfiguration malformed or incompatible with existing index, reasons:\n%s",
            pformat(problems),
        )
        raise RuntimeError

    config_dict = dict_recursive_join(base_dict, reconfig_dict)
    config = EvalConfig.model_validate(config_dict)

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                get_sanitized_config(config).model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )
        L.debug("Saved configuration to working directory: %s", output_path.absolute())
    return config
