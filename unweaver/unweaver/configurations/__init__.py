import json
import logging
import os
from pathlib import Path
from pprint import pformat

from pydantic import Field

from ..base import LOGGER
from ..utils import dict_recursive_join, load_dict_from_json_file, try_eval
from . import sections
from .checkable_config import CheckableConfig
from .defaults import DEFAULT_KEY
from .general_config import (
    EmbedderConfig,
    GeneralConfig,
    LLMConfig,
    PromptPathsConfig,
    TokenizerConfig,
)
from .index_config import ChunkerConfig, ExtractionConfig, IndexConfig
from .literals import RetrievalMethod
from .query_config import NaiveQueryConfig, QueryConfig, UnweaverQueryConfig

__all__ = [
    "ChunkerConfig",
    "EmbedderConfig",
    "ExtractionConfig",
    "GeneralConfig",
    "get_main_section",
    "RAGConfig",
    "IndexConfig",
    "LLMConfig",
    "NaiveQueryConfig",
    "PromptPathsConfig",
    "QueryConfig",
    "RetrievalMethod",
    "TokenizerConfig",
    "UnweaverQueryConfig",
]

L = logging.getLogger(LOGGER)

EXPERIMENTAL_PREFIX = "DGR_EXP_"


def get_main_section(word: str) -> str:
    if not word:
        return ""
    for keyword in (sections.GENERAL, sections.QUERY, sections.INDEX):
        if keyword.startswith(word):
            return keyword
    return ""


class RAGConfig(CheckableConfig):
    # Always keep subconfig field names consistent with sections.py!
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = [sections.GENERAL, sections.QUERY, sections.INDEX]
        return conflicts, warnings, subconfigs


def parse_extra(base_dict: dict, extra_config: list[str]) -> dict:
    for ec_field in extra_config:
        # colon can be part of the value - only the first one is a delimeter between keys and value
        keys_, value = ec_field.strip().split(":", 1)
        keys = keys_.split(".")
        section = get_main_section(keys[0])
        if not section:
            raise RuntimeError
        if section not in base_dict:
            base_dict[section] = {}
        match len(keys):
            case 3:
                if keys[1] not in base_dict[section]:
                    base_dict[section][keys[1]] = {}
                base_dict[section][keys[1]][keys[2]] = try_eval(value)
            case 2:
                base_dict[section][keys[1]] = try_eval(value)
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


def get_sanitized_config(config: RAGConfig) -> RAGConfig:
    # cleans the config of keys we are not willing to dump
    config = config.copy(deep=True)
    config.general.llm.api_keys = [[DEFAULT_KEY] for _ in config.general.llm.base_urls]
    config.general.embedder.api_keys = [
        [DEFAULT_KEY] for _ in config.general.embedder.base_urls
    ]
    return config


def save_experimental_vars(config_dict: dict):
    for env_var in os.environ:
        if env_var.startswith(EXPERIMENTAL_PREFIX):
            try:
                _, _, var_type, var_name = env_var.split("_", 4)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid format of the experimental variable {env_var}"
                ) from exc
            var_caster: type[int] | type[float] | type[bool] | type[str]
            match var_type:
                case "i":
                    var_caster = int
                case "f":
                    var_caster = float
                case "b":
                    var_caster = bool
                case _:
                    var_caster = str
            try:
                var_val = var_caster(os.environ[env_var])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid type of the experimental variable {env_var}."
                ) from exc
            config_dict[var_name] = var_val


def prepare_config(
    *,
    base_config_path: Path | None,
    reconfig_paths: list[Path],
    extra_config: list[str] | None,
    output_path: Path | None,
    strict_config_check: bool = False,
) -> RAGConfig:
    # Note that output_path is NOT written to until base_config_path
    # is loaded, so the configuration will not be overwritten.
    # Both parameters being the same is the typical use case when indexing.
    base_dict: dict = {}
    if base_config_path is not None:
        if not base_config_path.exists():
            raise ValueError(f"Invalid base configuration path {base_config_path}")
        base_dict = load_dict_from_json_file(base_config_path)
        # Add defaults if base_config is incomplete:
        config = RAGConfig.model_validate(base_dict)

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
    for reconfig_path in reconfig_paths:
        reconfig_dict = dict_recursive_join(
            reconfig_dict, load_dict_from_json_file(reconfig_path)
        )
        L.debug(
            "Loaded reconfig options from %s, updated reconfig:\n%s",
            reconfig_path.absolute(),
            pformat(reconfig_dict),
        )

    if extra_config is not None:
        reconfig_dict = parse_extra(reconfig_dict, extra_config)
        L.debug("Extended reconfig with extra parameters:\n%s", pformat(reconfig_dict))

    configured, problems = RAGConfig.check_override(
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
    config = RAGConfig.model_validate(config_dict)

    if output_path is not None:
        sanitized_config_dict = get_sanitized_config(config).model_dump()
        save_experimental_vars(sanitized_config_dict)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                sanitized_config_dict,
                f,
                ensure_ascii=False,
                indent=2,
            )
        L.debug("Saved configuration to working directory: %s", output_path.absolute())
    return config
