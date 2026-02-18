import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat

import mlflow
import mlflow.data

from .base import LOGGER
from .configurations import RAGConfig, get_sanitized_config, prepare_config
from .content_handlers import DatasetContentHandler
from .rag import Unweaver
from .utils import TimeLogger, init_logging

L = logging.getLogger(LOGGER)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Index provided texts and build graph(s)"
    )

    parser.add_argument(
        "input_path", type=Path, help="Directory with dataset to be indexed."
    )
    parser.add_argument(
        "working_dir",
        type=Path,
        help="""Directory to store outputs in.""",
    )

    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=None,
        help="""Path to the JSON formatted file with configuration of the pipeline.
    Can be repeated - values get overwritten in the order of occurrence.""",
    )
    parser.add_argument(
        "--extra",
        action="append",
        type=str,
        help="""Extra values to add to the config in key:value format. Example:
    python3 -m unweaver.index --extra g.llm.api_key:no_key --extra g.embedder.embedding_dim:4096
    Top-level sections (general, query, index) can be specified with any number of initial letters
    (eg. one, as in the example above).""",
    )
    parsed_args = parser.parse_args()
    return parsed_args


def prepare_working_dir(*, working_dir: Path):
    if working_dir.exists():
        L.info(
            "Working directory %s already exists, reusing it...", working_dir.absolute()
        )
    else:
        working_dir.mkdir(parents=True)
        L.info("Created working directory %s", working_dir.absolute())


def build_input_from_dataset(input_path: Path) -> tuple[list[str], Path]:
    assert input_path.is_dir()
    input_: list[str] = []
    root_dir = input_path.absolute()
    for path in sorted(root_dir.iterdir()):
        input_.append(path.stem)
    L.info("Discovered %d documents for indexing", len(input_))
    return input_, root_dir


async def main(
    input_path: Path,
    working_dir: Path,
    config_paths: list[Path],
    extra_config: list[str] | None,
):
    os.environ["GRAPHRAG_STAGE"] = "index"

    prepare_working_dir(working_dir=working_dir)

    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    init_logging(
        log_path_rotating=working_dir / f"{ts}.log",
        log_path_last=working_dir / "indexing_logs.log",
    )

    base_config_path = working_dir / "config.json"

    config: RAGConfig = prepare_config(
        base_config_path=None,
        reconfig_paths=config_paths,
        extra_config=extra_config,
        output_path=base_config_path,
        strict_config_check=False,
    )

    mlflow.set_tracking_uri(uri=config.general.remote_mlflow_uri)
    mlflow.set_experiment(config.general.experiment_name_index)
    with mlflow.start_run():
        mlflow.log_params(get_sanitized_config(config).model_dump())

        mlflow.log_param("input_path", input_path)
        mlflow.set_experiment_tag("input_path", input_path)

        start = time.time()

        rag = await Unweaver.in_working_dir(
            working_dir=working_dir,
            config=config,
            reset=True,
            run_name=None,
        )
        mlflow.log_param("indexing_run_id", rag.id)

        L.info("Started indexing with config:\n%s", pformat(config.model_dump()))

        input_, root_dir = build_input_from_dataset(input_path)
        handler = DatasetContentHandler(root_dir=root_dir)
        await rag.insert(input_=input_, handler=handler)

        if rag.usage_monitor is not None:
            rag.usage_monitor.dump_usage(working_dir / "token_stats.json")

        if TimeLogger.profiling_on:
            TimeLogger.dump_logs(working_dir / "timing_logs.json")
            mlflow.log_metrics(
                {
                    f"timing.{i:02}.{td['func']}.{td['tag']}": td["time"]
                    for i, td in enumerate(TimeLogger.timing_logs["block"])
                }
            )

        time_diff = time.time() - start

        mlflow.log_metric("indexing_time", time_diff)
        L.info("indexing time: %f", time_diff)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            input_path=args.input_path,
            working_dir=args.working_dir,
            config_paths=args.config if args.config is not None else [],
            extra_config=args.extra,
        )
    )
