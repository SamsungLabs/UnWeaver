import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat

import mlflow
import pandas as pd

from .base import LOGGER, QueryResult
from .configurations import RAGConfig, get_sanitized_config, prepare_config
from .rag import Unweaver
from .utils import TimeLogger, init_logging

OUTPUT_FOLDER = "query_results"


L = logging.getLogger(LOGGER)


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "question_path",
        type=Path,
        help="Path to a JSON file with questions.",
    )
    parser.add_argument(
        "working_dir",
        type=Path,
        help="Directory containing indexed files to be used.",
    )
    parser.add_argument(
        "--run_name", type=str, default="query", help="Name for the run output files."
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=None,
        help="""Path to the JSON formatted file with configuration of the pipeline.
    Can be repeated - values get overwritten in the order of occurrence.
    The config.json from working_dir is read first as the default.""",
    )
    parser.add_argument(
        "--extra",
        action="append",
        type=str,
        help="""Extra values to add to the general config in key:value format. Example:
    python3 -m unweaver.index --extra g.llm.api_key:no_key --extra q.retrieve_top_k:10""",
    )
    parsed_args = parser.parse_args()
    return parsed_args


async def main(
    question_path: Path,
    working_dir: Path,
    run_name: str,
    config_paths: list[Path],
    extra_config: list[str] | None,
):  # pylint: disable=too-many-statements,too-many-locals

    os.environ["GRAPHRAG_STAGE"] = "query"

    results_dir = working_dir / OUTPUT_FOLDER
    results_dir.mkdir(exist_ok=True)
    init_logging(
        log_path_rotating=working_dir
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
        log_path_last=results_dir / f"{run_name}_logs.log",
    )

    base_config_path = working_dir / "config.json"

    if not base_config_path.exists():
        raise ValueError("Base configuration not found - invalid index?")

    config: RAGConfig = prepare_config(
        base_config_path=base_config_path,
        reconfig_paths=config_paths,
        extra_config=extra_config,
        output_path=results_dir / f"{run_name}_query_config.json",
        strict_config_check=False,
    )

    mlflow.set_tracking_uri(uri=config.general.remote_mlflow_uri)
    mlflow.set_experiment(config.general.experiment_name_query)
    with mlflow.start_run():
        mlflow.log_params(get_sanitized_config(config).model_dump())
        mlflow.log_param("workingdir", working_dir)

        rag_class = Unweaver

        rag = await rag_class.in_working_dir(
            working_dir=working_dir,
            config=config,
            reset=False,
            run_name=run_name,
        )

        question_df = pd.read_json(question_path, orient="index")
        questions = question_df["question"].tolist()
        contexts_gt = (
            question_df["gt_context"].tolist()
            if "gt_context" in question_df.columns
            else None
        )

        L.info("Loaded %d queries for processing", len(questions))
        mlflow.log_param("num_questions", len(questions))

        L.info("Started querying with config:\n%s", pformat(config.query.model_dump()))

        start_time = time.perf_counter()
        results = await rag.query(
            questions, params=config.query, contexts_gt=contexts_gt
        )

        answers = []
        retrieval_results: list[dict] = []
        metadatas: list[dict] = []
        for result in results:
            assert isinstance(result, QueryResult)
            answers.append(result.answer)
            retrieval_results.append(result.retrieval_result.model_dump())
            metadatas.append(result.metadata)

        question_df["sys_retrieval_result"] = retrieval_results
        question_df["sys_metadata"] = metadatas

        question_df["sys_answer"] = answers

        end_time = time.perf_counter()
        mlflow.log_metric("querying_time", end_time - start_time)
        mlflow.log_metric(
            "avg_query_time", 1.0 * (end_time - start_time) / len(questions)
        )

        # pylint: disable=no-member
        question_df.to_json(
            results_dir / f"{run_name}_results.json", orient="index", indent=2
        )

        if rag.usage_monitor is not None:
            rag.usage_monitor.dump_usage(results_dir / f"{run_name}_token_stats.json")

        if TimeLogger.profiling_on:
            TimeLogger.dump_logs(results_dir / f"{run_name}_timing_logs.json")
            mlflow.log_metrics(
                {
                    f"timing.{i:02}.{td['func']}.{td['tag']}": td["time"]
                    for i, td in enumerate(TimeLogger.timing_logs["block"])
                }
            )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            question_path=args.question_path,
            working_dir=args.working_dir,
            run_name=args.run_name,
            config_paths=args.config if args.config is not None else [],
            extra_config=args.extra,
        )
    )
