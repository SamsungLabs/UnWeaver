import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import mlflow
import pandas as pd

from .configurations import EvalConfig, prepare_config
from .evaluation_custom import calculate_binary_metrics
from .evaluation_ragas import evaluate_with_ragas
from .literals import LOGGER, QUERY_RESULTS_DIR, RESULT_FILE_GLOB
from .profiling import extract_query_time, extract_token_stats, profile_time
from .utils import init_logging, load_json

L = logging.getLogger(LOGGER)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "working_dir",
        type=Path,
        help="working directory of evaluated RAG system",
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

    return parser.parse_args()


@dataclass
class EvalResults:
    runs_df: pd.DataFrame
    results_df: pd.DataFrame
    index_config: dict
    timing_metrics: dict
    timing_plots: dict


def run_evaluation(working_dir: Path, eval_config: EvalConfig) -> EvalResults:
    # INDEX WISE
    timing_metrics, timing_plots = profile_time(working_dir)

    index_config = load_json(working_dir / "config.json")
    assert isinstance(index_config, dict)
    index_config = {f"index_{k}": v for k, v in index_config.items()}

    # QUERY WISE
    runs_df, results_df = load_data(working_dir, eval_config)

    runs_df["token_stats"] = extract_token_stats(runs_df)
    runs_df["query_time"] = extract_query_time(runs_df)

    results_df, metric_names = evaluate_with_ragas(results_df, eval_config=eval_config)

    runs_df["metrics"] = aggregate_metrics(results_df, metric_names)

    return EvalResults(
        runs_df,
        results_df,
        index_config,
        timing_metrics,
        timing_plots,
    )


def load_data(
    working_dir: Path, eval_config: EvalConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_filepaths = list((working_dir / QUERY_RESULTS_DIR).glob(RESULT_FILE_GLOB))
    L.info("Found %d files with results", len(result_filepaths))

    runs_df = pd.DataFrame(
        [{"result_path": str(result_path)} for result_path in result_filepaths]
    )
    runs_df["query_config"] = (
        runs_df["result_path"]
        .str.replace("_results.json", "_query_config.json")
        .apply(Path)
        .apply(load_json)
    )

    dfs = []
    df_index = []
    for i, result_path in runs_df["result_path"].items():
        for run in range(eval_config.num_repeats):
            df = pd.read_json(result_path, orient="index")
            df["repeat"] = run
            dfs.append(df)
            df_index.append((i, run))
    results_df = pd.concat(dfs, keys=df_index)
    results_df.index.rename(["run", "repeat", "question_id"], inplace=True)
    results_df = results_df.merge(
        runs_df["result_path"], left_on="run", right_index=True
    )
    if "source" in results_df.columns:
        results_df["source"] = results_df["source"].apply(str)
    L.info("Total number of samples to evaluate: %d", len(results_df))

    return runs_df, results_df


def aggregate_metrics(results_df: pd.DataFrame, metric_names: list[str]):
    metrics = (
        results_df.groupby(level=["run", "repeat"])[metric_names]
        .mean()
        .groupby(level="run")
        .aggregate(["mean", "var"])
    )

    # Non-NullAnswer metrics
    if (
        "null_answer.sys" in results_df.columns
        and "null_answer.gold" in results_df.columns
    ):
        non_null_df = results_df[~results_df["null_answer.sys"]]
        non_null_metrics = (
            non_null_df.groupby(level=["run", "repeat"])[metric_names]
            .mean()
            .groupby(level="run")
            .aggregate(["mean", "var"])
        )
        metrics = metrics.merge(
            non_null_metrics,
            left_index=True,
            right_index=True,
            suffixes=("", "_non_null"),
        )

        bin_metrics = results_df.groupby(level=0).apply(
            lambda subdf: pd.Series(calculate_binary_metrics(subdf))
        )
        metrics = metrics.merge(bin_metrics, left_index=True, right_index=True)

    metrics["num", "samples"] = results_df.groupby(level=0).apply(len)

    metrics_series = metrics.apply(
        lambda row: {f"{key[0]}_{key[1]}": val for key, val in dict(row).items()},
        axis=1,
    )
    L.info("Finished metrics aggregation")

    return metrics_series


def dump_results(eval_results: EvalResults, working_dir: Path):
    out_path = working_dir / "eval_results.json"
    eval_results.runs_df.to_json(out_path, indent=2)
    L.info("Dumped results to %s", out_path)

    out_path = working_dir / QUERY_RESULTS_DIR / "individual_eval.pqt"
    results_df = (
        eval_results.results_df.set_index("result_path", append=True)
        .droplevel(["run"])
        .reorder_levels(["result_path", "repeat", "question_id"])
    )
    results_df.to_parquet(out_path)
    L.info("Dumped individual evaluation results to %s", out_path)


def log_rag_to_mlflow(eval_results: EvalResults):
    with mlflow.start_run():
        for _, run in eval_results.runs_df.iterrows():
            query_params = {
                f"query_{key}": val for key, val in run["query_config"].items()
            }
            if "query_retrieval_methods" in query_params:
                if len(query_params["query_retrieval_methods"]) == 1:
                    retrieval_params = {
                        **query_params[
                            f"query_{query_params['query_retrieval_methods'][0]}"
                        ]
                    }
                    query_params.update(retrieval_params)
                else:
                    for retrieval_method in query_params["query_retrieval_methods"]:
                        retrieval_params = {**query_params[f"query_{retrieval_method}"]}
                        retrieval_params = {
                            f"query_{retrieval_method}_{key}": val
                            for key, val in retrieval_params.items()
                        }
                        query_params.update(retrieval_params)
            rag_metrics = run["metrics"]
            with mlflow.start_run(nested=True, run_name=query_params.get("run_name")):
                for key, fig in eval_results.timing_plots.items():
                    mlflow.log_figure(fig, f"{key}.png")
                params = {"result_path": run["result_path"]}
                params.update(eval_results.index_config)
                params.update(query_params)
                mlflow.log_params(params)
                print(json.dumps(rag_metrics, indent=4, sort_keys=True))
                mlflow.log_metrics(rag_metrics)
                mlflow.log_metrics(eval_results.timing_metrics)
                mlflow.log_metrics(run["token_stats"])
                mlflow.log_metrics(run["query_time"])
                pprint(params)
                pprint(rag_metrics)


def main(
    working_dir: Path,
    *,
    config_paths: list[Path] | None = None,
    extra_config: list[str] | None = None,
):
    eval_config: EvalConfig = prepare_config(
        base_config_path=None,
        reconfig_paths=config_paths,
        extra_config=extra_config,
        output_path=None,
        strict_config_check=False,
    )

    if eval_config.remote_mlflow_uri:
        mlflow.set_tracking_uri(uri=eval_config.remote_mlflow_uri)
    mlflow.set_experiment(eval_config.experiment_name)

    L.info("Starting evaluation with working dir: %s", working_dir)
    eval_results = run_evaluation(
        working_dir=working_dir,
        eval_config=eval_config,
    )
    L.info("Finished evaluation with working dir: %s", working_dir)

    dump_results(eval_results, working_dir)
    L.info("Finished dumping results to drive")

    log_rag_to_mlflow(eval_results)
    L.info("Finished logging results to MLFlow")


if __name__ == "__main__":
    args = parse_args()
    init_logging()
    main(
        args.working_dir,
        config_paths=args.config if args.config is not None else [],
        extra_config=args.extra,
    )
