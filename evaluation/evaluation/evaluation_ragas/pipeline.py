import logging
import os
from typing import Any

import pandas as pd
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.dataset_schema import (
    EvaluationResult,
    SingleTurnSample,
    SingleTurnSampleOrMultiTurnSample,
)
from ragas.metrics import FactualCorrectness
from tqdm.asyncio import tqdm

from ..configurations import EvalConfig
from ..embedder import init_embedder
from ..literals import LOGGER
from ..llm import init_llm

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
tqdm.pandas()  # Required for progress_map to work

L = logging.getLogger(LOGGER)


METRICS: dict[str, Any] = {
    "factual_correctness": FactualCorrectness(
        mode="f1", atomicity="high", coverage="high", name="factual_correctness.f1"
    ),
    "factual_correctness.f1": FactualCorrectness(
        mode="f1", atomicity="high", coverage="high", name="factual_correctness.f1"
    ),
    "factual_correctness.precision": FactualCorrectness(
        mode="precision",
        atomicity="high",
        coverage="high",
        name="factual_correctness.precision",
    ),
    "factual_correctness.recall": FactualCorrectness(
        mode="recall",
        atomicity="high",
        coverage="high",
        name="factual_correctness.recall",
    ),
}


def evaluate_with_ragas(
    results_df: pd.DataFrame,
    eval_config: EvalConfig,
) -> tuple[pd.DataFrame, list[str]]:
    L.info("Started evaluation using RAGAS")

    metrics_ = [METRICS[m] for m in eval_config.metrics]
    L.info("Selected metrics initialized: %s", str(eval_config.metrics))

    dataset = build_dataset(results_df)

    llm = init_llm(llm_config=eval_config.llm)
    embedder = init_embedder(embedder_config=eval_config.embedder)

    evaluation = evaluate(
        dataset=dataset,
        metrics=metrics_,
        llm=llm,
        embeddings=embedder,
        # raise_exceptions=True,
        run_config=RunConfig(
            timeout=eval_config.llm.timeout,
            max_retries=eval_config.llm.max_retries,
            max_workers=eval_config.llm.max_workers,
        ),
        return_executor=False,
    )
    assert isinstance(evaluation, EvaluationResult)
    evaluation_df = evaluation.to_pandas()

    evaluation_df.columns = evaluation_df.columns.str.replace(
        r"\(mode=\w+\)", "", regex=True
    )  # Working around ragas's quirks

    evaluation_df.index = results_df.index
    assert (results_df["question"] == evaluation_df["user_input"]).all()
    assert (results_df["sys_answer"] == evaluation_df["response"]).all()
    assert (results_df["gt_answer"] == evaluation_df["reference"]).all()

    metric_names = [metric.name for metric in metrics_]
    results_df = results_df.merge(
        evaluation_df[metric_names], left_index=True, right_index=True
    )

    L.info("Finished evaluation using RAGAS")
    return results_df, metric_names


def build_dataset(results_df: pd.DataFrame):
    kwarg_renaming_dict = {
        "question": "user_input",
        "sys_context": "retrieved_contexts",
        "gt_context": "reference_contexts",
        "sys_answer": "response",
        "gt_answer": "reference",
    }
    column_subset = list(
        {key for key in kwarg_renaming_dict if key in results_df.columns}
    )

    if "sys_retrieval_result" in results_df.columns:
        results_df["sys_context"] = results_df["sys_retrieval_result"].apply(
            lambda d: d["context"] if isinstance(d, dict) and "context" in d else d
        )

    results_df["sys_context"] = results_df["sys_context"].apply(
        lambda x: (x if isinstance(x, list) else [x])
    )

    results_df["gt_answer"] = results_df["gt_answer"].apply(
        lambda x: str(x) if x is not None else ""
    )
    results_df["sys_answer"] = results_df["sys_answer"].apply(
        lambda x: str(x) if x is not None else ""
    )
    df = results_df[column_subset].rename(columns=kwarg_renaming_dict)

    samples: list[SingleTurnSampleOrMultiTurnSample] = [
        SingleTurnSample(**kwargs) for kwargs in df.apply(dict, axis=1)
    ]

    eval_dataset = EvaluationDataset(samples)
    return eval_dataset
