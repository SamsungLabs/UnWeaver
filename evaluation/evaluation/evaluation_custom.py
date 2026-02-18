import logging

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .literals import LOGGER

L = logging.getLogger(LOGGER)


def calculate_binary_metrics(eval_df: pd.DataFrame) -> dict:
    precission, recall, fscore, support = precision_recall_fscore_support(
        eval_df["null_answer.sys"],
        eval_df["null_answer.gold"],
        average="binary",
        pos_label=True,
        zero_division=np.nan,
    )
    accuracy = sum(eval_df["null_answer.sys"] == eval_df["null_answer.gold"]) / len(
        eval_df
    )
    binary_metrics = {
        ("non_answerable", "precision"): precission,
        ("non_answerable", "recall"): recall,
        ("non_answerable", "f1_score"): fscore,
        ("non_answerable", "accuracy"): accuracy,
        ("non_answerable", "count_sys"): sum(eval_df["null_answer.sys"]),
        ("non_answerable", "count_gt"): support or 0,
    }
    return binary_metrics
