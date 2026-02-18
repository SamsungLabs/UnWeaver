import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas
from matplotlib.collections import PolyCollection

from .utils import load_json

TIMING_FILE = "timing_logs.json"
QUERY_TOKEN_STATS_SUFFIX = "_token_stats.json"
BOX_HEIGHT = 0.5


def assign_rel_time(timing_df: pandas.DataFrame, min_time: float) -> pandas.DataFrame:
    timing_df["rel_start"] = timing_df["start"] - min_time
    timing_df["rel_end"] = timing_df["end"] - min_time
    timing_df = timing_df.sort_values("rel_start").reset_index()
    return timing_df


def assign_coords(timing_df: pandas.DataFrame) -> pandas.DataFrame:
    timing_df["lt"] = list(
        zip(timing_df["rel_start"], timing_df.index - BOX_HEIGHT / 2)
    )
    timing_df["lb"] = list(
        zip(timing_df["rel_start"], timing_df.index + BOX_HEIGHT / 2)
    )
    timing_df["rt"] = list(zip(timing_df["rel_end"], timing_df.index - BOX_HEIGHT / 2))
    timing_df["rb"] = list(zip(timing_df["rel_end"], timing_df.index + BOX_HEIGHT / 2))
    return timing_df


def plot_block_profile(timing_df: pandas.DataFrame) -> dict:
    # calculating vertex positions
    timing_df = assign_coords(timing_df)
    verts = timing_df[["lt", "rt", "rb", "lb", "lt"]].values.tolist()
    bars = PolyCollection(verts)

    fig_dict = {}
    fig, ax = plt.subplots()
    fig_dict["block_profile"] = fig
    ax.add_collection(bars)
    ax.autoscale()
    ax.set_yticks(timing_df.index)
    ax.set_yticklabels(timing_df["label"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Code block")
    fig.subplots_adjust(left=0.4)
    plt.close("all")
    return fig_dict


def plot_call_profile(timing_df: pandas.DataFrame) -> dict:
    # calculating vertex positions
    timing_df = assign_coords(timing_df)
    timing_df["phase"] = timing_df["log_data"].apply(lambda dic: dic["phase"])

    fig_dict = {}
    fig, ax = plt.subplots(dpi=300)
    fig_dict["call_profile"] = fig
    phase_legend = []
    for i, ((model, phase), phase_df) in enumerate(
        timing_df.groupby(["model", "phase"], sort=False)
    ):
        phase_verts = phase_df[["lt", "rt", "rb", "lb", "lt"]].values.tolist()
        bars = PolyCollection(phase_verts, facecolors=f"C{i}")
        ax.add_collection(bars)
        phase_legend.append(f"{model} {phase}")

    ax.autoscale()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Call number")
    plt.legend(phase_legend)
    plt.close("all")
    return fig_dict


def profile_time(working_dir: Path) -> tuple[dict[str, float], dict]:
    timing_file = working_dir / TIMING_FILE
    if timing_file.exists():
        timing_plot_dict = {}
        with open(timing_file, encoding="utf-8") as f:
            timing_data = json.load(f)

        block_df = pandas.DataFrame(timing_data["block"])
        min_time = min(block_df["start"])
        block_df = assign_rel_time(block_df, min_time)

        llm_df = pandas.DataFrame(timing_data["llm"])
        llm_df["model"] = "llm"
        emb_df = pandas.DataFrame(timing_data["embedding"])
        emb_df["model"] = "embedding"
        call_df = assign_rel_time(pandas.concat([llm_df, emb_df]), min_time)
        timing_plot_dict.update(plot_call_profile(call_df))

        block_df["tag"] = block_df["tag"].fillna("")
        block_df["label"] = block_df["func"] + " " + block_df["tag"]
        timing_metrics = dict(block_df.set_index("label")["time"])
        timing_metrics = block_df.set_index("label")["time"].to_dict()
        timing_metrics = {f"time_{key}": val for key, val in timing_metrics.items()}
        timing_plot_dict.update(plot_block_profile(block_df))
        return timing_metrics, timing_plot_dict
    return {}, {}


def extract_token_stats(results_df: pandas.DataFrame) -> pandas.Series:
    token_stats_paths = results_df["result_path"].apply(
        lambda pth: Path(  # type: ignore
            str(pth).replace("_results.json", QUERY_TOKEN_STATS_SUFFIX)
        )
    )
    if token_stats_paths.apply(lambda x: x.exists()).any():
        token_stats_dfs = token_stats_paths.apply(
            lambda pth: pandas.DataFrame(  # type:ignore
                load_json(pth)
            )
        )
        token_metrics = ["prompt_tokens", "completion_tokens", "total_tokens"]
        aggregate_stats = token_stats_dfs.apply(
            lambda row: (
                row[token_metrics].aggregate(["mean", "max", "std"]).stack()
                if not row.empty
                else pandas.Series()
            )
        )
        token_series = aggregate_stats.apply(
            lambda row: {f"{key[0]}_{key[1]}": val for key, val in dict(row).items()},
            axis=1,
        )
    else:
        token_series = pandas.Series(
            [{} for _ in token_stats_paths], index=token_stats_paths.index
        )
    return token_series


def extract_query_time(results_df: pandas.DataFrame) -> pandas.Series:
    time_stats_paths = results_df["result_path"].apply(
        lambda pth: Path(str(pth).replace("results.json", TIMING_FILE))  # type: ignore
    )
    if time_stats_paths.apply(lambda x: x.exists()).any():
        time_stats_dfs = time_stats_paths.apply(
            lambda pth: pandas.Series(load_json(pth))
        )

        def replace_nan_with_empty_list(obj):
            return [] if obj != obj else obj  # pylint: disable=comparison-with-itself

        time_stats_dfs = time_stats_dfs.map(
            replace_nan_with_empty_list
        )  # This avoids an inconvenient corner case in DF's fillna, when filling with empty lists
        llm_call_dfs = time_stats_dfs["llm"].apply(pandas.DataFrame)  # type: ignore[arg-type]
        aggregate_stats = llm_call_dfs.apply(
            lambda row: (
                row["time"].aggregate(["mean", "max", "std"])
                if "time" in row
                else pandas.Series()
            )
        )
        time_series = aggregate_stats.apply(
            lambda row: {f"query_time_{key}": val for key, val in dict(row).items()},
            axis=1,
        )
    else:
        time_series = pandas.Series(
            [{} for _ in time_stats_paths], index=time_stats_paths.index
        )
    return time_series
