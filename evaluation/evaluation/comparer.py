import argparse
import asyncio
import logging
from glob import glob
from itertools import combinations
from pathlib import Path

from openai import APIConnectionError, AsyncOpenAI
from pandas import DataFrame

from .utils import load_json, parse_json_list

NUM_TRIES = 10
SYSTEMS = ["A", "B", "TIE"]
METRICS = ["comprehensiveness", "diversity", "empowerment", "directness"]

L = logging.getLogger("kg_evaluation")


class Comparer:
    def __init__(self, model: str, template_path: str, client: AsyncOpenAI):
        self.model = model
        self.client = client
        with open(template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        self.score_df = DataFrame(index=SYSTEMS, columns=METRICS).fillna(0)

    async def __call__(self, query: str, gt_ans: str, sys_ans1: str, sys_ans2: str):
        """

        There are 2 core parts.
        The first one is masking the position of answer to mitigate LLM bias.
        The second one is reducing noise from LLM preference to choose response over TIE.

        """
        responses = await self._mask_input_get_responses(
            query, gt_ans, sys_ans1, sys_ans2
        )
        self._set_score_from_responses(responses)

    async def _mask_input_get_responses(
        self, query: str, gt_ans: str, sys_ans1: str, sys_ans2: str
    ) -> list[list[str]]:
        """
        This function is about masking the position in a symmetric way.
        """
        responses = []
        assert NUM_TRIES % 2 == 0
        for i in range(NUM_TRIES):
            masked_sys_ans1 = (
                sys_ans1 if i % 2 else sys_ans2
            )  # if even A is A if odd A is B
            masked_sys_ans2 = (
                sys_ans2 if i % 2 else sys_ans1
            )  # if even B is B if odd B is A
            prompt = self.prompt_template.format(
                query=query,
                gt_ans=gt_ans,
                sys_ans1=masked_sys_ans1,
                sys_ans2=masked_sys_ans2,
            )
            messages = [{"role": "user", "content": prompt}]
            response_dirty = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
            )
            if response_dirty is None:
                raise APIConnectionError
            response = parse_json_list(
                response_dirty.choices[0].message.content  # type: ignore[arg-type]
            )
            mapping = {"A": "A", "B": "B"} if i % 2 else {"A": "B", "B": "A"}
            unmasked_response = [mapping.get(item, item) for item in response]
            responses.append(unmasked_response)
        return responses

    def _set_score_from_responses(self, responses: list[list[str]]):
        """

        This function is about averaging the responses and minimizing
        the noise from the LLM answer. If it keeps cycling between answers
        A and B, then this mechanism will consider the final answer to be TIE

        """
        calc_mask = {"A": 1, "B": -1, "TIE": 0}
        reverse_mask = {1: "A", -1: "B", 0: "TIE"}
        avg_response = [0] * len(METRICS)
        for response in responses:
            response_numerical = [calc_mask.get(item, 0) for item in response]
            avg_response = list(
                map(lambda x, y: x + y, avg_response, response_numerical)
            )
        avg_response = [round(element / NUM_TRIES) for element in avg_response]
        avg_response = [reverse_mask.get(item) for item in avg_response]  # type: ignore[misc]

        for i, res in enumerate(avg_response):
            if res in SYSTEMS:
                self.score_df.loc[res, METRICS[i]] += 1  # type: ignore[operator]
            else:
                L.warning("Unexpected response: %s", res)

    def save_data(self, working_dir: Path, prefix: str = ""):
        self.score_df.to_csv(working_dir / f"{prefix}_score.csv")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "working_dir",
        type=Path,
        help="Working directory where the jsons with sys answers are stored.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file. Defaults to configs/config.json.",
        default="kg_evaluation/configs/config.json",
    )
    return parser.parse_args()


async def main(arguments):
    config = load_json(arguments.config)
    assert isinstance(config, dict)
    list_of_pairs = [
        (Path(p1), Path(p2))
        for p1, p2 in combinations(glob(str(arguments.working_dir / "*.json")), 2)
    ]
    model = config.pop("llm_model")
    template_path = config.pop("head2head_prompt_template")
    client = AsyncOpenAI(
        api_key=config.pop("llm_api_key"),
        base_url=config.pop("llm_base_url"),
        timeout=300,
    )
    for json1, json2 in list_of_pairs:
        comparer = Comparer(model, template_path, client)
        name = json1.stem + "_vs_" + json2.stem
        data1 = load_json(json1)
        data2 = load_json(json2)
        assert isinstance(data1, dict)
        assert isinstance(data2, dict)
        coroutines = [
            comparer(
                query=res1["question"],
                gt_ans=res1["gt_answer"],
                sys_ans1=res1["sys_answer"],
                sys_ans2=res2["sys_answer"],
            )
            for res1, res2 in zip(
                list(data1.values()), list(data2.values()), strict=True
            )
        ]
        await asyncio.gather(*coroutines)
        comparer.save_data(arguments.working_dir, prefix=name)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
