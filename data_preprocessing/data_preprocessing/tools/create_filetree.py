import json
from hashlib import md5
from pathlib import Path

from datasets import Dataset, load_dataset

from ..literals import DATASET_NAMES


def get_docs(output_folder_path: Path):
    ragbench = {}
    for dataset in DATASET_NAMES:
        ragbench[dataset] = load_dataset("rungalileo/ragbench", dataset, split="train")

    for dataset_name in ragbench.keys():
        dataset = ragbench[dataset_name].to_pandas()
        # we want each question to be present only once (some are generated multiple times with different LLMs)
        dataset_fixed = Dataset.from_pandas(
            dataset.drop_duplicates(subset=["question"])
        )

        dataset_folder = output_folder_path / f"{dataset_name}" / "files"
        dataset_folder.mkdir(exist_ok=True, parents=True)

        questions = {}
        for i, dp in enumerate(dataset_fixed):
            c_q = questions[i] = {}
            q = dp["question"]
            gt_context = dp["documents"]
            gt_answer = dp["response"]
            everything = dp

            hashed_gt_context = []
            for doc in gt_context:
                doc_hashed = md5(doc.encode("utf-8")).hexdigest()[:16]
                hashed_gt_context.append(doc_hashed)
                with open(dataset_folder / f"{doc_hashed}.txt", "w") as f:
                    f.write(doc)

            c_q["question"] = q
            c_q["gt_context"] = gt_context
            c_q["gt_context_hashed"] = hashed_gt_context
            c_q["gt_answer"] = gt_answer
            c_q["everything"] = everything

        with open(dataset_folder.parent / "questions.json", "w") as f:
            json.dump(questions, f, indent=4)
