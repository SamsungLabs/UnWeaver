import argparse
import asyncio
from pathlib import Path

from .literals import DATASET_NAMES
from .tools.convert_filetree_to_json import process_files
from .tools.create_filetree import get_docs
from .utils import init_logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder in which individual dataset subfolders will be created",
    )

    return parser.parse_args()


if __name__ == "__main__":
    init_logging()
    args = parse_args()

    get_docs(args.input)
    for dataset_name in DATASET_NAMES:
        input_dir = args.input / dataset_name / "files"
        output_dir = args.input / dataset_name / "files_preprocessed"
        asyncio.run(process_files(input_dir, output_dir))
