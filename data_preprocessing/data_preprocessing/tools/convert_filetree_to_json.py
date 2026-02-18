import json
import shutil
from dataclasses import asdict
from pathlib import Path

import aiofiles
import tqdm.asyncio

from ..docs import DirDoc, FileDoc
from ..utils import get_extension, get_unique_id


class FileTreeDatasetProcessor:

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.output_dir.mkdir()

    async def __call__(self):
        await tqdm.asyncio.tqdm.gather(
            *[self._process_file_or_dir(p) for p in self.input_dir.rglob("*")]
        )

    async def _process_file_or_dir(self, path: Path):
        if path.is_file():
            await self._process_file(path)
        elif path.is_dir() and not any(path.iterdir()):
            await self._process_empty_dir(path)

    async def _process_file(self, file_path: Path):
        file_id = get_unique_id("file-")

        file_dir = self.output_dir / file_id
        file_dir.mkdir()

        content_id = get_unique_id("part-")
        content_path = file_dir / f"{content_id}{get_extension(file_path)}"
        shutil.copyfile(file_path, content_path)

        doc = FileDoc(
            id=file_id,
            type="fs:file",
            content=[
                {
                    "id": content_id,
                    "path_textualized": str(content_path.relative_to(self.output_dir)),
                }
            ],
            name=file_path.name,
            path=str(file_path.parent.relative_to(self.input_dir)),
        )

        async with aiofiles.open(
            file_dir / f"{file_id}.json", "w", encoding="utf-8"
        ) as f:
            await f.write(json.dumps(asdict(doc), ensure_ascii=False, indent=2))

    async def _process_empty_dir(self, dir_path: Path):
        dir_id = get_unique_id("dir-")

        dir_dir = self.output_dir / dir_id
        dir_dir.mkdir()

        doc = DirDoc(
            id=dir_id,
            type="fs:directory",
            name=dir_path.name,
            path=str(dir_path.parent.relative_to(self.input_dir)),
        )

        async with aiofiles.open(
            dir_dir / f"{dir_id}.json", "w", encoding="utf-8"
        ) as f:
            await f.write(json.dumps(asdict(doc), ensure_ascii=False, indent=2))


async def process_files(input_dir: Path, output_dir: Path):
    await FileTreeDatasetProcessor(input_dir, output_dir)()
