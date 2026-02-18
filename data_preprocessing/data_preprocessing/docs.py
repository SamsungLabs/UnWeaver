from dataclasses import dataclass


@dataclass
class FileDoc:
    id: str
    type: str
    content: list[dict[str, str]]

    name: str
    path: str


@dataclass
class DirDoc:
    id: str
    type: str

    name: str
    path: str
