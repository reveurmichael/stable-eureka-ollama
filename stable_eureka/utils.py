from pathlib import Path


def read_from_file(path: Path) -> str:
    with open(path, "r") as file:
        return file.read()
