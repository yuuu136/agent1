from pathlib import Path


def get_file_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def get_project_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def get_abs_path(file_name: str) -> str:
    return str(Path(get_file_root()) / file_name)


def get_project_abs_path(file_name: str) -> str:
    return str(Path(get_project_root()) / file_name)
