import hashlib
import os
from typing import Any
from app.utils.logger_handler import logger
def get_file_md5(file_path: str) -> str | None:
    if not os.path.exists(file_path):
        logger.error(f"{file_path} 不存在")
        return None
    if not os.path.isfile(file_path):
        logger.error(f"{file_path} 不是文件")
        return None
    md5_obj = hashlib.md5()
    chunk_size = 1024 * 1024
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_obj.update(chunk)
    return md5_obj.hexdigest()
def allow_file_types(file_path: str, allow_types: tuple[str, ...]) -> list[str]:
    files = []
    if not os.path.exists(file_path):
        logger.error(f"{file_path} 不存在")
        return files
    if not os.path.isdir(file_path):
        logger.error(f"不是文件夹: {file_path}")
        return files

    normalized_allow_types = tuple(ext.lower() for ext in allow_types)
    for name in os.listdir(file_path):
        full_path = os.path.join(file_path, name)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(full_path)[-1].lower()
        if ext in normalized_allow_types:
            files.append(full_path)

    return files


def pdf_loader(file_path: str) -> Any:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(file_path)


def text_loader(file_path: str) -> Any:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(file_path, encoding="utf-8")
