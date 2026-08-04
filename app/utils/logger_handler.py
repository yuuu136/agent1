import datetime
import logging
import os

from app.utils.tool_path import get_project_abs_path


LOG_ROOT = get_project_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(message)s"
)


def get_logger(
    name: str,
    console_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    log_file: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))

    if logger.hasHandlers():
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = os.path.join(
            LOG_ROOT,
            f"{name}-{datetime.datetime.now().strftime('%Y-%m-%d')}.log",
        )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


logger = get_logger("agent")
