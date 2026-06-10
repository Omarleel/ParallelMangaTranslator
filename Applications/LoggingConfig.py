from __future__ import annotations

import logging
import os
from pathlib import Path


LOGGER_NAME = "parallel_manga_translator"


def _level_from_env(default: int) -> int:
    raw = os.getenv("PMT_LOG_LEVEL", "").strip().upper()
    if not raw:
        return default
    if raw.isdigit():
        return int(raw)
    return int(getattr(logging, raw, default))


def configure_logging(log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configura logging de forma idempotente para consola y archivo.

    Variables útiles:
    - PMT_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
    - PMT_LOG_FILE=debug.log
    """
    logger = logging.getLogger(LOGGER_NAME)
    resolved_level = _level_from_env(level)
    resolved_file = log_file or os.getenv("PMT_LOG_FILE", "debug.log")

    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not any(getattr(handler, "_pmt_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        console_handler._pmt_console = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    if resolved_file and not any(getattr(handler, "_pmt_file", None) == str(resolved_file) for handler in logger.handlers):
        file_path = Path(resolved_file)
        if file_path.parent and str(file_path.parent) not in {"", "."}:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler._pmt_file = str(resolved_file)  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    for handler in logger.handlers:
        if getattr(handler, "_pmt_console", False):
            handler.setLevel(resolved_level)

    return logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
