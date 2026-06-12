from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

LOGGER_NAME = "parallel_manga_translator"


def _resolve_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    raw = str(level or "INFO").strip().upper()
    if raw.isdigit():
        return int(raw)
    return int(getattr(logging, raw, logging.INFO))


def configure_logging(log_file: str | None = "debug.log", level: Union[int, str] = logging.INFO) -> logging.Logger:
    """Configura logging de forma idempotente.

    La configuración funcional del logger se entrega desde `config.yaml`; esta función
    no consulta variables de entorno.
    """
    logger = logging.getLogger(LOGGER_NAME)
    resolved_level = _resolve_level(level)
    resolved_file = log_file or ""

    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not any(getattr(handler, "_pmt_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
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
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
