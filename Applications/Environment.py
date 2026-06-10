from __future__ import annotations

import os
from typing import Any, Mapping

_TRUE_VALUES = {"1", "true", "yes", "on", "si", "sí"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def env_bool(name: str, default: bool = False) -> bool:
    """Lee una variable booleana de entorno con valores humanos comunes."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def env_int(name: str, default: int) -> int:
    """Lee un entero desde entorno y vuelve al valor por defecto si es inválido."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    """Lee un flotante desde entorno y vuelve al valor por defecto si es inválido."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def publish_env_defaults(defaults: Mapping[str, Any]) -> None:
    """Publica valores por defecto sin pisar variables ya definidas por el usuario."""
    for key, value in defaults.items():
        if value is not None and os.getenv(key) is None:
            os.environ[key] = str(value)
