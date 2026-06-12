from __future__ import annotations

from threading import RLock
from typing import Optional

from parallel_manga_translator.config.app_config import ApplicationConfig

_active_config: Optional[ApplicationConfig] = None
_lock = RLock()


def set_active_config(config: ApplicationConfig) -> None:
    """Registra la configuración cargada desde YAML para componentes internos."""
    global _active_config
    with _lock:
        _active_config = config


def get_active_config() -> ApplicationConfig:
    """Devuelve la configuración activa, cargándola desde config.yaml si aún no existe."""
    global _active_config
    with _lock:
        if _active_config is None:
            from parallel_manga_translator.config.config_manager import ConfigManager

            _active_config = ConfigManager().build_application_config()
        return _active_config
