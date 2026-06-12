from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict

import yaml


@dataclass(frozen=True)
class OnomatopoeiaDictionaries:
    """Estructuras indexadas que necesita el motor de onomatopeyas."""

    target_by_key: Dict[str, Dict[str, str]] = field(default_factory=dict)
    raw_source_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    language_aliases: Dict[str, str] = field(default_factory=dict)


class OnomatopoeiaYamlRepository:
    """Carga diccionarios YAML desde disco. No contiene lógica de detección/traducción."""

    def __init__(self, data_dir: Path, normalize_alias: Callable[[str], str]) -> None:
        self.data_dir = data_dir
        self.normalize_alias = normalize_alias

    def load(self) -> OnomatopoeiaDictionaries:
        dictionaries = OnomatopoeiaDictionaries()
        if not self.data_dir.exists():
            return dictionaries

        for yaml_path in sorted(self.data_dir.glob("*/onomatopoeias.yaml")):
            self._load_file(yaml_path, dictionaries)
        return dictionaries

    def _load_file(self, yaml_path: Path, dictionaries: OnomatopoeiaDictionaries) -> None:
        with yaml_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        if not isinstance(payload, dict):
            return

        language = str(payload.get("language") or yaml_path.parent.name).strip()
        if not language:
            return

        self._register_language_aliases(payload, yaml_path, language, dictionaries.language_aliases)
        entries = payload.get("entries") or []
        if not isinstance(entries, list):
            return

        for entry in entries:
            self._register_entry(entry, language, dictionaries)

    def _register_language_aliases(
        self,
        payload: Dict[str, Any],
        yaml_path: Path,
        language: str,
        language_aliases: Dict[str, str],
    ) -> None:
        aliases = payload.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in [language, yaml_path.parent.name, *aliases]:
            normalized_alias = self.normalize_alias(alias)
            if normalized_alias:
                language_aliases[normalized_alias] = language

    def _register_entry(
        self,
        entry: Any,
        language: str,
        dictionaries: OnomatopoeiaDictionaries,
    ) -> None:
        if not isinstance(entry, dict):
            return
        key = str(entry.get("key") or "").strip()
        if not key:
            return

        target = entry.get("target")
        if target is not None and str(target).strip():
            dictionaries.target_by_key.setdefault(key, {})[language] = str(target)

        for source in self._coerce_sources(entry):
            source = str(source).strip()
            if source:
                dictionaries.raw_source_map.setdefault(language, {})[source] = key

    @staticmethod
    def _coerce_sources(entry: Dict[str, Any]) -> list[str]:
        sources: list[str] = []
        for field_name in ("sources", "variants", "source"):
            value = entry.get(field_name)
            if value is None:
                continue
            if isinstance(value, str):
                sources.append(value)
            elif isinstance(value, list):
                sources.extend(str(item) for item in value if item is not None)
        return sources
