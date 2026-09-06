from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Mapping, Optional


class GlossaryManager:
    """Glosario editable por proyecto.

    Soporta JSON y YAML simple si PyYAML está instalado. Formatos aceptados:
      {"兄貴": "hermano", "魔力": "poder mágico"}
      {"terms": {"兄貴": "hermano"}}
      terms:
        兄貴: hermano
    """

    DEFAULT_FILENAMES = ("glossary.json", "glosario.json", "glossary.yaml", "glossary.yml", "glosario.yaml", "glosario.yml")

    def __init__(self, glossary_path: Optional[str] = None, project_dir: Optional[str] = None) -> None:
        self.path = self._resolve_path(glossary_path, project_dir)
        self.terms: Dict[str, str] = self._load_terms(self.path)

    @classmethod
    def _resolve_path(cls, glossary_path: Optional[str], project_dir: Optional[str]) -> Optional[Path]:
        explicit = glossary_path
        if explicit:
            path = Path(explicit)
            return path if path.exists() else None
        if project_dir:
            base = Path(project_dir)
            for name in cls.DEFAULT_FILENAMES:
                candidate = base / name
                if candidate.exists():
                    return candidate
        return None

    @staticmethod
    def _load_yaml(path: Path):
        try:
            import yaml  # type: ignore
        except Exception:
            # Fallback mínimo: key: value y sección terms:
            data: Dict[str, str] = {}
            in_terms = False
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.rstrip(":") == "terms":
                    in_terms = True
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().strip('"\'')
                value = value.strip().strip('"\'')
                if key and value and (in_terms or key != "terms"):
                    data[key] = value
            return {"terms": data}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    @classmethod
    def _load_terms(cls, path: Optional[Path]) -> Dict[str, str]:
        if not path or not path.exists():
            return {}
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = cls._load_yaml(path)
            else:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            if isinstance(data, Mapping) and isinstance(data.get("terms"), Mapping):
                data = data["terms"]
            if not isinstance(data, Mapping):
                return {}
            return {str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip() and str(v).strip()}
        except Exception:
            return {}

    def __bool__(self) -> bool:
        return bool(self.terms)

    def as_prompt_text(self, max_terms: int = 80) -> str:
        if not self.terms:
            return ""
        rows = list(self.terms.items())[:max_terms]
        return "\n".join(f"- {source} => {target}" for source, target in rows)

    def apply_to_translation(self, source: str, translation: str) -> str:
        """Ajuste conservador: si el término fuente está en el original y el destino no aparece,
        no fuerza reemplazos agresivos; añade ayuda solo para términos exactos de una palabra.
        """
        output = str(translation or "")
        source = str(source or "")
        for src, target in self.terms.items():
            if src not in source or not target:
                continue
            if target.lower() in output.lower():
                continue
            # Solo sustituye si el traductor dejó literalmente el término fuente.
            output = output.replace(src, target)
        return re.sub(r"\s+", " ", output).strip()
