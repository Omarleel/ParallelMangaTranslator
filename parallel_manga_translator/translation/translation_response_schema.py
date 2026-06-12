from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


LLM_TRANSLATION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "traducciones": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "traduccion": {"type": "string"},
                },
                "required": ["id", "traduccion"],
            },
        }
    },
    "required": ["traducciones"],
}


def validate_translation_response(data: Mapping[str, Any], expected_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """Valida la salida del LLM de forma estricta sin dependencia externa.

    El proveedor puede garantizar JSON, pero esta validación evita respuestas con ids
    duplicados, campos extra, listas incompletas o tipos incorrectos antes de renderizar.
    """
    if not isinstance(data, Mapping):
        raise TypeError("La respuesta de traducción debe ser un objeto JSON.")
    allowed_root = {"traducciones"}
    extra_root = set(data.keys()) - allowed_root
    if extra_root:
        raise ValueError(f"Campos raíz no permitidos: {sorted(extra_root)}")
    traducciones = data.get("traducciones")
    if not isinstance(traducciones, list):
        raise TypeError("'traducciones' debe ser una lista.")

    expected = set(int(x) for x in expected_ids)
    seen = set()
    validated: List[Dict[str, Any]] = []
    for row in traducciones:
        if not isinstance(row, Mapping):
            raise TypeError("Cada traducción debe ser un objeto.")
        allowed_row = {"id", "traduccion"}
        extra_row = set(row.keys()) - allowed_row
        if extra_row:
            raise ValueError(f"Campos no permitidos en traducción: {sorted(extra_row)}")
        if "id" not in row or "traduccion" not in row:
            raise ValueError("Cada traducción requiere 'id' y 'traduccion'.")
        idx = row["id"]
        if not isinstance(idx, int):
            raise TypeError(f"id inválido: {idx!r}")
        if idx not in expected:
            raise ValueError(f"id fuera de rango o inesperado: {idx}")
        if idx in seen:
            raise ValueError(f"id duplicado: {idx}")
        traducido = row["traduccion"]
        if not isinstance(traducido, str):
            raise TypeError(f"traduccion inválida para id={idx}")
        validated.append({"id": idx, "traduccion": traducido})
        seen.add(idx)

    if seen != expected:
        raise ValueError(f"Faltan traducciones para ids: {sorted(expected - seen)}")
    return validated
