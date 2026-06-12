from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

PaddleLine = Dict[str, Any]


def _first_present(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _to_plain_sequence(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _normalize_box(box: Any) -> list[list[float]]:
    try:
        points = np.array(box, dtype=np.float32)
        if points.size == 0:
            return []
        points = points.reshape((-1, 2))
        return [[float(x), float(y)] for x, y in points[:4]]
    except Exception:
        return []


def _from_dict(result: dict[str, Any]) -> List[PaddleLine]:
    # PaddleOCR 3.x suele exponer textos, scores y polígonos en diccionarios.
    texts = _to_plain_sequence(_first_present(result, "rec_texts", "texts", "text", default=[]))
    scores = _to_plain_sequence(_first_present(result, "rec_scores", "scores", "confidence", default=[]))
    boxes = _to_plain_sequence(_first_present(result, "dt_polys", "boxes", "box", "points", default=[]))

    if isinstance(texts, str):
        texts = [texts]
    if isinstance(scores, (int, float)):
        scores = [scores]
    if boxes is None:
        boxes = []

    lines: List[PaddleLine] = []
    for idx, text in enumerate(texts or []):
        box = boxes[idx] if isinstance(boxes, (list, tuple)) and idx < len(boxes) else []
        confidence = scores[idx] if isinstance(scores, (list, tuple)) and idx < len(scores) else 0.0
        lines.append({"box": _normalize_box(box), "text": str(text), "confidence": float(confidence or 0.0)})
    return lines


def _is_paddle_v2_line(item: Any) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return False
    tail = item[-1]
    if not isinstance(tail, (list, tuple)) or len(tail) < 2:
        return False
    try:
        float(tail[1])
    except Exception:
        return False
    return len(_normalize_box(item[0])) >= 4


def _from_v2_line(line: Any) -> PaddleLine:
    try:
        text, confidence = line[-1]
        return {"box": _normalize_box(line[0]), "text": str(text), "confidence": float(confidence or 0.0)}
    except Exception:
        return {"box": [], "text": "", "confidence": 0.0}


def normalize_paddle_result(result: Any) -> List[PaddleLine]:
    """Convierte salidas de PaddleOCR 2.x/3.x a una lista uniforme.

    Formato de salida canónico:
    `[{"box": [[x, y], ...], "text": "...", "confidence": 0.98}, ...]`
    """

    if result is None:
        return []

    if isinstance(result, dict):
        # Algunos wrappers ya devuelven una línea normalizada.
        if "text" in result and ("box" in result or "points" in result):
            confidence = _first_present(result, "confidence", "score", default=0.0)
            box = _first_present(result, "box", "points", default=[])
            return [
                {
                    "box": _normalize_box(box),
                    "text": str(result.get("text") or ""),
                    "confidence": float(confidence or 0.0),
                }
            ]
        return _from_dict(result)

    if not isinstance(result, (list, tuple)) or len(result) == 0:
        return []

    if all(isinstance(item, dict) for item in result):
        lines: List[PaddleLine] = []
        for item in result:
            lines.extend(normalize_paddle_result(item))
        return lines

    if all(_is_paddle_v2_line(item) for item in result):
        return [_from_v2_line(item) for item in result]

    # PaddleOCR 2.x normalmente envuelve las líneas de una imagen dentro de otra lista:
    # [ [ [box, (text, conf)], ... ] ]. Para batch o formatos híbridos, aplanamos
    # recursivamente sin asumir una versión específica.
    lines: List[PaddleLine] = []
    for item in result:
        lines.extend(normalize_paddle_result(item))
    return lines
