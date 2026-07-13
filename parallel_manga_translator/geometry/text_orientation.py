from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def normalize_rotation_angle(angle: Any, *, max_abs: float = 89.0, snap_zero: float = 0.65) -> float:
    """Normaliza una inclinación visual al rango [-89, 89].

    El valor usa coordenadas de imagen: los grados positivos inclinan el texto hacia
    abajo a la derecha (sentido horario en pantalla). Las orientaciones de 180° se
    consideran equivalentes para texto y se pliegan al rango de edición útil.
    """
    try:
        value = float(angle)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    while value <= -90.0:
        value += 180.0
    while value > 90.0:
        value -= 180.0
    value = max(-abs(float(max_abs)), min(abs(float(max_abs)), value))
    if abs(value) < max(0.0, float(snap_zero)):
        value = 0.0
    return float(value)


def _ordered_quad(points: Any) -> Optional[np.ndarray]:
    """Ordena cuatro vértices alrededor de su centro sin asumir su inclinación."""
    try:
        array = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    except Exception:
        return None
    if len(array) < 4 or not np.isfinite(array[:4]).all():
        return None
    array = array[:4]
    center = array.mean(axis=0)
    polar = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
    quad = array[np.argsort(polar)]
    edge_lengths = [float(np.linalg.norm(quad[(index + 1) % 4] - quad[index])) for index in range(4)]
    if min(edge_lengths) < 1.0:
        return None
    # Empezar cerca de la esquina superior izquierda vuelve los metadatos estables,
    # pero la estimación del ángulo no depende de este punto inicial.
    start = int(np.argmin(quad[:, 0] + quad[:, 1]))
    return np.roll(quad, -start, axis=0)


def _edge_angle(vector: np.ndarray) -> Optional[float]:
    length = float(np.linalg.norm(vector))
    if length < 1.0:
        return None
    dx, dy = float(vector[0]), float(vector[1])
    # La dirección de lectura de una arista es axial: invertirla 180° no cambia
    # la inclinación visual. Orientarla hacia la derecha mantiene [-90°, 90°].
    if dx < 0.0 or (abs(dx) < 1e-8 and dy < 0.0):
        dx, dy = -dx, -dy
    return normalize_rotation_angle(math.degrees(math.atan2(dy, dx)), snap_zero=0.0)


def _opposite_edge_angle(quad: np.ndarray, first_edge: int) -> Tuple[Optional[float], float]:
    indices = (first_edge, (first_edge + 2) % 4)
    rows: List[Tuple[float, float]] = []
    for index in indices:
        vector = quad[(index + 1) % 4] - quad[index]
        angle = _edge_angle(vector)
        length = float(np.linalg.norm(vector))
        if angle is not None:
            rows.append((angle, length))
    if not rows:
        return None, 0.0
    doubled = [math.radians(angle * 2.0) for angle, _ in rows]
    x = sum(weight * math.cos(rad) for (_, weight), rad in zip(rows, doubled))
    y = sum(weight * math.sin(rad) for (_, weight), rad in zip(rows, doubled))
    angle = math.degrees(math.atan2(y, x)) / 2.0 if abs(x) + abs(y) >= 1e-8 else rows[0][0]
    return normalize_rotation_angle(angle, snap_zero=0.0), sum(weight for _, weight in rows) / len(rows)


def polygon_text_angle(points: Any) -> Optional[float]:
    """Obtiene la inclinación del eje largo de un polígono OCR.

    Los detectores no siempre devuelven los vértices empezando en la misma esquina.
    Por eso se evalúan los dos pares de aristas opuestas y se usa el eje más largo,
    que normalmente coincide con la línea base del texto.
    """
    quad = _ordered_quad(points)
    if quad is None:
        return None
    first_angle, first_length = _opposite_edge_angle(quad, 0)
    second_angle, second_length = _opposite_edge_angle(quad, 1)
    candidates = [(first_angle, first_length), (second_angle, second_length)]
    candidates = [(angle, length) for angle, length in candidates if angle is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[1], reverse=True)
    longest_angle, longest_length = candidates[0]
    if len(candidates) > 1:
        other_angle, other_length = candidates[1]
        # En cajas casi cuadradas la orientación es ambigua. Elegir el eje más
        # horizontal evita giros de 90° provocados por variaciones de un píxel.
        if other_length >= longest_length * 0.90 and abs(other_angle) < abs(longest_angle):
            longest_angle = other_angle
    return normalize_rotation_angle(longest_angle)

def _detection_box_and_confidence(detection: Any) -> Tuple[Any, float]:
    if isinstance(detection, dict):
        box = detection.get("box") or detection.get("points") or detection.get("dt_poly") or []
        confidence = detection.get("confidence", detection.get("score", 0.0))
    else:
        try:
            box = detection[0]
        except Exception:
            box = []
        try:
            confidence = detection[2]
        except Exception:
            confidence = 0.0
    try:
        confidence_value = max(0.05, min(1.0, float(confidence or 0.0)))
    except Exception:
        confidence_value = 0.25
    return box, confidence_value


def _axial_distance(a: float, b: float) -> float:
    delta = abs(normalize_rotation_angle(a - b, snap_zero=0.0))
    return min(delta, 180.0 - delta)


def estimate_text_rotation(detections: Iterable[Any]) -> Dict[str, Any]:
    """Combina varios polígonos OCR y devuelve una inclinación robusta.

    La media es axial (ángulos separados por 180° son equivalentes). Se eliminan
    detecciones claramente discordantes para que un pequeño falso positivo no gire
    todo el bloque traducido.
    """
    samples: List[Tuple[float, float, List[List[float]]]] = []
    for detection in detections or []:
        box, confidence = _detection_box_and_confidence(detection)
        quad = _ordered_quad(box)
        if quad is None:
            continue
        angle = polygon_text_angle(quad)
        if angle is None:
            continue
        edge_length = (float(np.linalg.norm(quad[1] - quad[0])) + float(np.linalg.norm(quad[2] - quad[3]))) / 2.0
        weight = max(1.0, edge_length) * confidence
        samples.append((angle, weight, [[float(x), float(y)] for x, y in quad.tolist()]))

    if not samples:
        return {
            "angle": 0.0,
            "confidence": 0.0,
            "polygons": [],
            "sample_count": 0,
        }

    def axial_mean(rows: Sequence[Tuple[float, float, List[List[float]]]]) -> Tuple[float, float]:
        x = sum(weight * math.cos(math.radians(angle * 2.0)) for angle, weight, _ in rows)
        y = sum(weight * math.sin(math.radians(angle * 2.0)) for angle, weight, _ in rows)
        total = max(1e-8, sum(weight for _, weight, _ in rows))
        strength = min(1.0, math.hypot(x, y) / total)
        mean = math.degrees(math.atan2(y, x)) / 2.0
        return normalize_rotation_angle(mean), strength

    initial_angle, _ = axial_mean(samples)
    filtered = [row for row in samples if _axial_distance(row[0], initial_angle) <= 24.0]
    if not filtered:
        filtered = samples
    angle, strength = axial_mean(filtered)
    return {
        "angle": round(normalize_rotation_angle(angle), 3),
        "confidence": round(float(strength), 4),
        "polygons": [row[2] for row in filtered],
        "sample_count": len(filtered),
    }


def text_rotation_metadata(detections: Iterable[Any], *, source: str = "ocr_polygons") -> Dict[str, Any]:
    result = estimate_text_rotation(detections)
    return {
        "text_rotation_angle": result["angle"],
        "text_rotation_confidence": result["confidence"],
        "text_rotation_source": source,
        "text_polygons": result["polygons"],
        "text_rotation_samples": result["sample_count"],
    }
