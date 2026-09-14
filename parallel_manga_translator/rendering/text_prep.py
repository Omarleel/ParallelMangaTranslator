"""Preparación del texto antes de medirlo o repartirlo.

Cuatro líneas que compartían `TextFittingMixin` y el repartidor de ranuras. Vivían en el
primero, así que el segundo tenía que llamarlas por herencia: acoplamiento entre hermanos
para una función pura.
"""

from __future__ import annotations

import re


def normalize_text(texto: str) -> str:
    texto = str(texto or " ").replace("\r", "\n")
    texto = re.sub(r"[ \t\f\v]+", " ", texto)
    return texto.strip() or " "


__all__ = ["normalize_text"]
