"""Reglas que comparten los dos proveedores de traducción.

Vivían en `TraditionalTranslationMixin`, y `LlmTranslationMixin` las tomaba prestadas por
herencia de hermano. Esa era la razón de fondo por la que `GroqTranslationProvider` tenía
que heredar el motor tradicional **entero**: no quería su motor, quería estas cinco cosas.

Son funciones con parámetros explícitos y no un colaborador con estado a propósito:
`provider` se decide durante la construcción del traductor tradicional —se prueban varios y
se fija el que responde—, así que cualquier objeto que guardase una copia se desincronizaría
en silencio. Es el mismo fallo que ya se coló una vez duplicando `bubble_fill_strategy`.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional

#: Espacio de ancho completo. Los proveedores lo devuelven en textos CJK y, si se deja,
#: el renderizador lo mide como un carácter más al partir líneas.
ESPACIO_ANCHO_COMPLETO = "　"


def is_blank(texto: Optional[str]) -> bool:
    return texto is None or not str(texto).strip()


def normalize_translation(texto: str) -> str:
    texto = str(texto or "")
    texto = texto.replace(ESPACIO_ANCHO_COMPLETO, " ")
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def provider_lang_code(ui_langs: Mapping[str, str], ui_lang: str, provider: str) -> str:
    """Traduce el idioma de la UI al código que espera cada proveedor."""
    code = ui_langs[ui_lang]

    if provider == "google":
        if code == "zh":
            return "zh-CN"
        return code

    if provider == "deepl":
        if code == "zh":
            return "ZH"
        if code == "pt":
            return "PT-PT"
        if code == "auto":
            return "auto"
        return code.upper()

    raise ValueError(f"Proveedor no soportado: {provider}")


def same_language(
    ui_langs: Mapping[str, str],
    provider: Optional[str],
    idioma_entrada: str,
    idioma_salida: str,
) -> bool:
    """Evita traducir cuando origen y destino son efectivamente iguales."""
    if provider not in {"google", "deepl"}:
        return False

    src = provider_lang_code(ui_langs, idioma_entrada, provider)
    tgt = provider_lang_code(ui_langs, idioma_salida, provider)

    if src.lower() == "auto":
        return False

    return src.split("-")[0].lower() == tgt.split("-")[0].lower()


def persistent_key(
    cache,
    texto: str,
    metodo: str,
    provider: Optional[str],
    idioma_entrada: str,
    idioma_salida: str,
) -> str:
    """Clave de caché de una traducción concreta, con su motor y su par de idiomas."""
    return cache.hash_text(metodo, provider or "unknown", idioma_entrada, idioma_salida, texto)


__all__ = [
    "ESPACIO_ANCHO_COMPLETO",
    "is_blank",
    "normalize_translation",
    "persistent_key",
    "provider_lang_code",
    "same_language",
]
