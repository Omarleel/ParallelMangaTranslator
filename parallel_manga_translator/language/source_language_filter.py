from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping


class SourceLanguageFilter:
    """Filtro ligero por escritura/idioma de origen.

    La intención no es hacer detección lingüística perfecta, sino impedir que textos
    claramente escritos en otro sistema (por ejemplo, latín dentro de una página
    japonesa) entren al flujo de limpieza, OCR especializado y traducción.
    """

    CJK_LANGUAGES = {"Chino", "Japonés", "Coreano"}
    LATIN_LANGUAGES = {"Inglés", "Español", "Francés", "Italiano", "Portugués"}

    _RE_KANA = re.compile(r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff]")
    _RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3005\u303b]")
    _RE_HANGUL = re.compile(r"[\uac00-\ud7af]")
    _RE_LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
    _RE_DIGIT = re.compile(r"\d")

    def __init__(self, idioma_entrada: str) -> None:
        self.idioma_entrada = str(idioma_entrada or "").strip()

    @staticmethod
    def normalize(text: Any) -> str:
        return unicodedata.normalize("NFKC", str(text or "")).strip()

    @classmethod
    def signal_counts(cls, text: Any) -> dict[str, int]:
        normalized = cls.normalize(text)
        kana = len(cls._RE_KANA.findall(normalized))
        cjk = len(cls._RE_CJK.findall(normalized))
        hangul = len(cls._RE_HANGUL.findall(normalized))
        latin = len(cls._RE_LATIN.findall(normalized))
        digits = len(cls._RE_DIGIT.findall(normalized))
        meaningful_without_digits = kana + cjk + hangul + latin
        visible = len(re.sub(r"\s+", "", normalized))
        return {
            "kana": kana,
            "cjk": cjk,
            "hangul": hangul,
            "latin": latin,
            "digits": digits,
            "meaningful_without_digits": meaningful_without_digits,
            "meaningful": meaningful_without_digits + digits,
            "visible": visible,
        }

    @classmethod
    def has_meaningful_text(cls, text: Any) -> bool:
        counts = cls.signal_counts(text)
        return counts["meaningful_without_digits"] > 0 or (counts["digits"] >= 2 and counts["visible"] == counts["digits"])

    def has_source_language_signal(self, text: Any) -> bool:
        counts = self.signal_counts(text)
        language = self.idioma_entrada
        if language == "Japonés":
            # Manga japonés puede contener kanji sin kana; por eso aceptamos CJK.
            return counts["kana"] + counts["cjk"] > 0
        if language == "Chino":
            return counts["cjk"] > 0
        if language == "Coreano":
            return counts["hangul"] > 0
        if language in self.LATIN_LANGUAGES:
            return counts["latin"] > 0
        # Idioma no modelado: no bloquees el flujo por falta de un detector específico.
        return self.has_meaningful_text(text)

    def should_process_text(self, text: Any, *, allow_empty: bool = False) -> bool:
        normalized = self.normalize(text)
        if not normalized:
            return bool(allow_empty)
        if not self.has_meaningful_text(normalized):
            return bool(allow_empty)
        return self.has_source_language_signal(normalized)

    @staticmethod
    def _metadata(region: Any) -> Mapping[str, Any]:
        metadata = getattr(region, "metadata", {}) or {}
        return metadata if isinstance(metadata, Mapping) else {}

    def region_text_hints(self, region: Any) -> list[str]:
        hints: list[str] = []
        source_hint = getattr(region, "source_text_hint", "")
        if source_hint:
            hints.append(str(source_hint))
        metadata = self._metadata(region)
        for key in (
            "text",
            "ocr_text",
            "source_text",
            "source_text_hint",
            "clean_guard_ocr_text",
            "ocr_group_text",
        ):
            value = metadata.get(key)
            if value:
                hints.append(str(value))
        # Conserva orden, elimina duplicados triviales.
        seen = set()
        unique: list[str] = []
        for hint in hints:
            normalized = self.normalize(hint)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(hint)
        return unique

    @staticmethod
    def _region_kind(region: Any) -> str:
        return str(getattr(region, "kind", "") or "").strip().lower()

    @staticmethod
    def _metadata_bool(metadata: Mapping[str, Any], *keys: str) -> bool:
        return any(bool(metadata.get(key)) for key in keys)

    def should_preserve_region_without_processing(self, region: Any) -> bool:
        """True para regiones visuales que deben quedarse intactas.

        Regla importante: si la onomatopeya está dentro de un globo de diálogo,
        debe procesarse y traducirse como texto del globo. Sólo se preservan
        regiones externas/SFX o marcas explícitas de no traducible. La excepción
        también corrige metadata vieja generada por la versión v6, donde un globo
        corto podía quedar con ``skip_cleanup_translation`` sólo por una pista
        OCR tipo ``は``/``パ``.
        """
        metadata = self._metadata(region)
        region_kind = self._region_kind(region)
        bubble_kind = region_kind in {"dialogue", "narration", "unknown"}
        bubble_onomatopoeia = bubble_kind and self._metadata_bool(
            metadata,
            "bubble_onomatopoeia",
            "translate_inside_bubble",
            "free_text_onomatopoeia",
            "onomatopoeia",
        )

        if bubble_onomatopoeia:
            # Marcas explícitas de preservación manual siguen ganando, pero las
            # marcas automáticas de v6 (visual_expression/skip por short hint) ya
            # no bloquean onomatopeyas dentro de globos.
            if metadata.get("preserve_original") or metadata.get("non_translatable_expression"):
                return True
            return False

        if self._metadata_bool(
            metadata,
            "skip_cleanup_translation",
            "preserve_original",
            "visual_expression",
            "non_translatable_expression",
        ):
            return True
        return False

    def explain_preserved_region(self, region: Any) -> str:
        metadata = self._metadata(region)
        region_kind = self._region_kind(region)
        if region_kind in {"dialogue", "narration", "unknown"} and self._metadata_bool(
            metadata, "bubble_onomatopoeia", "translate_inside_bubble", "free_text_onomatopoeia", "onomatopoeia"
        ) and not (metadata.get("preserve_original") or metadata.get("non_translatable_expression")):
            return "onomatopeya_en_globo_se_traduce"
        for key in ("skip_cleanup_translation", "preserve_original", "visual_expression", "non_translatable_expression"):
            if metadata.get(key):
                return str(key)
        if self._metadata_bool(metadata, "free_text_onomatopoeia_keep", "free_text_onomatopoeia", "onomatopoeia"):
            return "expresion_visual_u_onomatopeya_preservada"
        return "region_preservada"

    @classmethod
    def _longest_latin_word_len(cls, text: Any) -> int:
        normalized = cls.normalize(text)
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", normalized)
        return max((len(word) for word in words), default=0)

    def _is_reliable_region_hint(self, text: Any, region: Any = None) -> bool:
        """Devuelve True solo cuando la pista OCR global es lo bastante fiable.

        EasyOCR a veces devuelve ruido muy corto sobre texto japonés vertical, por
        ejemplo ``"a ;"``. Ese tipo de pista no debe bloquear un globo completo:
        se trata como desconocida y se deja que MangaOCR lea la región. En cambio,
        un texto latino claro como ``"EAST"`` sí es una pista fiable para preservar
        carteles/texto extranjero cuando el origen seleccionado es japonés/chino.
        """
        counts = self.signal_counts(text)
        if not self.has_meaningful_text(text):
            return False

        metadata = self._metadata(region)
        if metadata.get("ocr_global_hint_used_as_bbox_only") or metadata.get("ocr_global_hint_untrusted"):
            # Para columnas CJK verticales recuperadas desde una caja de EasyOCR,
            # la lectura global puede ser sólo una letra equivocada o texto latino
            # accidental. No uses esa pista para aceptar/rechazar la región; deja
            # que el OCR del recorte decida después.
            return False

        language = self.idioma_entrada
        region_kind = self._region_kind(region)

        if language == "Japonés":
            if counts["kana"] + counts["cjk"] > 0:
                return True
            # En texto libre/carteles, cualquier pista textual ajena se respeta
            # para no borrar arte original. En globos de diálogo somos menos
            # agresivos porque el OCR global falla mucho con texto vertical.
            if region_kind == "free_text":
                return True
            return self._longest_latin_word_len(text) >= 3 or counts["latin"] >= 4 or counts["hangul"] >= 2

        if language == "Chino":
            if counts["cjk"] > 0:
                return True
            if region_kind == "free_text":
                return True
            return self._longest_latin_word_len(text) >= 3 or counts["latin"] >= 4 or counts["hangul"] >= 2

        if language == "Coreano":
            if counts["hangul"] > 0:
                return True
            if region_kind == "free_text":
                return True
            return self._longest_latin_word_len(text) >= 3 or counts["latin"] >= 4 or counts["kana"] + counts["cjk"] >= 2

        return True

    def _reliable_region_hints(self, region: Any) -> list[str]:
        return [
            hint
            for hint in self.region_text_hints(region)
            if self._is_reliable_region_hint(hint, region)
        ]

    def should_process_region(self, region: Any, *, allow_unknown: bool = True) -> bool:
        hints = self._reliable_region_hints(region)
        if not hints:
            return bool(allow_unknown)
        combined = " ".join(hints)
        return self.has_source_language_signal(combined)

    def explain_text(self, text: Any) -> str:
        if not self.normalize(text):
            return "texto_vacio"
        if not self.has_meaningful_text(text):
            return "sin_senal_textual"
        if self.has_source_language_signal(text):
            return "idioma_origen_detectado"
        return "idioma_distinto_al_origen"

    def explain_region(self, region: Any) -> str:
        raw_hints = [hint for hint in self.region_text_hints(region) if self.has_meaningful_text(hint)]
        hints = self._reliable_region_hints(region)
        if not hints:
            return "pista_global_debil_ignorada" if raw_hints else "sin_pista_textual_global"
        if self.has_source_language_signal(" ".join(hints)):
            return "pista_global_en_idioma_origen"
        return "pista_global_en_idioma_distinto"
