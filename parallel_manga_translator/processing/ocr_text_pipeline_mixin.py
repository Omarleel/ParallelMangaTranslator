from __future__ import annotations

from typing import List, Optional, Tuple


from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.infrastructure.logging_config import get_logger

Box = Tuple[int, int, int, int]
logger = get_logger(__name__)


class OcrTextPipelineMixin:
    """OCR y normalización textual."""

    @staticmethod
    def _region_skipped_by_specialized_ocr_guard(region: Optional[TextRegion]) -> bool:
        if region is None:
            return False
        metadata = getattr(region, "metadata", {}) or {}
        return bool(
            metadata.get("specialized_ocr_guard_empty")
            or metadata.get("processing_skip_reason") == "ocr_especializado_sin_texto"
        )

    def _cached_clean_guard_ocr_text(self, region: TextRegion) -> str:
        """Reutiliza OCR local previo sólo cuando es seguro conservarlo.

        El OCR preventivo de CleanManga se hace antes de borrar una región free_text.
        Si ese mismo recorte ya fue leído, no hace falta volver a mandarlo al OCR en
        la fase de transcripción/traducción. Para columnas CJK verticales recuperadas
        desde EasyOCR, la pista global se ignora, pero un OCR local cacheado sí se
        puede reutilizar. El guard especializado previo a la limpieza también cachea
        el texto de cualquier región que sí pasó la verificación.
        """
        metadata = getattr(region, "metadata", {}) or {}
        if self._region_skipped_by_specialized_ocr_guard(region):
            return ""
        reusable = (
            metadata.get("specialized_ocr_guard_passed")
            or metadata.get("region_ocr_cache_reusable")
            or (
                region.kind == "free_text"
                and (
                    metadata.get("free_text_onomatopoeia_keep")
                    or metadata.get("free_text_onomatopoeia")
                    or metadata.get("onomatopoeia")
                )
            )
            or metadata.get("vertical_text_retry")
            or metadata.get("force_region_ocr")
        )
        if not reusable:
            return ""
        return str(metadata.get("region_ocr_text") or metadata.get("clean_guard_ocr_text") or "").strip()

    def obtener_textos(self, imagenes_interes):
        if self.ultimas_regiones and len(self.ultimas_regiones) == len(imagenes_interes):
            textos: List[str] = [""] * len(imagenes_interes)
            pendientes = []
            indices_pendientes: List[int] = []

            for indice, (imagen_interes, region) in enumerate(zip(imagenes_interes, self.ultimas_regiones)):
                if self._region_skipped_by_specialized_ocr_guard(region):
                    textos[indice] = ""
                    continue
                if not self._region_allows_source_language(region):
                    textos[indice] = ""
                    continue
                cached_text = self._cached_clean_guard_ocr_text(region)
                if cached_text and self._text_is_source_language(cached_text, region):
                    textos[indice] = self.normalizar_texto_ocr(cached_text)
                elif cached_text:
                    textos[indice] = ""
                else:
                    indices_pendientes.append(indice)
                    pendientes.append(imagen_interes)

            if pendientes:
                textos_ocr = self.ocr_manager.extract_texts(pendientes)
                for indice, texto in zip(indices_pendientes, textos_ocr):
                    texto_normalizado = self.normalizar_texto_ocr(texto)
                    region = self.ultimas_regiones[indice] if indice < len(self.ultimas_regiones) else None
                    if region is not None:
                        metadata = getattr(region, "metadata", None)
                        if isinstance(metadata, dict):
                            metadata["region_ocr_text"] = texto_normalizado
                            metadata["region_ocr_text_source"] = "translation_region_ocr"
                            metadata["region_ocr_cache_reusable"] = True
                    textos[indice] = texto_normalizado if self._text_is_source_language(texto_normalizado, region) else ""
            return textos

        textos = self.ocr_manager.extract_texts(imagenes_interes)
        textos_limpios = [self.normalizar_texto_ocr(texto) for texto in textos]
        return [texto if self._text_is_source_language(texto) else "" for texto in textos_limpios]

    def reemplazar_caracter_especial(self, texto):
        return self.text_normalizer.replace_special_characters(texto)

    def suprimir_caracteres_repetidos(self, texto, min_reps=3):
        return self.text_normalizer.suppress_repeated_characters(texto, min_reps=min_reps)

    def suprimir_simbolos_y_espacios(self, texto):
        return self.text_normalizer.suppress_symbols_and_spaces(texto)

    def normalizar_texto_ocr(self, texto: str) -> str:
        return self.text_normalizer.normalize_ocr_text(texto)
