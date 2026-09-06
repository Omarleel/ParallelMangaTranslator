from __future__ import annotations

from typing import Optional, Tuple


from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.geometry.text_orientation import effective_text_rotation_angle
from parallel_manga_translator.infrastructure.logging_config import get_logger

Box = Tuple[int, int, int, int]
logger = get_logger(__name__)


class RenderingPipelineMixin:
    """Renderizado final y publicación de colas de salida."""

    def _region_rotation_angle(self, region: Optional[TextRegion]) -> float:
        if region is None:
            return 0.0
        metadata = getattr(region, "metadata", {}) or {}
        source_language = metadata.get("source_language") or getattr(self, "idioma_entrada", "")
        return effective_text_rotation_angle(metadata, source_language=source_language)

    @staticmethod
    def _es_estilo_onomatopeya(estilo: str) -> bool:
        return str(estilo or "").strip().lower().startswith("onomatopeya")

    def _push_original_texts_to_queue(self, cuadros_delimitadores, textos):
        if self.transcripcion_queue is None:
            return
        for idx, ((x, y, w, h), texto) in enumerate(zip(cuadros_delimitadores, textos)):
            if idx < len(getattr(self, "ultimos_source_language_flags", [])) and not self.ultimos_source_language_flags[idx]:
                continue
            region = self.ultimas_regiones[idx] if idx < len(self.ultimas_regiones) else None
            estilo = self.ultimo_estilos_texto[idx] if idx < len(self.ultimo_estilos_texto) else "dialogo"
            elemento = {
                "Índice": idx,
                "Coordenadas": [[x, y], [x + w, y + h]],
                "Texto": texto,
                "Estilo": estilo,
            }
            if region is not None:
                elemento.update({
                    "Tipo": region.kind,
                    "Confianza": round(float(region.confidence), 4),
                    "Coordenadas texto original": [[region.text_bbox[0], region.text_bbox[1]], [region.text_bbox[0] + region.text_bbox[2], region.text_bbox[1] + region.text_bbox[3]]],
                    "Fuente máscara": region.metadata.get("mask_source", ""),
                    "Ángulo de texto": self._region_rotation_angle(region),
                    "Ángulo detectado del original": float(region.metadata.get("text_rotation_detected_angle", region.metadata.get("text_rotation_angle", 0.0)) or 0.0),
                    "Confianza de inclinación": float(region.metadata.get("text_rotation_confidence", 0.0) or 0.0),
                })
            if idx < len(self.ultimas_asignaciones_hablante):
                speaker = self.ultimas_asignaciones_hablante[idx]
                elemento.update({
                    "Hablante": speaker.get("speaker_id", "unknown"),
                    "Confianza hablante": round(float(speaker.get("confidence") or 0.0), 4),
                    "Evidencia hablante": speaker.get("evidence", ""),
                })
            self.transcripcion_queue.put({
                "agregar_a_sublista": {
                    "clave_lista": "Transcripción",
                    "pagina": self.indice_imagen + 1,
                    "clave_sublista": "Globos de texto",
                    "elemento_sublista": elemento,
                }
            })

    def _push_translated_texts_to_queue(self, cuadros_delimitadores, textos_traducidos, textos_para_render=None):
        if self.traduccion_queue is None:
            return
        textos_para_render = textos_para_render or textos_traducidos
        for idx, ((x, y, w, h), texto_traducido) in enumerate(zip(cuadros_delimitadores, textos_traducidos)):
            if idx < len(getattr(self, "ultimos_source_language_flags", [])) and not self.ultimos_source_language_flags[idx]:
                continue
            region = self.ultimas_regiones[idx] if idx < len(self.ultimas_regiones) else None
            estilo = self.ultimo_estilos_texto[idx] if idx < len(self.ultimo_estilos_texto) else "dialogo"
            elemento = {
                "Índice": idx,
                "Coordenadas": [[x, y], [x + w, y + h]],
                "Texto": texto_traducido,
                "Estilo": estilo,
            }
            if region is not None:
                elemento.update({
                    "Tipo": region.kind,
                    "Confianza": round(float(region.confidence), 4),
                    "Fuente máscara": region.metadata.get("mask_source", ""),
                    "Ángulo de texto": self._region_rotation_angle(region),
                    "Ángulo detectado del original": float(region.metadata.get("text_rotation_detected_angle", region.metadata.get("text_rotation_angle", 0.0)) or 0.0),
                    "Confianza de inclinación": float(region.metadata.get("text_rotation_confidence", 0.0) or 0.0),
                })
                try:
                    texto_layout = textos_para_render[idx] if idx < len(textos_para_render) else texto_traducido
                    elemento["Layout UI"] = self.text_renderer.build_layout(
                        (x, y, w, h),
                        texto_layout,
                        estilo,
                        clip_mask=region.local_mask(),
                        rotation_angle=self._region_rotation_angle(region),
                        image_shape=getattr(region.mask, "shape", None),
                        reading_order_right_to_left=self.reading_order_resolver.page_reads_right_to_left,
                    )
                except Exception as exc:
                    logger.debug("No se pudo calcular layout UI para región %s: %s", idx, exc)
            if idx < len(self.ultimas_asignaciones_hablante):
                speaker = self.ultimas_asignaciones_hablante[idx]
                elemento.update({
                    "Hablante": speaker.get("speaker_id", "unknown"),
                    "Confianza hablante": round(float(speaker.get("confidence") or 0.0), 4),
                })
            self.traduccion_queue.put({
                "agregar_a_sublista": {
                    "clave_lista": "Traducción",
                    "pagina": self.indice_imagen + 1,
                    "clave_sublista": "Globos de texto",
                    "elemento_sublista": elemento,
                }
            })

    def traducir_textos_de_regiones(self, cuadros_delimitadores, textos):
        """Paso 3: normaliza, traduce y deja el rastro en las colas de JSON.

        Devuelve los textos ya resueltos para rotular. El orden de las dos escrituras a
        las colas es significativo y se conserva tal cual estaba.
        """
        textos_limpios = [self.normalizar_texto_ocr(texto) for texto in textos]
        self.ultimos_textos_originales = textos_limpios
        textos_traducidos = self.traducir_textos(textos_limpios)
        self.ultimos_textos_traducidos = textos_traducidos
        self._push_original_texts_to_queue(cuadros_delimitadores, textos_limpios)
        textos_para_render = self.resolver_textos_para_render(textos_limpios, textos_traducidos)
        self._push_translated_texts_to_queue(cuadros_delimitadores, textos_traducidos, textos_para_render)
        return textos_para_render

    def incrustar_textos(self, imagen_limpia, cuadros_delimitadores, textos):
        """Composición de traducir + rotular. Se conserva porque es la API que usa el pipeline."""
        textos_para_render = self.traducir_textos_de_regiones(cuadros_delimitadores, textos)
        return self.rotular(imagen_limpia, cuadros_delimitadores, textos_para_render)

    def rotular(self, imagen_limpia, cuadros_delimitadores, textos_para_render):
        """Paso 4: dibuja los textos ya resueltos sobre la imagen limpia."""
        clip_masks = [region.local_mask() for region in self.ultimas_regiones] if self.ultimas_regiones and len(self.ultimas_regiones) == len(cuadros_delimitadores) else None
        return self.text_renderer.render(
            imagen_limpia,
            cuadros_delimitadores,
            textos_para_render,
            text_styles=self.ultimo_estilos_texto,
            clip_masks=clip_masks,
            rotation_angles=[self._region_rotation_angle(region) for region in self.ultimas_regiones]
            if self.ultimas_regiones and len(self.ultimas_regiones) == len(cuadros_delimitadores)
            else None,
            reading_order_right_to_left=self.reading_order_resolver.page_reads_right_to_left,
        )
