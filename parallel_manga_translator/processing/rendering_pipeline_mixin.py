from __future__ import annotations

from typing import Optional, Tuple


from parallel_manga_translator.models.page_context import PageContext
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

    def _push_original_texts_to_queue(self, ctx: PageContext) -> None:
        if self.transcripcion_queue is None:
            return
        for idx, ((x, y, w, h), texto) in enumerate(zip(ctx.cuadros, ctx.textos_originales)):
            if idx < len(ctx.flags_idioma_origen) and not ctx.flags_idioma_origen[idx]:
                continue
            region = ctx.region_en(idx)
            estilo = ctx.estilos[idx] if idx < len(ctx.estilos) else "dialogo"
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
            if idx < len(ctx.asignaciones_hablante):
                speaker = ctx.asignaciones_hablante[idx]
                elemento.update({
                    "Hablante": speaker.get("speaker_id", "unknown"),
                    "Confianza hablante": round(float(speaker.get("confidence") or 0.0), 4),
                    "Evidencia hablante": speaker.get("evidence", ""),
                })
            self.transcripcion_queue.put({
                "agregar_a_sublista": {
                    "clave_lista": "Transcripción",
                    "pagina": ctx.indice_pagina + 1,
                    "clave_sublista": "Globos de texto",
                    "elemento_sublista": elemento,
                }
            })

    def _push_translated_texts_to_queue(self, ctx: PageContext) -> None:
        if self.traduccion_queue is None:
            return
        textos_para_render = ctx.textos_para_render or ctx.textos_traducidos
        for idx, ((x, y, w, h), texto_traducido) in enumerate(zip(ctx.cuadros, ctx.textos_traducidos)):
            if idx < len(ctx.flags_idioma_origen) and not ctx.flags_idioma_origen[idx]:
                continue
            region = ctx.region_en(idx)
            estilo = ctx.estilos[idx] if idx < len(ctx.estilos) else "dialogo"
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
            if idx < len(ctx.asignaciones_hablante):
                speaker = ctx.asignaciones_hablante[idx]
                elemento.update({
                    "Hablante": speaker.get("speaker_id", "unknown"),
                    "Confianza hablante": round(float(speaker.get("confidence") or 0.0), 4),
                })
            self.traduccion_queue.put({
                "agregar_a_sublista": {
                    "clave_lista": "Traducción",
                    "pagina": ctx.indice_pagina + 1,
                    "clave_sublista": "Globos de texto",
                    "elemento_sublista": elemento,
                }
            })

    def traducir_textos_de_regiones(self, ctx: PageContext) -> None:
        """Paso 3: normaliza, traduce y deja el rastro en las colas de JSON.

        Deja en el contexto los textos ya resueltos para rotular. El orden de las dos
        escrituras a las colas es significativo y se conserva tal cual estaba.
        """
        ctx.textos_originales = [self.normalizar_texto_ocr(texto) for texto in ctx.textos]
        self.traducir_textos(ctx)
        self._push_original_texts_to_queue(ctx)
        ctx.textos_para_render = self.resolver_textos_para_render(ctx)
        self._push_translated_texts_to_queue(ctx)

    def publicar_transcripcion(self, ctx: PageContext) -> None:
        """Escribe la transcripción en su cola sin traducir ni rotular.

        `traducir_textos_de_regiones` hace esto como parte de traducir; aquí se necesita
        suelto, porque el modo «limpiar y transcribir» tiene que dejar el mismo
        `Transcripción.json` que dejaría una ejecución completa.
        """
        ctx.textos_originales = [self.normalizar_texto_ocr(texto) for texto in ctx.textos]
        self.clasificar_pagina(ctx, ctx.textos_originales)
        self._push_original_texts_to_queue(ctx)

    def rotular(self, ctx: PageContext) -> None:
        """Paso 4: dibuja los textos ya resueltos sobre la imagen limpia."""
        cuadros_delimitadores = ctx.cuadros
        textos_para_render = ctx.textos_para_render
        alineadas = ctx.regiones_alineadas
        regiones = ctx.regiones_ordenadas
        clip_masks = [region.local_mask() for region in regiones] if alineadas else None
        # Colores del texto original, si `quality.estimate_text_colors` los estimó en el
        # paso 1. Las posiciones sin estimación van a None y el renderizador cae a su
        # regla de contraste de siempre.
        text_colors = [region.metadata.get("text_fill_color") for region in regiones] if alineadas else None
        stroke_colors = [region.metadata.get("text_stroke_color") for region in regiones] if alineadas else None
        ctx.imagen_final = self.text_renderer.render(
            ctx.imagen_limpia,
            cuadros_delimitadores,
            textos_para_render,
            text_styles=ctx.estilos,
            clip_masks=clip_masks,
            rotation_angles=[self._region_rotation_angle(region) for region in regiones]
            if alineadas
            else None,
            reading_order_right_to_left=self.reading_order_resolver.page_reads_right_to_left,
            text_colors=text_colors,
            stroke_colors=stroke_colors,
        )
