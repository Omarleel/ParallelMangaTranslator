from __future__ import annotations

from typing import Tuple


from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.page_context import PageContext
from parallel_manga_translator.rendering.text_color_estimator import estimar_colores

Box = Tuple[int, int, int, int]
logger = get_logger(__name__)


class TranslationOrchestratorMixin:
    """Orquestación de alto nivel para traducir una página."""

    def insertar_json_queue(self, transcripcion_queue, traduccion_queue):
        """Las colas de salida del trabajo. El índice de página ya no viene por aquí:
        es estado de la página y viaja en `PageContext.indice_pagina`."""
        self.transcripcion_queue = transcripcion_queue
        self.traduccion_queue = traduccion_queue

    def extraer_regiones(self, ctx: PageContext) -> None:
        """Paso 1: qué zonas de la página llevan texto y sus recortes.

        Deja en el contexto `cuadros`, `recortes` y `regiones_ordenadas`. Los pasos
        siguientes las leen de ahí; antes se las pasaban por un atributo del traductor.
        """
        if ctx.regiones:
            ctx.cuadros, ctx.recortes, ctx.regiones_ordenadas = self.obtener_areas_interes_desde_regiones(
                ctx.imagen, ctx.regiones
            )
        else:
            ctx.cuadros, ctx.recortes = self.obtener_areas_interes(ctx.imagen, ctx.mascara_capa)
            ctx.regiones_ordenadas = []
        self._estimar_colores_de_regiones(ctx)

    def _estimar_colores_de_regiones(self, ctx: PageContext) -> None:
        """Anota en cada region el color de su tinta y de su contorno originales.

        Se hace aqui porque es el unico punto donde conviven la imagen ORIGINAL y las
        mascaras de tinta ya calculadas: en el rotulado solo queda la imagen limpia, de la
        que el texto original ya fue borrado.

        Se trabaja sobre el recorte de cada region, no sobre la pagina entera: con mascaras
        del tamano de la pagina esto multiplicaba el coste sin cambiar el resultado.
        """
        imagen = ctx.imagen
        if not self.estimate_text_colors or imagen is None:
            return
        for region in ctx.regiones_ordenadas:
            tinta = region.text_mask if getattr(region, "text_mask", None) is not None else region.clean_mask
            if tinta is None:
                continue
            x, y, w, h = region.bbox
            margen = 12   # holgura para muestrear el anillo exterior y el fondo
            x0, y0 = max(0, x - margen), max(0, y - margen)
            x1, y1 = min(imagen.shape[1], x + w + margen), min(imagen.shape[0], y + h + margen)
            recorte = imagen[y0:y1, x0:x1]
            recorte_tinta = tinta[y0:y1, x0:x1]
            zona = region.mask[y0:y1, x0:x1] if getattr(region, "mask", None) is not None else None
            try:
                colores = estimar_colores(recorte, recorte_tinta, zona_segura=zona)
            except Exception as exc:
                logger.debug("No se pudo estimar el color de una region: %s", exc)
                continue
            if colores is None:
                continue
            region.metadata["text_fill_color"] = list(colores.relleno)
            if colores.contorno is not None:
                region.metadata["text_stroke_color"] = list(colores.contorno)
            if colores.separacion_relleno is not None:
                # Diagnostico, no se usa para rotular: dice si un contorno se descarto por
                # poco o por mucho.
                region.metadata["text_stroke_sep"] = [
                    round(colores.separacion_relleno, 1),
                    round(colores.separacion_fondo, 1) if colores.separacion_fondo is not None else None,
                ]

    def traducir_manga(self, imagen, imagen_limpia, mascara_capa, text_regions=None, indice_pagina=0):
        """Los cuatro pasos encadenados sobre un mismo contexto:

            ctx = PageContext(imagen=imagen, imagen_limpia=limpia, mascara_capa=mascara)
            tm.extraer_regiones(ctx)
            tm.obtener_textos(ctx)               # aquí acaba un modo solo-OCR
            tm.traducir_textos_de_regiones(ctx)
            tm.rotular(ctx)

        Se conserva porque es una API cómoda para una página suelta. El pipeline real no
        pasa por aquí: compone las mismas cuatro etapas en `processing/pipeline.py`.
        """
        ctx = PageContext(
            imagen=imagen,
            imagen_limpia=imagen_limpia,
            mascara_capa=mascara_capa,
            regiones=list(text_regions or []),
            indice_pagina=indice_pagina,
        )
        self.extraer_regiones(ctx)
        self.obtener_textos(ctx)
        self.traducir_textos_de_regiones(ctx)
        self.rotular(ctx)
        return ctx.imagen_final
