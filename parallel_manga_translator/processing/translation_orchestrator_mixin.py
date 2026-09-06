from __future__ import annotations

from typing import Tuple


from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.rendering.text_color_estimator import estimar_colores

Box = Tuple[int, int, int, int]
logger = get_logger(__name__)


class TranslationOrchestratorMixin:
    """Orquestación de alto nivel para traducir una página."""

    def insertar_json_queue(self, indice_imagen, transcripcion_queue, traduccion_queue):
        self.indice_imagen = indice_imagen
        self.transcripcion_queue = transcripcion_queue
        self.traduccion_queue = traduccion_queue

    def extraer_regiones(self, imagen, mascara_capa, text_regions=None):
        """Paso 1: qué zonas de la página llevan texto y sus recortes.

        Devuelve ``(cuadros_delimitadores, imagenes_interes)`` y deja las regiones
        ordenadas en ``ultimas_regiones``, que consumen los pasos siguientes.
        """
        if text_regions:
            cuadros_delimitadores, imagenes_interes, regiones_ordenadas = self.obtener_areas_interes_desde_regiones(imagen, text_regions)
            self.ultimas_regiones = regiones_ordenadas
        else:
            cuadros_delimitadores, imagenes_interes = self.obtener_areas_interes(imagen, mascara_capa)
            self.ultimas_regiones = []
        self._estimar_colores_de_regiones(imagen)
        return cuadros_delimitadores, imagenes_interes

    def _estimar_colores_de_regiones(self, imagen) -> None:
        """Anota en cada region el color de su tinta y de su contorno originales.

        Se hace aqui porque es el unico punto donde conviven la imagen ORIGINAL y las
        mascaras de tinta ya calculadas: en el rotulado solo queda la imagen limpia, de la
        que el texto original ya fue borrado.

        Se trabaja sobre el recorte de cada region, no sobre la pagina entera: con mascaras
        del tamano de la pagina esto multiplicaba el coste sin cambiar el resultado.
        """
        if not getattr(self, "estimate_text_colors", False) or imagen is None:
            return
        for region in self.ultimas_regiones or []:
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

    def traducir_manga(self, imagen, imagen_limpia, mascara_capa, text_regions=None):
        """Los cuatro pasos encadenados. Cada uno se puede invocar por separado:

            cuadros, recortes = tm.extraer_regiones(imagen, mascara, regiones)
            textos = tm.obtener_textos(recortes)          # aquí acaba un modo solo-OCR
            para_render = tm.traducir_textos_de_regiones(cuadros, textos)
            salida = tm.rotular(imagen_limpia, cuadros, para_render)

        `eval_runner` ya reconstruía a mano esa secuencia parcial para medir sin
        traducir; ahora puede usar los mismos pasos que el pipeline real.
        """
        cuadros_delimitadores, imagenes_interes = self.extraer_regiones(imagen, mascara_capa, text_regions)
        textos = self.obtener_textos(imagenes_interes)
        return self.incrustar_textos(imagen_limpia, cuadros_delimitadores, textos)
