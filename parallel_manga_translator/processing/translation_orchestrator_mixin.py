from __future__ import annotations

from typing import Tuple


from parallel_manga_translator.infrastructure.logging_config import get_logger

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
        return cuadros_delimitadores, imagenes_interes

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
