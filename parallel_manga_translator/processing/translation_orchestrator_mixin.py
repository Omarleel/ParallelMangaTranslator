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

    def traducir_manga(self, imagen, imagen_limpia, mascara_capa, text_regions=None):
        if text_regions:
            cuadros_delimitadores, imagenes_interes, regiones_ordenadas = self.obtener_areas_interes_desde_regiones(imagen, text_regions)
            self.ultimas_regiones = regiones_ordenadas
        else:
            cuadros_delimitadores, imagenes_interes = self.obtener_areas_interes(imagen, mascara_capa)
            self.ultimas_regiones = []
        textos = self.obtener_textos(imagenes_interes)
        return self.incrustar_textos(imagen_limpia, cuadros_delimitadores, textos)
