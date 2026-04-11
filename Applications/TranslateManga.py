from __future__ import annotations

import re
from collections import deque
from typing import Sequence

import cv2
import numpy as np
import torch
import transformers

from Applications.OcrManager import OcrManager
from Applications.TextRendering import TextRenderer
from Applications.TranslatorManager import TranslatorManager
from Utils.Constantes import RUTA_FUENTE, TAMANIO_MINIMO_FUENTE

transformers.logging.set_verbosity_error()


class TranslateManga:
    def __init__(self, idioma_entrada, idioma_salida, metodo_traduccion="Tradicional", groq_api_key="", lore_manga=""):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.idioma_entrada = idioma_entrada
        self.idioma_salida = idioma_salida
        self.metodo_traduccion = metodo_traduccion

        self.translator_manager = TranslatorManager(
            idioma_entrada,
            idioma_salida,
            metodo=metodo_traduccion,
            groq_api_key=groq_api_key,
            lore_manga=lore_manga,
        )
        self.ocr_manager = OcrManager(idioma_entrada=idioma_entrada)
        self.text_renderer = TextRenderer(font_path=RUTA_FUENTE, min_font_size=TAMANIO_MINIMO_FUENTE)
        self.historial_contexto = deque(maxlen=4)
        self.indice_imagen = 0
        self.transcripcion_queue = None
        self.traduccion_queue = None

    def insertar_json_queue(self, indice_imagen, transcripcion_queue, traduccion_queue):
        self.indice_imagen = indice_imagen
        self.transcripcion_queue = transcripcion_queue
        self.traduccion_queue = traduccion_queue

    def traducir_manga(self, imagen, imagen_limpia, mascara_capa):
        cuadros_delimitadores, imagenes_interes = self.obtener_areas_interes(imagen, mascara_capa)
        textos = self.obtener_textos(imagenes_interes)
        return self.incrustar_textos(imagen_limpia, cuadros_delimitadores, textos)

    def obtener_areas_interes(self, imagen, mascara_capa):
        cuadros_delimitadores = []
        imagenes_interes = []
        height, width = mascara_capa.shape

        if self.idioma_entrada == "Japonés":
            kernel_h, kernel_w = round(height * 0.003215), round(width * 0.003215)
        else:
            kernel_h, kernel_w = round(height * 0.005), round(width * 0.03)

        kernel_h = max(1, kernel_h)
        kernel_w = max(1, kernel_w)

        _, mascara_binaria = cv2.threshold(mascara_capa, 127, 255, cv2.THRESH_BINARY)
        mascara_binaria = np.uint8(mascara_binaria)
        kernel = np.ones((kernel_h, kernel_w), np.uint8)
        mascara_dilatada = cv2.dilate(mascara_binaria, kernel, iterations=1)
        contours, _ = cv2.findContours(mascara_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        for x, y, w, h in sorted(boxes, key=lambda item: (item[1], item[0])):
            area_interes = imagen[y:y + h, x:x + w]
            cuadros_delimitadores.append((x, y, w, h))
            imagenes_interes.append(area_interes)

        return cuadros_delimitadores, imagenes_interes

    def obtener_textos(self, imagenes_interes):
        return self.ocr_manager.extract_texts(imagenes_interes)

    @staticmethod
    def reemplazar_caracter_especial(texto):
        caracteres_especiales = {
            "。": ".",
            "·": ".",
            "？": "?",
            "．": ".",
            "・": ".",
            "！": "!",
            "０": "",
        }
        for especial, normal in caracteres_especiales.items():
            texto = texto.replace(especial, normal)
        return texto

    @staticmethod
    def suprimir_caracteres_repetidos(texto, min_reps=3):
        patron = r"(.)\1{{{},}}".format(min_reps)

        def reemplazo(match):
            return match.group(1) * 3

        return re.sub(patron, reemplazo, texto)

    @staticmethod
    def suprimir_simbolos_y_espacios(texto):
        for char in texto:
            if char.isalnum():
                return texto
        return ""

    def traducir_textos(self, textos: Sequence[str]):
        if self.metodo_traduccion == "LLM":
            textos_traducidos_brutos = self.translator_manager.traducir_textos_llm(textos, list(self.historial_contexto))
        else:
            textos_traducidos_brutos = []
            for texto in textos:
                traducido = self.translator_manager.traducir_texto(texto)
                if traducido is None or len(traducido) <= 1:
                    traducido = ""
                textos_traducidos_brutos.append(traducido)

        textos_traducidos_limpios = []
        for texto_traducido in textos_traducidos_brutos:
            texto_traducido = self.reemplazar_caracter_especial(texto_traducido).strip()
            texto_traducido = self.suprimir_caracteres_repetidos(texto_traducido)
            texto_traducido = self.suprimir_simbolos_y_espacios(texto_traducido)
            textos_traducidos_limpios.append(texto_traducido)

        if self.metodo_traduccion == "LLM":
            contexto_bilingue = [
                f"{orig} -> {trad}"
                for orig, trad in zip(textos, textos_traducidos_limpios)
                if orig.strip() and trad.strip()
            ]
            if contexto_bilingue:
                self.historial_contexto.append(contexto_bilingue)

        return textos_traducidos_limpios

    def _push_original_texts_to_queue(self, cuadros_delimitadores, textos):
        if self.transcripcion_queue is None:
            return
        for (x, y, w, h), texto in zip(cuadros_delimitadores, textos):
            self.transcripcion_queue.put({
                "agregar_a_sublista": {
                    "clave_lista": "Transcripción",
                    "pagina": self.indice_imagen + 1,
                    "clave_sublista": "Globos de texto",
                    "elemento_sublista": {
                        "Coordenadas": [[x, y], [x + w, y + h]],
                        "Texto": texto,
                    },
                }
            })

    def _push_translated_texts_to_queue(self, cuadros_delimitadores, textos_traducidos):
        if self.traduccion_queue is None:
            return
        for (x, y, w, h), texto_traducido in zip(cuadros_delimitadores, textos_traducidos):
            self.traduccion_queue.put({
                "agregar_a_sublista": {
                    "clave_lista": "Traducción",
                    "pagina": self.indice_imagen + 1,
                    "clave_sublista": "Globos de texto",
                    "elemento_sublista": {
                        "Coordenadas": [[x, y], [x + w, y + h]],
                        "Texto": texto_traducido,
                    },
                }
            })

    def incrustar_textos(self, imagen_limpia, cuadros_delimitadores, textos):
        self._push_original_texts_to_queue(cuadros_delimitadores, textos)
        textos_traducidos = self.traducir_textos(textos)
        self._push_translated_texts_to_queue(cuadros_delimitadores, textos_traducidos)
        return self.text_renderer.render(imagen_limpia, cuadros_delimitadores, textos_traducidos)