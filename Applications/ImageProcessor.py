from __future__ import annotations

import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from Applications.CleanManga import CleanManga
from Applications.FileManager import FileManager
from Applications.TranslateManga import TranslateManga
from .LoggingConfig import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    def __init__(self, idioma_entrada, idioma_salida, modelo_inpaint, metodo_traduccion="Tradicional", groq_api_key="", lore_manga=""):
        self.file_manager = FileManager()
        self.clean_manga = CleanManga(modelo_inpaint)
        self.translate_manga = TranslateManga(
            idioma_entrada,
            idioma_salida,
            metodo_traduccion=metodo_traduccion,
            groq_api_key=groq_api_key,
            lore_manga=lore_manga,
        )

    @staticmethod
    def _read_image(image_path: str):
        with open(image_path, "rb") as file_handle:
            byte_array = file_handle.read()
        image_nparr = np.frombuffer(byte_array, np.uint8)
        return cv2.imdecode(image_nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def _write_image(output_path: str, imagen) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, imagen)

    @staticmethod
    def _is_retryable_memory_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda" in message or "out of memory" in message

    def procesar(self, ruta_carpeta_entrada, ruta_limpieza_salida, ruta_traduccion_salida, lote, transcripcion_queue, traduccion_queue):
        for indice_imagen, archivo in lote.items():
            nombre_base, ext = os.path.splitext(archivo)
            if ext.lower() == ".webp":
                ext = ".jpg"
            
            match = re.search(r'(\d+)', nombre_base)
            if match:
                numero_archivo = int(match.group(1))
                nuevo_archivo = f"{numero_archivo:04d}{ext}"
            else:
                nuevo_archivo = f"{nombre_base}{ext}"

            archivo_limpieza_esperado = os.path.join(ruta_limpieza_salida, nuevo_archivo)
            archivo_traduccion_esperado = os.path.join(ruta_traduccion_salida, nuevo_archivo)

            if os.path.exists(archivo_limpieza_esperado) and os.path.exists(archivo_traduccion_esperado):
                logger.info("Omitiendo %s: La imagen ya fue procesada en una ejecución anterior.", nuevo_archivo)
                continue

            logger.info("Procesando archivo: %s", archivo)
            image_path = os.path.join(ruta_carpeta_entrada, archivo)
            imagen = self._read_image(image_path)
            if imagen is None:
                logger.error("No se pudo leer la imagen: %s", image_path)
                continue

            self._registrar_pagina(transcripcion_queue, "Transcripción", indice_imagen, imagen)
            self._registrar_pagina(traduccion_queue, "Traducción", indice_imagen, imagen)

            try:
                self._process_with_retry(
                    indice_imagen=indice_imagen,
                    archivo=nuevo_archivo,
                    imagen=imagen,
                    ruta_limpieza_salida=ruta_limpieza_salida,
                    ruta_traduccion_salida=ruta_traduccion_salida,
                    transcripcion_queue=transcripcion_queue,
                    traduccion_queue=traduccion_queue,
                )
            except Exception as exc:
                logger.exception("Fallo definitivo al procesar %s: %s", archivo, exc)
            finally:
                del imagen
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _registrar_pagina(self, queue, tipo: str, indice_imagen: int, imagen) -> None:
        queue.put({
            "agregar_elemento_a_lista": {
                tipo: {
                    "Página": indice_imagen + 1,
                    "Formato": self.obtener_formato_manga(imagen),
                    "Globos de texto": [],
                }
            }
        })

    def _process_with_retry(
        self,
        indice_imagen,
        archivo,
        imagen,
        ruta_limpieza_salida,
        ruta_traduccion_salida,
        transcripcion_queue,
        traduccion_queue,
        max_retries: int = 3,
    ) -> None:
        imagen_actual = imagen

        for intento in range(1, max_retries + 1):
            try:
                mascara_capa, imagen_limpia = self.clean_manga.limpiar_manga(imagen_actual)
                archivo_limpieza_salida = os.path.join(ruta_limpieza_salida, archivo)
                self._write_image(archivo_limpieza_salida, imagen_limpia)

                self.translate_manga.insertar_json_queue(
                    indice_imagen=indice_imagen,
                    transcripcion_queue=transcripcion_queue,
                    traduccion_queue=traduccion_queue,
                )
                imagen_traducida = self.translate_manga.traducir_manga(imagen_actual, imagen_limpia, mascara_capa)
                archivo_traduccion_salida = os.path.join(ruta_traduccion_salida, archivo)
                self._write_image(archivo_traduccion_salida, imagen_traducida)
                return
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                logger.warning("Error potencial de memoria al procesar %s (intento %s/%s): %s", archivo, intento, max_retries, exc)
                if intento >= max_retries or not self._is_retryable_memory_error(exc):
                    raise
                imagen_actual = self.reducir_imagen(imagen_actual)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1)
            except Exception:
                raise

    @staticmethod
    def obtener_formato_manga(imagen):
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        valor_medio = cv2.mean(imagen_gris)[0]
        if valor_medio < 50 or valor_medio > 200:
            return "Blanco y negro (B/N)"
        return "Color"

    @staticmethod
    def reducir_imagen(imagen):
        porcentaje_reduccion = 0.75
        nuevo_alto, nuevo_ancho = [max(1, int(dim * porcentaje_reduccion)) for dim in imagen.shape[:2]]
        return cv2.resize(imagen, (nuevo_ancho, nuevo_alto))