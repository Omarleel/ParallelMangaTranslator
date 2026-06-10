from __future__ import annotations

import os
import re
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from Applications.CleanManga import CleanManga
from Applications.FileManager import FileManager
from Applications.TranslateManga import TranslateManga
from Applications.MetricsManager import MetricsWriter, PageMetrics
from Applications.ErrorHandling import (
    PageFailureReport,
    StageProcessingError,
    processing_stage,
    unwrap_original_exception,
    write_failure_report,
)
from .LoggingConfig import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    def __init__(self, idioma_entrada, idioma_salida, modelo_inpaint, metodo_traduccion="Tradicional", groq_api_key="", lore_manga=""):
        self.file_manager = FileManager()
        self.clean_manga = CleanManga(modelo_inpaint, idioma_entrada=idioma_entrada)
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
        root = unwrap_original_exception(exc)
        message = str(root).lower()
        return isinstance(root, torch.cuda.OutOfMemoryError) or "cuda" in message or "out of memory" in message

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
                self._registrar_fallo(ruta_traduccion_salida, indice_imagen, nuevo_archivo, image_path, exc)
            finally:
                del imagen
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _registrar_fallo(self, ruta_traduccion_salida: str, indice_imagen: int, archivo: str, image_path: str, exc: Exception) -> None:
        output_root = str(Path(ruta_traduccion_salida).parent)
        fallidas = Path(output_root) / "Fallidas"
        fallidas.mkdir(parents=True, exist_ok=True)
        try:
            if Path(image_path).exists():
                shutil.copy2(image_path, fallidas / Path(image_path).name)
        except Exception as copy_exc:
            logger.warning("No se pudo copiar imagen fallida %s: %s", image_path, copy_exc)

        report = PageFailureReport.from_exception(
            exc,
            page_index=indice_imagen,
            filename=archivo,
            metadata={"source_path": image_path},
        )
        try:
            report_path = write_failure_report(output_root, report)
            logger.error("Reporte de fallo guardado: %s", report_path)
        except Exception as report_exc:
            logger.warning("No se pudo escribir reporte de fallo estructurado: %s", report_exc)

        metrics = PageMetrics(
            page_index=indice_imagen,
            filename=archivo,
            status="failed",
            error=f"{report.stage}: {report.error_type}: {report.error_message}",
        )
        MetricsWriter(output_root).write_page(metrics)

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
        output_root = str(Path(ruta_traduccion_salida).parent)
        metrics = PageMetrics(
            page_index=indice_imagen,
            filename=archivo,
            image_width=int(imagen.shape[1]),
            image_height=int(imagen.shape[0]),
        )

        for intento in range(1, max_retries + 1):
            try:
                t0 = time.perf_counter()
                with processing_stage("limpieza", logger=logger, page_index=indice_imagen, filename=archivo):
                    mascara_capa, imagen_limpia, regiones = self.clean_manga.limpiar_manga(imagen_actual)
                metrics.timings["limpieza"] = round(time.perf_counter() - t0, 4)
                metrics.detected_regions = len(regiones)
                metrics.detected_bubbles = sum(1 for r in regiones if r.kind in {"dialogue", "narration", "unknown"})
                metrics.detected_sfx = sum(1 for r in regiones if r.kind in {"sfx", "free_text"})

                archivo_limpieza_salida = os.path.join(ruta_limpieza_salida, archivo)
                with processing_stage("guardar_limpieza", logger=logger, page_index=indice_imagen, filename=archivo):
                    self._write_image(archivo_limpieza_salida, imagen_limpia)

                self.translate_manga.insertar_json_queue(
                    indice_imagen=indice_imagen,
                    transcripcion_queue=transcripcion_queue,
                    traduccion_queue=traduccion_queue,
                )
                t1 = time.perf_counter()
                with processing_stage("ocr_traduccion_render", logger=logger, page_index=indice_imagen, filename=archivo):
                    imagen_traducida = self.translate_manga.traducir_manga(imagen_actual, imagen_limpia, mascara_capa, text_regions=regiones)
                metrics.timings["ocr_traduccion_render"] = round(time.perf_counter() - t1, 4)
                metrics.ocr_empty = sum(1 for t in getattr(self.translate_manga, "ultimos_textos_originales", []) if not str(t).strip())
                metrics.translations_empty = sum(1 for t in getattr(self.translate_manga, "ultimos_textos_traducidos", []) if not str(t).strip())

                archivo_traduccion_salida = os.path.join(ruta_traduccion_salida, archivo)
                with processing_stage("guardar_traduccion", logger=logger, page_index=indice_imagen, filename=archivo):
                    self._write_image(archivo_traduccion_salida, imagen_traducida)
                metrics.retries = intento - 1
                MetricsWriter(output_root).write_page(metrics)
                return
            except (torch.cuda.OutOfMemoryError, RuntimeError, StageProcessingError) as exc:
                logger.warning("Error potencial de memoria al procesar %s (intento %s/%s): %s", archivo, intento, max_retries, exc)
                if intento >= max_retries or not self._is_retryable_memory_error(exc):
                    metrics.status = "failed"
                    metrics.error = str(exc)
                    metrics.retries = intento - 1
                    MetricsWriter(output_root).write_page(metrics)
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