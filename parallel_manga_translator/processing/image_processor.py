from __future__ import annotations

import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from parallel_manga_translator.processing.clean_manga import CleanManga
from parallel_manga_translator.io.file_manager import FileManager
from parallel_manga_translator.io.image_naming import normalized_page_output_name
from parallel_manga_translator.processing.translate_manga import TranslateManga
from parallel_manga_translator.quality.metrics_manager import MetricsWriter, PageMetrics
from parallel_manga_translator.infrastructure.error_handling import (
    PageFailureReport,
    StageProcessingError,
    processing_stage,
    unwrap_original_exception,
    write_failure_report,
)
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.infrastructure.execution_control import JobControlError, cooperative_sleep, execution_checkpoint
from parallel_manga_translator.infrastructure.gpu_scheduler import (
    gpu_scheduler_snapshot,
    reset_gpu_scheduler_stats,
)
from parallel_manga_translator.config.app_config import CharacterMemoryConfig, OcrConfig, OnomatopoeiaConfig, ProcessingConfig, QualityConfig, TranslationConfig

logger = get_logger(__name__)


@dataclass
class PreparedPage:
    indice_imagen: int
    source_filename: str
    output_filename: str
    image_path: str
    imagen_original: np.ndarray
    imagen_limpia: np.ndarray
    mascara_capa: np.ndarray
    regiones: list[Any]
    metrics: PageMetrics


_PIPELINE_STOP = object()


class ImageProcessor:
    def __init__(
        self,
        idioma_entrada,
        idioma_salida,
        modelo_inpaint,
        metodo_traduccion="Tradicional",
        groq_api_key="",
        lore_manga="",
        ocr_config: OcrConfig | None = None,
        translation_config: TranslationConfig | None = None,
        processing_config: ProcessingConfig | None = None,
        quality_config: QualityConfig | None = None,
        onomatopoeia_config: OnomatopoeiaConfig | None = None,
        character_memory_config: CharacterMemoryConfig | None = None,
    ):
        self.file_manager = FileManager()
        self.clean_manga = CleanManga(
            modelo_inpaint,
            idioma_entrada=idioma_entrada,
            quality_config=quality_config,
            onomatopoeia_config=onomatopoeia_config,
            processing_config=processing_config,
            ocr_config=ocr_config,
        )
        self.translate_manga = TranslateManga(
            idioma_entrada,
            idioma_salida,
            metodo_traduccion=metodo_traduccion,
            groq_api_key=groq_api_key,
            lore_manga=lore_manga,
            ocr_config=ocr_config,
            translation_config=translation_config,
            quality_config=quality_config,
            onomatopoeia_config=onomatopoeia_config,
            character_memory_config=character_memory_config,
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
            execution_checkpoint()
            nuevo_archivo = normalized_page_output_name(archivo, indice_imagen)

            archivo_limpieza_esperado = os.path.join(ruta_limpieza_salida, nuevo_archivo)
            archivo_traduccion_esperado = os.path.join(ruta_traduccion_salida, nuevo_archivo)

            if os.path.exists(archivo_limpieza_esperado) and os.path.exists(archivo_traduccion_esperado):
                logger.info("Omitiendo %s: La imagen ya fue procesada en una ejecución anterior.", nuevo_archivo)
                continue

            logger.info("Procesando archivo: %s -> %s", archivo, nuevo_archivo)
            image_path = os.path.join(ruta_carpeta_entrada, archivo)
            imagen = self._read_image(image_path)
            if imagen is None:
                logger.error("No se pudo leer la imagen: %s", image_path)
                continue

            self._registrar_pagina(transcripcion_queue, "Transcripción", indice_imagen, imagen)
            self._registrar_pagina(traduccion_queue, "Traducción", indice_imagen, imagen)

            self.clean_manga.set_debug_page_context(
                indice_imagen,
                source_filename=archivo,
                output_filename=nuevo_archivo,
            )
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
            except JobControlError:
                raise
            except Exception as exc:
                logger.exception("Fallo definitivo al procesar %s: %s", archivo, exc)
                self._registrar_fallo(ruta_traduccion_salida, indice_imagen, nuevo_archivo, image_path, exc)
            finally:
                self.clean_manga.clear_debug_page_context()
                del imagen
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def procesar_pipeline(
        self,
        ruta_carpeta_entrada,
        ruta_limpieza_salida,
        ruta_traduccion_salida,
        lote: Mapping[int, str],
        transcripcion_queue,
        traduccion_queue,
        prefetch: int = 2,
    ) -> None:
        """Procesa páginas con una tubería productor/consumidor dentro del mismo proceso.

        El productor mantiene una única copia de los modelos y prepara la página
        siguiente. El hilo principal consume en orden y ejecuta OCR, traducción y render.
        Si el OCR usa CUDA, un planificador FIFO comparte la GPU entre YOLO/OCR/inpainting
        mientras las secciones CPU de ambos hilos continúan solapadas.
        """
        pendientes: list[tuple[int, str, str]] = []
        for indice_imagen, archivo in lote.items():
            nuevo_archivo = normalized_page_output_name(archivo, indice_imagen)
            archivo_limpieza = os.path.join(ruta_limpieza_salida, nuevo_archivo)
            archivo_traduccion = os.path.join(ruta_traduccion_salida, nuevo_archivo)
            if os.path.exists(archivo_limpieza) and os.path.exists(archivo_traduccion):
                logger.info("Omitiendo %s: La imagen ya fue procesada en una ejecución anterior.", nuevo_archivo)
                continue
            pendientes.append((indice_imagen, archivo, nuevo_archivo))

        if not pendientes:
            return

        capacidad = max(1, min(4, int(prefetch)))
        pipeline_started = time.perf_counter()
        reset_gpu_scheduler_stats()
        prepared_queue: queue.Queue = queue.Queue(maxsize=capacidad)
        cancelar = threading.Event()
        producer_errors: list[BaseException] = []
        prepared_count = [0]
        finished_count = 0

        def publicar(item) -> bool:
            while not cancelar.is_set():
                try:
                    prepared_queue.put(item, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        def productor_gpu() -> None:
            try:
                for indice_imagen, archivo, nuevo_archivo in pendientes:
                    if cancelar.is_set():
                        break
                    logger.info(
                        "Pipeline PREP | preparando %s -> %s | cola=%s/%s",
                        archivo,
                        nuevo_archivo,
                        prepared_queue.qsize(),
                        capacidad,
                    )
                    image_path = os.path.join(ruta_carpeta_entrada, archivo)
                    imagen = self._read_image(image_path)
                    if imagen is None:
                        logger.error("No se pudo leer la imagen: %s", image_path)
                        continue

                    self._registrar_pagina(transcripcion_queue, "Transcripción", indice_imagen, imagen)
                    self._registrar_pagina(traduccion_queue, "Traducción", indice_imagen, imagen)
                    self.clean_manga.set_debug_page_context(
                        indice_imagen,
                        source_filename=archivo,
                        output_filename=nuevo_archivo,
                    )
                    try:
                        preparada = self._prepare_page_with_retry(
                            indice_imagen=indice_imagen,
                            source_filename=archivo,
                            output_filename=nuevo_archivo,
                            image_path=image_path,
                            imagen=imagen,
                            ruta_limpieza_salida=ruta_limpieza_salida,
                            ruta_traduccion_salida=ruta_traduccion_salida,
                        )
                        if not publicar(preparada):
                            break
                        prepared_count[0] += 1
                    except JobControlError:
                        raise
                    except Exception as exc:
                        logger.exception("Fallo en preparación GPU de %s: %s", archivo, exc)
                        self._registrar_fallo(
                            ruta_traduccion_salida, indice_imagen, nuevo_archivo, image_path, exc
                        )
                    finally:
                        self.clean_manga.clear_debug_page_context()
                        del imagen
            except BaseException as exc:  # comunica fallos inesperados al consumidor
                producer_errors.append(exc)
                logger.exception("Fallo inesperado en el productor PREP/GPU: %s", exc)
            finally:
                publicar(_PIPELINE_STOP)

        productor = threading.Thread(
            target=productor_gpu,
            name="pmt-prep-gpu",
            daemon=True,
        )
        productor.start()

        try:
            while True:
                item = prepared_queue.get()
                try:
                    if item is _PIPELINE_STOP:
                        break
                    assert isinstance(item, PreparedPage)
                    logger.info(
                        "Pipeline FINAL | OCR/traducción/render %s | preparadas_en_cola=%s",
                        item.output_filename,
                        prepared_queue.qsize(),
                    )
                    try:
                        self._finish_prepared_page(
                            item,
                            ruta_traduccion_salida=ruta_traduccion_salida,
                            transcripcion_queue=transcripcion_queue,
                            traduccion_queue=traduccion_queue,
                        )
                        finished_count += 1
                    except JobControlError:
                        raise
                    except Exception as exc:
                        logger.exception(
                            "Fallo en OCR/traducción/render de %s: %s",
                            item.source_filename,
                            exc,
                        )
                        self._registrar_fallo(
                            ruta_traduccion_salida,
                            item.indice_imagen,
                            item.output_filename,
                            item.image_path,
                            exc,
                        )
                    finally:
                        del item
                finally:
                    prepared_queue.task_done()
        finally:
            cancelar.set()
            productor.join()
            if torch.cuda.is_available():
                # Una sola limpieza al cerrar la tubería; hacerlo por página puede
                # sincronizar CUDA y destruir el beneficio del solapamiento.
                torch.cuda.empty_cache()
            logger.info(
                "Pipeline CPU/GPU finalizada | preparadas=%s | completadas=%s | tiempo=%.2fs",
                prepared_count[0],
                finished_count,
                time.perf_counter() - pipeline_started,
            )
            gpu_stats = gpu_scheduler_snapshot()
            if gpu_stats.operations:
                logger.info(
                    "Planificador GPU | operaciones=%s | espera_total=%.2fs | "
                    "ocupacion_exclusiva=%.2fs | detalle=%s",
                    gpu_stats.operations,
                    gpu_stats.wait_seconds,
                    gpu_stats.hold_seconds,
                    gpu_stats.by_operation,
                )

        if producer_errors:
            raise RuntimeError("El productor PREP/GPU terminó inesperadamente") from producer_errors[0]

    def _prepare_page_with_retry(
        self,
        *,
        indice_imagen: int,
        source_filename: str,
        output_filename: str,
        image_path: str,
        imagen: np.ndarray,
        ruta_limpieza_salida: str,
        ruta_traduccion_salida: str,
        max_retries: int = 3,
    ) -> PreparedPage:
        imagen_actual = imagen
        output_root = str(Path(ruta_traduccion_salida).parent)
        metrics = PageMetrics(
            page_index=indice_imagen,
            filename=output_filename,
            image_width=int(imagen.shape[1]),
            image_height=int(imagen.shape[0]),
        )

        for intento in range(1, max_retries + 1):
            try:
                t0 = time.perf_counter()
                with processing_stage(
                    "limpieza", logger=logger, page_index=indice_imagen, filename=output_filename
                ):
                    if bool(getattr(self.clean_manga, "visual_inpaint_debug", False)):
                        self.clean_manga.set_visual_inpaint_debug_context(
                            output_root=output_root,
                            page_index=indice_imagen,
                            filename=output_filename,
                        )
                    else:
                        self.clean_manga.clear_visual_inpaint_debug_context()
                    mascara_capa, imagen_limpia, regiones = self.clean_manga.limpiar_manga(imagen_actual)

                metrics.timings["limpieza"] = round(time.perf_counter() - t0, 4)
                metrics.detected_regions = len(regiones)
                metrics.detected_bubbles = sum(
                    1 for region in regiones if region.kind in {"dialogue", "narration", "unknown"}
                )
                metrics.detected_sfx = sum(
                    1 for region in regiones if region.kind in {"sfx", "free_text"}
                )
                metrics.retries = intento - 1

                archivo_limpieza = os.path.join(ruta_limpieza_salida, output_filename)
                with processing_stage(
                    "guardar_limpieza", logger=logger, page_index=indice_imagen, filename=output_filename
                ):
                    self._write_image(archivo_limpieza, imagen_limpia)

                return PreparedPage(
                    indice_imagen=indice_imagen,
                    source_filename=source_filename,
                    output_filename=output_filename,
                    image_path=image_path,
                    imagen_original=imagen_actual,
                    imagen_limpia=imagen_limpia,
                    mascara_capa=mascara_capa,
                    regiones=list(regiones),
                    metrics=metrics,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError, StageProcessingError) as exc:
                logger.warning(
                    "Error potencial de memoria al preparar %s (intento %s/%s): %s",
                    output_filename,
                    intento,
                    max_retries,
                    exc,
                )
                if intento >= max_retries or not self._is_retryable_memory_error(exc):
                    metrics.status = "failed"
                    metrics.error = str(exc)
                    metrics.retries = intento - 1
                    MetricsWriter(output_root).write_page(metrics)
                    raise
                imagen_actual = self.reducir_imagen(imagen_actual)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cooperative_sleep(1)

        raise RuntimeError(f"No se pudo preparar {output_filename}")

    def _finish_prepared_page(
        self,
        item: PreparedPage,
        *,
        ruta_traduccion_salida: str,
        transcripcion_queue,
        traduccion_queue,
    ) -> None:
        self.translate_manga.insertar_json_queue(
            indice_imagen=item.indice_imagen,
            transcripcion_queue=transcripcion_queue,
            traduccion_queue=traduccion_queue,
        )
        t0 = time.perf_counter()
        with processing_stage(
            "ocr_traduccion_render",
            logger=logger,
            page_index=item.indice_imagen,
            filename=item.output_filename,
        ):
            imagen_traducida = self.translate_manga.traducir_manga(
                item.imagen_original,
                item.imagen_limpia,
                item.mascara_capa,
                text_regions=item.regiones,
            )
        item.metrics.timings["ocr_traduccion_render"] = round(time.perf_counter() - t0, 4)
        item.metrics.ocr_empty = sum(
            1
            for texto in getattr(self.translate_manga, "ultimos_textos_originales", [])
            if not str(texto).strip()
        )
        item.metrics.translations_empty = sum(
            1
            for texto in getattr(self.translate_manga, "ultimos_textos_traducidos", [])
            if not str(texto).strip()
        )

        archivo_traduccion = os.path.join(ruta_traduccion_salida, item.output_filename)
        with processing_stage(
            "guardar_traduccion",
            logger=logger,
            page_index=item.indice_imagen,
            filename=item.output_filename,
        ):
            self._write_image(archivo_traduccion, imagen_traducida)

        output_root = str(Path(ruta_traduccion_salida).parent)
        MetricsWriter(output_root).write_page(item.metrics)

    def _registrar_fallo(self, ruta_traduccion_salida: str, indice_imagen: int, archivo: str, image_path: str, exc: Exception) -> None:
        output_root = str(Path(ruta_traduccion_salida).parent)
        fallidas = Path(output_root) / "fallidas"
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
            "establecer_elemento_en_lista": {
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
                    if bool(getattr(self.clean_manga, "visual_inpaint_debug", False)):
                        self.clean_manga.set_visual_inpaint_debug_context(
                            output_root=output_root,
                            page_index=indice_imagen,
                            filename=archivo,
                        )
                    else:
                        self.clean_manga.clear_visual_inpaint_debug_context()
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
            except JobControlError:
                raise
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
                cooperative_sleep(1)
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