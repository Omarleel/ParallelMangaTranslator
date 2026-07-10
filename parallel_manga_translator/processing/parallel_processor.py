from __future__ import annotations

import os
import re
import time
import torch
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import List
from pathlib import Path

from parallel_manga_translator.io.json_generator import JsonWriter
from parallel_manga_translator.io.utilities import Utilities
from parallel_manga_translator.io.export_manager import ExportManager
from parallel_manga_translator.quality.metrics_manager import MetricsWriter
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.config.constants import PESO_MODELOS
from parallel_manga_translator.config.runtime_config import get_active_config

logger = get_logger(__name__)

@dataclass(frozen=True)
class ResourceProfile:
    max_parallel_workers: int
    total_memory_gb: float

@dataclass(frozen=True)
class ExecutionPlan:
    batch_size: int
    num_processes: int
    parallel_enabled: bool

class ParallelProcessor:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".webp")

    def __init__(self):
        self.utilities = Utilities()
        self._ensure_mp_start_method()
        self.resource_profile = self._detect_resources()

    @staticmethod
    def _ensure_mp_start_method() -> None:
        current_method = mp.get_start_method(allow_none=True)
        if current_method != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

    @staticmethod
    def _detect_resources() -> ResourceProfile:
        cpu_count = os.cpu_count() or 2
        max_parallel_workers = max(1, cpu_count - 1)
        total_memory_gb = 8.0
        dispositivo_detectado = "CPU"

        if torch.cuda.is_available():
            try:
                properties = torch.cuda.get_device_properties(torch.device("cuda"))
                total_memory_gb = float(properties.total_memory / 1024 ** 3)
                dispositivo_detectado = f"GPU ({properties.name})"
                # No conviene crear un proceso por SM de la GPU: cada proceso vuelve a cargar OCR/inpainting
                # y normalmente termina siendo más lento o causa OOM. Por defecto, 1 worker GPU.
                max_parallel_workers = 2 if total_memory_gb >= 20 else 1
            except Exception as exc:
                logger.warning("No se pudo leer la info de la GPU. Error: %s", exc)

        configured_workers = get_active_config().processing.max_workers
        if configured_workers is not None:
            max_parallel_workers = max(1, int(configured_workers))

        logger.info("Recursos inicializados | Dispositivo: %s | Workers máximos: %s | Memoria: %.2f GB", 
                    dispositivo_detectado, max_parallel_workers, total_memory_gb)

        return ResourceProfile(max_parallel_workers=max_parallel_workers, total_memory_gb=total_memory_gb)

    def _natural_sort_key(self, filename: str):
        nombre = Path(filename).name
        return [
            int(parte) if parte.isdigit() else parte.lower()
            for parte in re.split(r"(\d+)", nombre)
        ]

    def _list_images(self, input_dir: str) -> List[str]:
        archivos = [
            archivo
            for archivo in os.listdir(input_dir)
            if archivo.lower().endswith(self.IMAGE_EXTENSIONS)
        ]
        return sorted(archivos, key=self._natural_sort_key)

    def _build_execution_plan(self, total_images: int, requested_batch_size: int, parallel: bool) -> ExecutionPlan:
        batch_size = max(1, int(requested_batch_size))
        lotes_imagenes = [list(range(i, min(i + batch_size, total_images))) for i in range(0, total_images, batch_size)]
        num_processes = len(lotes_imagenes)

        while num_processes > self.resource_profile.max_parallel_workers or num_processes * PESO_MODELOS > self.resource_profile.total_memory_gb:
            if batch_size >= total_images:
                break
            batch_size += 1
            lotes_imagenes = [list(range(i, min(i + batch_size, total_images))) for i in range(0, total_images, batch_size)]
            num_processes = len(lotes_imagenes)

        return ExecutionPlan(
            batch_size=batch_size,
            num_processes=max(1, num_processes),
            parallel_enabled=parallel and num_processes > 1,
        )


    @staticmethod
    def _ocr_uses_external_gpu_process(active_config) -> bool:
        """Detecta PaddleOCR GPU aislado, que no comparte el bloqueo del proceso principal."""
        if not bool(active_config.ocr.gpu):
            return False

        engines = {
            str(active_config.ocr.detection_engine or "auto").strip().lower(),
            str(active_config.ocr.transcription_engine or "auto").strip().lower(),
        }
        subprocess_aliases = {
            "paddle_subprocess",
            "paddle-worker",
            "paddle_worker",
        }
        if engines & subprocess_aliases:
            return True

        paddle_aliases = {"paddle", "paddleocr", "paddle_ocr"}
        paddle_requested = bool(engines & paddle_aliases)
        mode = str(active_config.ocr.paddle_subprocess or "auto").strip().lower()
        subprocess_enabled = mode not in {"0", "false", "no", "off", "never"}
        return paddle_requested and subprocess_enabled

    def _compilar_a_pdf(self, ruta_traduccion: str, titulo_manga: str):
        ExportManager.export_pdf(ruta_traduccion, titulo_manga)

    def _compilar_a_cbz(self, ruta_traduccion: str, titulo_manga: str):
        ExportManager.export_cbz(ruta_traduccion, titulo_manga)

    @staticmethod
    def _start_json_writers(ruta_limpieza_salida: str, ruta_traduccion_salida: str):
        transcripcion_queue = mp.Queue(maxsize=1000)
        traduccion_queue = mp.Queue(maxsize=1000)
        transcripcion_queue.put({"guardar_en_archivo": os.path.join(ruta_limpieza_salida, "Transcripción.json")})
        traduccion_queue.put({"guardar_en_archivo": os.path.join(ruta_traduccion_salida, "Traducción.json")})
        transcripcion_process = JsonWriter(transcripcion_queue)
        traduccion_process = JsonWriter(traduccion_queue)
        transcripcion_process.start()
        traduccion_process.start()
        return transcripcion_queue, traduccion_queue, transcripcion_process, traduccion_process

    @staticmethod
    def _seed_json_metadata(queue, titulo: str, paginas: int) -> None:
        queue.put({"agregar_entrada": {"Título": titulo, "Páginas": paginas}})

    @staticmethod
    def _finalize_json_writers(ruta_limpieza_salida, ruta_traduccion_salida, transcripcion_queue, traduccion_queue, transcripcion_process, traduccion_process):
        transcripcion_queue.put({"ordenar_por_paginas": {"tipo": "Transcripción"}})
        traduccion_queue.put({"ordenar_por_paginas": {"tipo": "Traducción"}})
        transcripcion_queue.put({"guardar_en_archivo": {"path": os.path.join(ruta_limpieza_salida, "Transcripción.json"), "finalizar": True}})
        traduccion_queue.put({"guardar_en_archivo": {"path": os.path.join(ruta_traduccion_salida, "Traducción.json"), "finalizar": True}})
        transcripcion_process.join()
        traduccion_process.join()
        transcripcion_queue.close()
        traduccion_queue.close()

    def procesar(self, ruta_carpeta_entrada, ruta_carpeta_salida, process_func, batch_size=8, parallel=True):
        execution_started_at = time.time()
        execution_started_perf = time.perf_counter()
        processing_started_at = None
        processing_started_perf = None
        processing_finished_at = None
        processing_finished_perf = None
        execution_mode = "desconocido"
        cantidad_archivos = 0
        try:
            lista_imagenes = self._list_images(ruta_carpeta_entrada)
            cantidad_archivos = len(lista_imagenes)
            if cantidad_archivos == 0:
                logger.warning("No se encontraron imágenes en %s", ruta_carpeta_entrada)
                return False

            plan = self._build_execution_plan(cantidad_archivos, batch_size, parallel)
            hardware_usado = "GPU" if torch.cuda.is_available() else "CPU"
            active_config = get_active_config()
            process_owner = getattr(process_func, "__self__", None)
            pipeline_callable = getattr(process_owner, "procesar_pipeline", None)
            external_gpu_ocr = self._ocr_uses_external_gpu_process(active_config)
            pipeline_enabled = bool(
                parallel
                and active_config.processing.cpu_gpu_pipeline
                and torch.cuda.is_available()
                and plan.num_processes == 1
                and callable(pipeline_callable)
                and not external_gpu_ocr
            )
            execution_mode = "pipeline_cpu_gpu" if pipeline_enabled else (
                "multiproceso" if plan.parallel_enabled else "secuencial"
            )
            logger.info(
                "Plan de ejecución | hardware=%s | imágenes=%s | batch=%s | procesos=%s | modo=%s",
                hardware_usado,
                cantidad_archivos,
                plan.batch_size,
                plan.num_processes,
                execution_mode,
            )
            if pipeline_enabled and active_config.ocr.gpu:
                logger.info(
                    "Pipeline híbrida activa con OCR GPU | planificador FIFO serializa "
                    "YOLO/OCR/inpainting; las secciones CPU permanecen solapadas"
                )
            elif (
                parallel
                and active_config.processing.cpu_gpu_pipeline
                and torch.cuda.is_available()
                and external_gpu_ocr
            ):
                logger.warning(
                    "Pipeline desactivada: PaddleOCR GPU se ejecuta en un subproceso externo "
                    "y no puede compartir el planificador CUDA del proceso principal. "
                    "Usa EasyOCR/MangaOCR o configura ocr.paddle_subprocess=false."
                )

            lotes_imagenes = [lista_imagenes[i:i + plan.batch_size] for i in range(0, cantidad_archivos, plan.batch_size)]
            lotes_imagenes = self.utilities.convertir_a_diccionarios(lotes_imagenes)

            ruta_limpieza_salida = os.path.join(ruta_carpeta_salida, "limpieza")
            ruta_traduccion_salida = os.path.join(ruta_carpeta_salida, "traduccion")
            os.makedirs(ruta_limpieza_salida, exist_ok=True)
            os.makedirs(ruta_traduccion_salida, exist_ok=True)

            transcripcion_queue, traduccion_queue, transcripcion_process, traduccion_process = self._start_json_writers(
                ruta_limpieza_salida, ruta_traduccion_salida)

            try:
                self._seed_json_metadata(transcripcion_queue, os.path.basename(ruta_carpeta_entrada), cantidad_archivos)
                self._seed_json_metadata(traduccion_queue, os.path.basename(ruta_carpeta_entrada), cantidad_archivos)
                processing_started_at = time.time()
                processing_started_perf = time.perf_counter()

                if pipeline_enabled:
                    paginas_ordenadas = {
                        indice: archivo
                        for lote in lotes_imagenes
                        for indice, archivo in lote.items()
                    }
                    logger.info(
                        "Pipeline CPU/GPU activa | prefetch=%s | una copia de modelos CUDA | "
                        "gpu_scheduler=%s",
                        active_config.processing.pipeline_prefetch,
                        "fifo" if active_config.ocr.gpu else "no_requerido",
                    )
                    pipeline_callable(
                        ruta_carpeta_entrada,
                        ruta_limpieza_salida,
                        ruta_traduccion_salida,
                        paginas_ordenadas,
                        transcripcion_queue,
                        traduccion_queue,
                        prefetch=active_config.processing.pipeline_prefetch,
                    )
                elif plan.parallel_enabled:
                    processes = []
                    for lote in lotes_imagenes:
                        p = mp.Process(target=process_func, args=(ruta_carpeta_entrada, ruta_limpieza_salida, 
                                       ruta_traduccion_salida, lote, transcripcion_queue, traduccion_queue))
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()
                else:
                    for lote in lotes_imagenes:
                        process_func(ruta_carpeta_entrada, ruta_limpieza_salida, 
                                     ruta_traduccion_salida, lote, transcripcion_queue, traduccion_queue)
                processing_finished_at = time.time()
                processing_finished_perf = time.perf_counter()
            finally:
                if processing_started_at is not None and processing_finished_at is None:
                    processing_finished_at = time.time()
                    processing_finished_perf = time.perf_counter()
                self._finalize_json_writers(ruta_limpieza_salida, ruta_traduccion_salida, transcripcion_queue, 
                                            traduccion_queue, transcripcion_process, traduccion_process)
                
                titulo_manga = os.path.basename(ruta_carpeta_entrada)
                self._compilar_a_pdf(ruta_traduccion_salida, titulo_manga)
                self._compilar_a_cbz(ruta_traduccion_salida, titulo_manga)
                execution_finished_at = time.time()
                execution_finished_perf = time.perf_counter()
                execution_duration_seconds = execution_finished_perf - execution_started_perf
                processing_duration_seconds = (
                    processing_finished_perf - processing_started_perf
                    if processing_started_perf is not None and processing_finished_perf is not None
                    else 0.0
                )
                reporte_metricas = MetricsWriter.aggregate(
                    ruta_carpeta_salida,
                    execution_started_at=execution_started_at,
                    execution_finished_at=execution_finished_at,
                    processing_started_at=processing_started_at,
                    processing_finished_at=processing_finished_at,
                    execution_duration_seconds=execution_duration_seconds,
                    processing_duration_seconds=processing_duration_seconds,
                    total_input_pages=cantidad_archivos,
                    execution_mode=execution_mode,
                )
                if reporte_metricas:
                    logger.info(
                        "Reporte de métricas generado: %s | tiempo_real=%.2fs | procesamiento=%.2fs",
                        reporte_metricas,
                        execution_duration_seconds,
                        processing_duration_seconds,
                    )

            return True
        except Exception as exc:
            logger.exception("Error al procesar: %s", exc)
            return False