from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import torch
import torch.multiprocessing as mp

from Applications.JsonGenerator import JsonWriter
from Applications.Utilities import Utilities
from .LoggingConfig import get_logger
from Utils.Constantes import PESO_MODELOS

from pathlib import Path
import re

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

        if torch.cuda.is_available():
            try:
                properties = torch.cuda.get_device_properties(torch.device("cuda"))
                max_parallel_workers = max(1, int(properties.multi_processor_count))
                total_memory_gb = float(properties.total_memory / 1024 ** 3)
            except Exception as exc:
                logger.warning("No se pudo leer la info de la GPU. Se usarán valores CPU. Error: %s", exc)

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
        queue.put({
            "agregar_entrada": {
                "Título": titulo,
                "Páginas": paginas,
            }
        })

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
        try:
            lista_imagenes = self._list_images(ruta_carpeta_entrada)
            cantidad_archivos = len(lista_imagenes)
            if cantidad_archivos == 0:
                logger.warning("No se encontraron imágenes en %s", ruta_carpeta_entrada)
                return False

            plan = self._build_execution_plan(cantidad_archivos, batch_size, parallel)
            logger.info(
                "Plan de ejecución | imágenes=%s | batch=%s | procesos=%s | paralelo=%s",
                cantidad_archivos,
                plan.batch_size,
                plan.num_processes,
                plan.parallel_enabled,
            )

            lotes_imagenes = [lista_imagenes[i:i + plan.batch_size] for i in range(0, cantidad_archivos, plan.batch_size)]
            lotes_imagenes = self.utilities.convertir_a_diccionarios(lotes_imagenes)

            ruta_limpieza_salida = os.path.join(ruta_carpeta_salida, "Limpieza")
            ruta_traduccion_salida = os.path.join(ruta_carpeta_salida, "Traduccion")
            os.makedirs(ruta_limpieza_salida, exist_ok=True)
            os.makedirs(ruta_traduccion_salida, exist_ok=True)

            transcripcion_queue, traduccion_queue, transcripcion_process, traduccion_process = self._start_json_writers(
                ruta_limpieza_salida,
                ruta_traduccion_salida,
            )

            try:
                self._seed_json_metadata(transcripcion_queue, os.path.basename(ruta_carpeta_entrada), cantidad_archivos)
                self._seed_json_metadata(traduccion_queue, os.path.basename(ruta_carpeta_entrada), cantidad_archivos)

                if plan.parallel_enabled:
                    processes = []
                    for lote in lotes_imagenes:
                        process = mp.Process(
                            target=process_func,
                            args=(
                                ruta_carpeta_entrada,
                                ruta_limpieza_salida,
                                ruta_traduccion_salida,
                                lote,
                                transcripcion_queue,
                                traduccion_queue,
                            ),
                        )
                        process.start()
                        processes.append(process)

                    for process in processes:
                        process.join()

                    alive_processes = [process for process in processes if process.is_alive()]
                    for process in alive_processes:
                        process.terminate()
                else:
                    for lote in lotes_imagenes:
                        process_func(
                            ruta_carpeta_entrada,
                            ruta_limpieza_salida,
                            ruta_traduccion_salida,
                            lote,
                            transcripcion_queue,
                            traduccion_queue,
                        )
            finally:
                self._finalize_json_writers(
                    ruta_limpieza_salida,
                    ruta_traduccion_salida,
                    transcripcion_queue,
                    traduccion_queue,
                    transcripcion_process,
                    traduccion_process,
                )

            return True
        except Exception as exc:
            logger.exception("Error al procesar imágenes en paralelo: %s", exc)
            return False