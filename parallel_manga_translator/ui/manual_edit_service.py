"""Edicion manual de una pagina: render, pinceles, inpaint por trazo y OCR puntual.

Eran 16 metodos de `JobManager`, casi 400 lineas, la mayor de sus responsabilidades. La
caché de inpainters manuales (`_manual_inpainters` y su lock) tambien vivia alli, aunque
sólo la usa el editor: estado de una funcion concreta guardado en el orquestador general.

Del manager sólo necesita cuatro cosas, y por eso el corte es real y no una separacion de
mentira: buscar un trabajo, la persistencia del manifiesto, el lock que serializa los
cambios de estado, y la configuracion de un trabajo. Todo lo demas era suyo.

`JobManager` sigue siendo la fachada que ve `ui/app.py`; delega aqui.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2

from parallel_manga_translator.config.constants import normalizar_modelo_inpaint
from parallel_manga_translator.inpainting import AOTInpainter, LamaInpainterMPE, LamaLarge, OpenCVInpainter
from parallel_manga_translator.infrastructure.execution_control import (
    ExecutionControl,
    execution_control_scope,
)
from parallel_manga_translator.infrastructure.gpu_scheduler import gpu_slot
from parallel_manga_translator.ui.manual_renderer import apply_background_brush_strokes, apply_pending_inpaint_only, parse_brush_strokes, parse_manual_regions, render_manual_composite, render_manual_region_preview, resolve_manual_region_metrics, write_corrections
from parallel_manga_translator.ui.job_manifest_store import JobManifestStore
from parallel_manga_translator.ui.job_state import (
    JobState,
    PageState,
    normalize_choice,
    page_of,
    page_to_public,
)
from parallel_manga_translator.ui.page_regions import (
    merge_page_regions,
)
from typing import Callable


class ManualEditService:
    """Todo lo que el editor del navegador puede hacer sobre una pagina ya procesada."""

    def __init__(
        self,
        *,
        get_job: Callable[[str], JobState],
        manifests: JobManifestStore,
        lock: threading.RLock,
        config_for_job: Callable[..., Any],
    ) -> None:
        self._get_job = get_job
        self._manifests = manifests
        # El mismo lock del manager, a proposito: estos metodos mutan `JobState` y esa
        # mutacion tiene que serializarse con la del worker, no en paralelo a ella.
        self._lock = lock
        self._config_for_job = config_for_job
        # Estado propio del editor: los inpainters manuales se cachean por modelo porque
        # cargarlos cuesta GPU y el usuario pincela muchas veces seguidas.
        self._manual_inpainters: Dict[str, Any] = {}
        self._manual_inpaint_lock = threading.Lock()

    def save_manual_render(
        self,
        job_id: str,
        page_index: int,
        regions_payload: Sequence[Dict[str, Any]],
        brush_strokes_payload: Sequence[Dict[str, Any]] | None = None,
        operation: str = "render",
        inpaint_model: str | None = None,
        background_revision: str | None = None,
    ) -> Dict[str, Any]:
        job = self._get_job(job_id)
        page = page_of(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para edición.")
        clean_path = Path(page.clean_path)
        original_path = Path(page.original_path)
        translated_path = Path(page.translated_path)
        if not clean_path.exists() or not original_path.exists() or not translated_path.exists():
            raise ValueError("Faltan imágenes base para renderizar la corrección.")
        image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo leer la imagen limpia.")
        h, w = image.shape[:2]
        regions = parse_manual_regions(regions_payload, w, h)
        brush_strokes = parse_brush_strokes(brush_strokes_payload or [], w, h)
        normalized_operation = (operation or "render").strip().lower()

        # El historial de la UI guarda esta revisión junto con bbox/texto. Al
        # deshacer/rehacer, el servidor vuelve exactamente a la capa de limpieza
        # correspondiente y después recompone las regiones, sin reutilizar texto
        # rasterizado de una versión posterior.
        active_revision, active_background = self._resolve_background_revision(
            job, page, background_revision
        )
        next_revision = active_revision
        next_background = active_background

        if normalized_operation == "inpaint":
            target_revision, target_path = self._new_background_revision_path(job, page)
            selected_manual_model = self._resolve_manual_inpaint_model(job, inpaint_model)
            changed = apply_pending_inpaint_only(
                base_path=active_background,
                output_path=target_path,
                brush_strokes=brush_strokes,
                inpaint_fn=self._manual_inpaint_callable(selected_manual_model),
            )
            if changed:
                next_revision, next_background = target_revision, target_path
            else:
                target_path.unlink(missing_ok=True)
            for stroke in brush_strokes:
                if (stroke.mode or "").strip().lower() == "inpaint":
                    stroke.applied = True
            brush_strokes = self._without_mask_erasers(brush_strokes)
        elif self._has_background_brush_strokes(brush_strokes):
            target_revision, target_path = self._new_background_revision_path(job, page)
            changed = apply_background_brush_strokes(
                base_path=active_background,
                clean_path=clean_path,
                original_path=original_path,
                output_path=target_path,
                brush_strokes=brush_strokes,
            )
            if changed:
                next_revision, next_background = target_revision, target_path
            else:
                target_path.unlink(missing_ok=True)

        # La salida final se reconstruye siempre como dos capas: fondo/limpieza y
        # regiones de texto. Nunca parte de corrected_path, que puede contener una
        # posición de texto ya rasterizada y causar fantasmas al deshacer un movimiento.
        render_manual_composite(
            background_path=next_background,
            original_path=original_path,
            output_path=page.corrected_path,
            regions=regions,
        )

        # Las pinceladas ya quedaron incorporadas en una revisión inmutable del fondo.
        # El identificador de esa revisión sí permanece en el historial del navegador.
        brush_strokes = []

        # Conservamos source_bbox como metadato de compatibilidad. Ya no se necesita
        # para borrar texto previo porque la composición nunca usa una imagen que tenga
        # texto editable rasterizado, pero avanzar el origen sigue siendo útil para
        # proyectos y pruebas creados con versiones anteriores.
        for region in regions:
            region.source_bbox = region.bbox

        write_corrections(page.corrections_path, regions, brush_strokes)
        with self._lock:
            previous_by_index = {
                int(region.get("index", idx)): region
                for idx, region in enumerate(page.regions)
                if isinstance(region, dict)
            }
            payload_by_index = {
                int(region.get("index", idx)): region
                for idx, region in enumerate(regions_payload)
                if isinstance(region, dict)
            }
            page.regions = [
                {
                    **previous_by_index.get(region.index, {}),
                    "index": region.index,
                    "bbox": list(region.bbox),
                    "source_bbox": list(region.source_bbox or region.bbox),
                    "original_text": payload_by_index.get(region.index, {}).get("original_text", previous_by_index.get(region.index, {}).get("original_text", "")),
                    "translated_text": region.text,
                    "style": region.style,
                    "type": previous_by_index.get(region.index, {}).get("type", "manual" if region.manual else "dialogue"),
                    "restore_original": region.restore_original,
                    "visible": region.visible,
                    "modified": region.modified,
                    "manual": region.manual or bool(previous_by_index.get(region.index, {}).get("manual", False)),
                    "deleted": region.deleted,
                    "auto_font_size": region.auto_font_size,
                    "font_size": region.font_size,
                    "rotation_angle": region.rotation_angle,
                    "ui_layout": region.ui_layout or previous_by_index.get(region.index, {}).get("ui_layout"),
                }
                for region in regions
            ]
            page.brush_strokes = []
            page.background_revision = next_revision
            page.manual_background_path = "" if next_revision == "base" else str(next_background)
            page.updated_at = time.time()
            job.updated_at = page.updated_at
            self._manifests.save(job)
        return page_to_public(page, job_id)

    def render_region_preview(self, job_id: str, page_index: int, region_payload: Dict[str, Any]) -> bytes:
        """Rasteriza una región con la misma ruta usada por el guardado final."""
        job = self._get_job(job_id)
        page = page_of(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para edición.")
        clean_path = Path(page.clean_path)
        original_path = Path(page.original_path)
        if not clean_path.exists() or not original_path.exists():
            raise ValueError("Faltan imágenes base para previsualizar la región.")
        image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo leer la imagen limpia.")
        h, w = image.shape[:2]
        regions = parse_manual_regions([region_payload], w, h)
        if not regions:
            raise ValueError("La región de previsualización no es válida.")
        return render_manual_region_preview(
            clean_path=clean_path,
            original_path=original_path,
            region=regions[0],
        )

    def reset_manual_render(self, job_id: str, page_index: int) -> Dict[str, Any]:
        job = self._get_job(job_id)
        with self._lock:
            page = page_of(job, page_index)
            self.discard_edits(job, page)
            page.updated_at = time.time()
            job.updated_at = page.updated_at
            self._manifests.save(job)
            return page_to_public(page, job_id)

    def resolve_region_metrics(self, job_id: str, page_index: int, region_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve la tipografía y geometría exactas de una región sin guardar cambios."""
        job = self._get_job(job_id)
        page = page_of(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para edición.")
        clean_path = Path(page.clean_path)
        if not clean_path.exists():
            raise ValueError("Falta la imagen limpia para medir la región.")
        image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo leer la imagen limpia.")
        h, w = image.shape[:2]
        regions = parse_manual_regions([region_payload], w, h)
        if not regions:
            raise ValueError("La región de medición no es válida.")
        return resolve_manual_region_metrics(clean_path=clean_path, region=regions[0])

    def transcribe_manual_region(self, job_id: str, page_index: int, bbox: Sequence[float], translate: bool = True) -> Dict[str, Any]:
        job = self._get_job(job_id)
        page = page_of(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para OCR manual.")
        image = cv2.imread(str(page.original_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo leer la imagen original para OCR.")
        h, w = image.shape[:2]
        from parallel_manga_translator.ui.manual_renderer import _safe_box  # reutiliza validación central

        x, y, bw, bh = _safe_box(bbox, w, h)
        crop = image[y:y + bh, x:x + bw]
        if crop.size == 0:
            raise ValueError("La región seleccionada no contiene píxeles válidos.")

        config = self._config_for_job(job)
        from parallel_manga_translator.ocr.ocr_manager import OcrManager

        ocr = OcrManager(config.translation.idioma_entrada, config.ocr)
        original_text = ocr.extract_texts([crop])[0] if crop.size else ""
        rotation_angle = 0.0
        rotation_confidence = 0.0
        try:
            from parallel_manga_translator.geometry.text_orientation import estimate_text_rotation
            from parallel_manga_translator.ocr.text_detection import TextDetectionFactory

            detector = TextDetectionFactory.create(config.translation.idioma_entrada, config.ocr)
            rotation = estimate_text_rotation(detector.detect_text_boxes(crop))
            rotation_angle = float(rotation.get("angle", 0.0) or 0.0)
            rotation_confidence = float(rotation.get("confidence", 0.0) or 0.0)
        except Exception:
            pass
        translated_text = ""
        if translate and original_text.strip():
            translated_text = self.translate_manual_text(job_id, page_index, original_text)["translated_text"]
        return {
            "bbox": [x, y, bw, bh],
            "original_text": original_text,
            "translated_text": translated_text,
            "rotation_angle": rotation_angle,
            "rotation_confidence": rotation_confidence,
            "source_language": config.translation.idioma_entrada,
            "target_language": config.translation.idioma_salida,
            "ocr_engine": normalize_choice(job.options.transcription_engine, "auto"),
            "translator": normalize_choice(job.options.translator, "llm"),
        }

    def translate_manual_text(self, job_id: str, page_index: int, original_text: str) -> Dict[str, Any]:
        """Traduce una transcripción corregida manualmente sin volver a ejecutar OCR."""
        job = self._get_job(job_id)
        page = page_of(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para traducir.")

        source_text = str(original_text or "")
        if not source_text.strip():
            return {
                "original_text": source_text,
                "translated_text": "",
            }

        config = self._config_for_job(job)
        from parallel_manga_translator.translation.translator_manager import TranslatorManager

        control = ExecutionControl()
        with execution_control_scope(control):
            translator = TranslatorManager.from_config(config.translation, config.character_memory)
            translated_text = translator.traducir_textos([source_text])[0]

        with self._lock:
            job.updated_at = time.time()
            self._manifests.save(job)

        return {
            "original_text": source_text,
            "translated_text": str(translated_text or ""),
            "source_language": config.translation.idioma_entrada,
            "target_language": config.translation.idioma_salida,
            "translator": normalize_choice(job.options.translator, "llm"),
        }

    def discard_edits(self, job: JobState, page: PageState) -> None:
        """Borra la corrección manual de una página y la devuelve a la salida automática."""
        for raw in [page.corrected_path, page.corrections_path]:
            if not raw:
                continue
            path = Path(raw)
            if path.exists():
                path.unlink()
        background_dir = self._manual_background_dir(job, page)
        if background_dir.exists():
            shutil.rmtree(background_dir, ignore_errors=True)
        # Restaurar regiones desde JSON del pipeline si existe.
        page.regions = merge_page_regions(job, page)
        page.brush_strokes = []
        page.background_revision = "base"
        page.manual_background_path = ""

    def _manual_inpaint_callable(self, model_name: str):
        factories = {
            "opencv-tela": OpenCVInpainter,
            "lama_mpe": LamaInpainterMPE,
            "lama_large_512px": LamaLarge,
            "aot": AOTInpainter,
        }
        factory = factories.get(model_name)
        if factory is None:
            raise ValueError(f"Modelo de inpainting manual no soportado: {model_name}")

        def run(image, mask):
            if model_name == "opencv-tela":
                return OpenCVInpainter().inpaint(image, mask)
            with self._manual_inpaint_lock:
                inpainter = self._manual_inpainters.get(model_name)
                if inpainter is None:
                    inpainter = factory()
                    self._manual_inpainters[model_name] = inpainter

                async def infer():
                    if getattr(inpainter, "model", None) is None and hasattr(inpainter, "_load"):
                        await inpainter._load()
                    with gpu_slot("ui.manual_inpaint", enabled=True):
                        if hasattr(inpainter, "_inpaint"):
                            result = inpainter._inpaint(image, mask)
                        else:
                            result = inpainter.inpaint(image, mask)
                        if asyncio.iscoroutine(result):
                            result = await result
                        return result

                return asyncio.run(infer())

        # Perfil de recorte conservador. No cambia modelo, precisión ni pesos: solo
        # evita ejecutar la red sobre partes lejanas de la página que el pincel no
        # puede modificar. LaMa Large conserva una ventana mínima de 1024x1024 y
        # 384 px de contexto alrededor de trazos mayores; sigue en fp32.
        crop_profiles = {
            "lama_large_512px": {"enabled": True, "min_side": 1024, "context_px": 384, "alignment": 64, "max_model_side": 1536},
            "lama_mpe": {"enabled": True, "min_side": 896, "context_px": 320, "alignment": 64, "max_model_side": 1024},
            "aot": {"enabled": True, "min_side": 768, "context_px": 256, "alignment": 64, "max_model_side": 1024},
        }
        run._pmt_crop_profile = crop_profiles.get(model_name, {"enabled": False})
        return run

    def _resolve_manual_inpaint_model(self, job: JobState, requested: str | None) -> str:
        raw = str(requested or "job").strip().lower()
        if raw in {"job", "configured", "config", "default"}:
            raw = str(getattr(job.options, "inpaint_model", "auto") or "auto")
        model = normalizar_modelo_inpaint(raw, "auto")
        # El pincel manual siempre trabaja con una máscara raster. B/N necesita cajas
        # de detección, así que para este flujo se usa LaMa como opción automática
        # de calidad y OpenCV solo cuando el usuario lo selecciona explícitamente.
        if model in {"auto", "B/N"}:
            return "lama_mpe"
        return model

    @staticmethod
    def _without_mask_erasers(brush_strokes):
        return [
            stroke
            for stroke in brush_strokes
            if (stroke.mode or "").strip().lower() not in {"mask_eraser", "erase_mask", "eraser"}
        ]

    def _resolve_background_revision(
        self,
        job: JobState,
        page: PageState,
        requested_revision: str | None,
    ) -> tuple[str, Path]:
        revision = str(requested_revision or page.background_revision or "base").strip() or "base"
        if revision == "base":
            clean_path = Path(page.clean_path)
            if not clean_path.exists():
                raise ValueError("Falta la imagen limpia para reconstruir el fondo.")
            return "base", clean_path

        revision_path = self._background_revision_path(job, page, revision)
        if revision_path.exists():
            return revision, revision_path

        # Compatibilidad con manifiestos creados por versiones intermedias: si la
        # revisión actual apunta a un archivo explícito, se acepta y se archiva para
        # que desde este momento también pueda participar en deshacer/rehacer.
        legacy_path = Path(page.manual_background_path) if page.manual_background_path else None
        if revision == page.background_revision and legacy_path and legacy_path.exists():
            revision_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, revision_path)
            return revision, revision_path
        raise ValueError("La revisión de fondo solicitada ya no está disponible.")

    def _background_revision_path(self, job: JobState, page: PageState, revision: str) -> Path:
        safe_revision = re.sub(r"[^A-Za-z0-9_-]+", "", str(revision or ""))
        if not safe_revision:
            raise ValueError("La revisión de fondo no es válida.")
        return self._manual_background_dir(job, page) / f"{safe_revision}.png"

    def _new_background_revision_path(self, job: JobState, page: PageState) -> tuple[str, Path]:
        revision = uuid.uuid4().hex
        path = self._background_revision_path(job, page, revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        return revision, path

    @staticmethod
    def _has_background_brush_strokes(brush_strokes) -> bool:
        return any(
            (stroke.mode or "restore_clean").strip().lower() != "inpaint"
            for stroke in brush_strokes
        )

    def _inpaint_backup_path(self, job: JobState, page: PageState) -> Path:
        backup_dir = Path(job.output_dir) / ".ui_backups" / "before_inpaint"
        return backup_dir / page.output_filename

    def _manual_background_dir(self, job: JobState, page: PageState) -> Path:
        return Path(job.output_dir) / ".ui_backgrounds" / f"page_{page.index:04d}"
