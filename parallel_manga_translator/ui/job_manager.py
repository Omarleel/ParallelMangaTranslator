from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from parallel_manga_translator.cli import build_default_config, build_image_processor, prepare_assets, prepare_runtime
from parallel_manga_translator.config.app_config import LlmConfig, OcrConfig
from parallel_manga_translator.config.runtime_config import set_active_config
from parallel_manga_translator.infrastructure.logging_config import configure_logging, get_logger
from parallel_manga_translator.io.image_naming import normalized_page_output_name
from parallel_manga_translator.ui.manual_renderer import apply_pending_inpaint_only, parse_brush_strokes, parse_manual_regions, read_corrections, read_corrections_payload, render_manual_page, render_manual_region_preview, restore_mask_erased_pixels, write_corrections
from parallel_manga_translator.ui.queue_adapter import CapturingJsonQueue

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PROJECT_JOBS_DIR = Path(os.getenv("PMT_UI_JOBS_DIR", ".pmt_ui_jobs")).resolve()


def normalize_choice(value: Any, default: str = "auto") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"", "none", "null", "nil", "default"}:
        return default
    return normalized


@dataclass
class JobOptions:
    source_language: str = "Japonés"
    target_language: str = "Español"
    detection_engine: str = "auto"
    transcription_engine: str = "auto"
    translator: str = "llm"  # google | llm


@dataclass
class PageState:
    index: int
    source_filename: str
    output_filename: str
    status: str = "pending"  # pending | processing | ready | failed
    message: str = ""
    original_path: str = ""
    clean_path: str = ""
    translated_path: str = ""
    corrected_path: str = ""
    corrections_path: str = ""
    regions: List[Dict[str, Any]] = field(default_factory=list)
    brush_strokes: List[Dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    @property
    def display_status(self) -> str:
        if self.corrected_path and Path(self.corrected_path).exists():
            return "corrected"
        return self.status


@dataclass
class JobState:
    job_id: str
    title: str
    root_dir: str
    input_dir: str
    output_dir: str
    status: str = "queued"  # queued | processing | ready | failed
    message: str = "Esperando inicio del procesamiento."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    pages: List[PageState] = field(default_factory=list)
    processed_count: int = 0
    failed_count: int = 0
    active_page: int = 0
    options: JobOptions = field(default_factory=JobOptions)

    @property
    def total_count(self) -> int:
        return len(self.pages)

    @property
    def progress(self) -> float:
        total = self.total_count or 1
        return round(((self.processed_count + self.failed_count) / total) * 100, 2)


def normalized_output_name(filename: str, page_index: int) -> str:
    return normalized_page_output_name(filename, page_index)


def natural_sort_key(filename: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(filename).name)]


def safe_flat_name(original_name: str, used: set[str]) -> str:
    candidate = Path(original_name).name.replace("\x00", "")
    candidate = candidate or f"page_{len(used) + 1}.jpg"
    stem, ext = os.path.splitext(candidate)
    ext = ext.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError("Formato de imagen no soportado.")
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(". ") or f"page_{len(used) + 1}"
    candidate = f"{safe_stem}{ext}"
    base = candidate
    counter = 2
    while candidate.lower() in used:
        candidate = f"{safe_stem}_{counter}{ext}"
        counter += 1
    used.add(candidate.lower())
    return candidate


def safe_export_slug(value: str, fallback: str = "manga") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._-")
    return slug[:80] or fallback


def unique_arcname(folder: str, filename: str, used: set[str]) -> str:
    candidate = f"{folder}/{Path(filename).name}"
    if candidate.lower() not in used:
        used.add(candidate.lower())
        return candidate
    stem, ext = os.path.splitext(Path(filename).name)
    counter = 2
    while True:
        candidate = f"{folder}/{stem}_{counter}{ext}"
        if candidate.lower() not in used:
            used.add(candidate.lower())
            return candidate
        counter += 1


def page_to_public(page: PageState, job_id: str) -> Dict[str, Any]:
    corrected_exists = bool(page.corrected_path and Path(page.corrected_path).exists())
    return {
        "index": page.index,
        "source_filename": page.source_filename,
        "output_filename": page.output_filename,
        "status": page.status,
        "display_status": "corrected" if corrected_exists else page.status,
        "message": page.message,
        "regions": page.regions,
        "brush_strokes": page.brush_strokes,
        "has_corrected": corrected_exists,
        "images": {
            "original": f"/api/jobs/{job_id}/pages/{page.index}/image/original",
            "clean": f"/api/jobs/{job_id}/pages/{page.index}/image/clean",
            "translated": f"/api/jobs/{job_id}/pages/{page.index}/image/translated",
            "corrected": f"/api/jobs/{job_id}/pages/{page.index}/image/corrected",
            "current": f"/api/jobs/{job_id}/pages/{page.index}/image/current",
        },
        "updated_at": page.updated_at,
    }


def job_to_public(job: JobState) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "title": job.title,
        "status": job.status,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "processed_count": job.processed_count,
        "failed_count": job.failed_count,
        "total_count": job.total_count,
        "progress": job.progress,
        "active_page": job.active_page,
        "options": asdict(job.options) if hasattr(job.options, "__dataclass_fields__") else job.options,
        "pages": [page_to_public(page, job.job_id) for page in job.pages],
    }


class JobManager:
    def __init__(self, jobs_root: Path = PROJECT_JOBS_DIR, config_path: str = "config.yaml") -> None:
        self.jobs_root = jobs_root
        self.config_path = config_path
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.RLock()
        # El pipeline carga modelos grandes y usa configuración global; procesar de a un job evita pelear por GPU/config.
        self._processing_lock = threading.Lock()
        self._assets_prepared = False

    def create_job_from_paths(self, *, title: str, input_dir: Path, output_dir: Path, root_dir: Path, options: JobOptions | None = None) -> JobState:
        image_files = [p.name for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        image_files = sorted(image_files, key=natural_sort_key)
        if not image_files:
            raise ValueError("No se encontraron imágenes válidas (.jpg, .jpeg, .png, .bmp o .webp).")

        pages = []
        for idx, filename in enumerate(image_files):
            output_name = normalized_output_name(filename, idx)
            pages.append(
                PageState(
                    index=idx,
                    source_filename=filename,
                    output_filename=output_name,
                    original_path=str(input_dir / filename),
                    clean_path=str(output_dir / "limpieza" / output_name),
                    translated_path=str(output_dir / "traduccion" / output_name),
                    corrected_path=str(output_dir / "corregida" / output_name),
                    corrections_path=str(output_dir / "correcciones" / f"{Path(output_name).stem}.json"),
                )
            )
        job = JobState(
            job_id=root_dir.name,
            title=title,
            root_dir=str(root_dir),
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            pages=pages,
            options=options or JobOptions(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._save_manifest(job)
        return job

    def create_job_from_uploads(self, files: Sequence[Any], zip_file: Optional[Any] = None, title: str = "", options: JobOptions | None = None) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        root_dir = self.jobs_root / job_id
        input_dir = root_dir / "entrada"
        output_dir = root_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()

        if zip_file is not None and getattr(zip_file, "filename", ""):
            archive_path = root_dir / "upload.zip"
            self._copy_upload_file(zip_file, archive_path)
            self._extract_zip_images(archive_path, input_dir, used)
            title = title or Path(zip_file.filename).stem or "manga"

        for file in files or []:
            filename = getattr(file, "filename", "") or ""
            if not filename or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            target_name = safe_flat_name(filename, used)
            self._copy_upload_file(file, input_dir / target_name)
            title = title or Path(filename).parent.name or "manga"

        return self.create_job_from_paths(
            title=title.strip() or "Manga sin título",
            input_dir=input_dir,
            output_dir=output_dir,
            root_dir=root_dir,
            options=options or JobOptions(),
        )

    @staticmethod
    def _copy_upload_file(upload: Any, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as out:
            upload.file.seek(0)
            shutil.copyfileobj(upload.file, out)

    def _extract_zip_images(self, archive_path: Path, input_dir: Path, used: set[str]) -> None:
        max_uncompressed = 1024 * 1024 * 1024  # 1 GB límite razonable para evitar zips accidentales enormes.
        total = 0
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                ext = Path(info.filename).suffix.lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                total += info.file_size
                if total > max_uncompressed:
                    raise ValueError("El zip supera el límite de 1 GB de imágenes descomprimidas.")
                target_name = safe_flat_name(info.filename, used)
                with archive.open(info) as src, (input_dir / target_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    def start_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        thread = threading.Thread(target=self._run_job, args=(job.job_id,), daemon=True)
        thread.start()

    def get_job(self, job_id: str) -> JobState:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self._load_manifest(job_id)
                self._jobs[job_id] = job
            return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            loaded = list(self._jobs.values())
        known = {job.job_id for job in loaded}
        for manifest in self.jobs_root.glob("*/manifest.json"):
            job_id = manifest.parent.name
            if job_id not in known:
                try:
                    loaded.append(self._load_manifest(job_id))
                except Exception:
                    continue
        return [job_to_public(job) for job in sorted(loaded, key=lambda item: item.created_at, reverse=True)]

    def image_path(self, job_id: str, page_index: int, variant: str) -> Path:
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        if variant == "original":
            return Path(page.original_path)
        if variant == "clean":
            return Path(page.clean_path)
        if variant == "translated":
            return Path(page.translated_path)
        if variant == "corrected":
            return Path(page.corrected_path)
        if variant == "current":
            corrected = Path(page.corrected_path)
            return corrected if corrected.exists() else Path(page.translated_path)
        raise ValueError("Variante de imagen no soportada.")

    def create_export_zip(self, job_id: str) -> Path:
        job = self.get_job(job_id)
        export_dir = Path(job.root_dir) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_name = f"{safe_export_slug(job.title, job.job_id)}_resultado.zip"
        export_path = export_dir / export_name
        if export_path.exists():
            export_path.unlink()

        used_names: set[str] = set()
        exported_pages: List[Dict[str, Any]] = []
        with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for page in sorted(job.pages, key=lambda item: item.index):
                corrected = Path(page.corrected_path)
                translated = Path(page.translated_path)
                if corrected.exists():
                    image_path = corrected
                    variant = "corregida"
                elif page.status == "ready" and translated.exists():
                    image_path = translated
                    variant = "traduccion"
                else:
                    continue

                image_arcname = unique_arcname("imagenes_finales", page.output_filename, used_names)
                archive.write(image_path, image_arcname)

                correction_path = Path(page.corrections_path)
                correction_arcname = ""
                if correction_path.exists():
                    correction_arcname = unique_arcname("correcciones", correction_path.name, used_names)
                    archive.write(correction_path, correction_arcname)

                exported_pages.append(
                    {
                        "page_index": page.index,
                        "source_filename": page.source_filename,
                        "output_filename": page.output_filename,
                        "variant": variant,
                        "image": image_arcname,
                        "corrections": correction_arcname,
                    }
                )

            if not exported_pages:
                raise ValueError("Todavía no hay páginas listas para exportar.")

            manifest = {
                "title": job.title,
                "job_id": job.job_id,
                "exported_at": time.time(),
                "included_pages": len(exported_pages),
                "total_pages": len(job.pages),
                "note": "Cada imagen final usa la corrección manual si existe; si no, usa la traducción automática lista.",
                "pages": exported_pages,
            }
            archive.writestr("manifest_export.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        return export_path

    def _inpaint_backup_path(self, job: JobState, page: PageState) -> Path:
        backup_dir = Path(job.output_dir) / ".ui_backups" / "before_inpaint"
        return backup_dir / page.output_filename

    @staticmethod
    def _without_mask_erasers(brush_strokes):
        return [
            stroke
            for stroke in brush_strokes
            if (stroke.mode or "").strip().lower() not in {"mask_eraser", "erase_mask", "eraser"}
        ]

    def save_manual_render(self, job_id: str, page_index: int, regions_payload: Sequence[Dict[str, Any]], brush_strokes_payload: Sequence[Dict[str, Any]] | None = None, operation: str = "render") -> Dict[str, Any]:
        job = self.get_job(job_id)
        page = self._page(job, page_index)
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
        current_base = Path(page.corrected_path) if Path(page.corrected_path).exists() else translated_path
        if normalized_operation == "inpaint":
            # El inpaint manual debe actuar sobre la imagen actual y no sobre el render
            # completo; así no se vuelven a calcular fuentes ni se redibujan regiones.
            backup_path = self._inpaint_backup_path(job, page)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_base, backup_path)
            apply_pending_inpaint_only(
                base_path=current_base,
                output_path=page.corrected_path,
                brush_strokes=brush_strokes,
            )
            for stroke in brush_strokes:
                if (stroke.mode or "").strip().lower() == "inpaint":
                    stroke.applied = True
            # Los borradores solo sirven para restar la máscara antes de aplicar; no
            # deben quedar como trazos permanentes en la página ni en la UI.
            brush_strokes = self._without_mask_erasers(brush_strokes)
        elif normalized_operation in {"mask_eraser", "erase_mask", "eraser"}:
            # Si el usuario borra sobre una zona ya inpainted, restaura desde el
            # manga original. Esta acción funciona como un pincel de "quitar inpaint",
            # no como un nuevo render de texto, así que no redibuja regiones.
            restore_source = original_path
            restore_mask_erased_pixels(
                base_path=current_base,
                restore_path=restore_source,
                output_path=page.corrected_path,
                brush_strokes=brush_strokes,
            )
            brush_strokes = self._without_mask_erasers(brush_strokes)
        else:
            # Las ediciones ligeras deben conservar cualquier inpaint ya aplicado.
            # Por eso, si existe imagen corregida, se usa como base en vez de volver
            # a reconstruir toda la página desde la traducción automática.
            render_manual_page(
                clean_path=clean_path,
                original_path=original_path,
                translated_path=translated_path,
                output_path=page.corrected_path,
                regions=regions,
                brush_strokes=brush_strokes,
                base_path=current_base if current_base.exists() else translated_path,
            )
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
                    "ui_layout": region.ui_layout or previous_by_index.get(region.index, {}).get("ui_layout"),
                }
                for region in regions
            ]
            page.brush_strokes = [
                {"points": [list(point) for point in stroke.points], "radius": stroke.radius, "mode": stroke.mode, "applied": stroke.applied}
                for stroke in brush_strokes
            ]
            page.updated_at = time.time()
            job.updated_at = page.updated_at
            self._save_manifest(job)
        return page_to_public(page, job_id)


    def render_region_preview(self, job_id: str, page_index: int, region_payload: Dict[str, Any]) -> bytes:
        """Rasteriza una región con la misma ruta usada por el guardado final."""
        job = self.get_job(job_id)
        page = self._page(job, page_index)
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
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        for raw in [page.corrected_path, page.corrections_path]:
            path = Path(raw)
            if path.exists():
                path.unlink()
        # Restaurar regiones desde JSON del pipeline si existe.
        page.regions = self._merge_page_regions(job, page)
        page.brush_strokes = []
        page.updated_at = time.time()
        job.updated_at = page.updated_at
        self._save_manifest(job)
        return page_to_public(page, job_id)

    def _build_config_for_job(self, job: JobState):
        config = build_default_config(self.config_path)
        options = job.options if isinstance(job.options, JobOptions) else JobOptions(**dict(job.options or {}))
        translator = normalize_choice(options.translator, "llm")
        method = "LLM" if translator == "llm" else "Tradicional"
        traditional_provider = "google" if translator == "google" else config.translation.traditional_provider
        llm = replace(config.translation.llm, provider=config.translation.llm.provider or "groq")
        translation = replace(
            config.translation,
            idioma_entrada=options.source_language or config.translation.idioma_entrada,
            idioma_salida=options.target_language or config.translation.idioma_salida,
            metodo_traduccion=method,
            traditional_provider=traditional_provider,
            llm=llm,
            project_dir=job.input_dir,
        )
        ocr = replace(
            config.ocr,
            detection_engine=normalize_choice(options.detection_engine, "auto"),
            transcription_engine=normalize_choice(options.transcription_engine, "auto"),
        )
        return replace(config, translation=translation, ocr=ocr)

    def transcribe_manual_region(self, job_id: str, page_index: int, bbox: Sequence[float], translate: bool = True) -> Dict[str, Any]:
        job = self.get_job(job_id)
        page = self._page(job, page_index)
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

        config = self._build_config_for_job(job)
        set_active_config(config)
        from parallel_manga_translator.ocr.ocr_manager import OcrManager
        from parallel_manga_translator.translation.translator_manager import TranslatorManager

        ocr = OcrManager(config.translation.idioma_entrada, config.ocr)
        original_text = ocr.extract_texts([crop])[0] if crop.size else ""
        translated_text = ""
        if translate and original_text.strip():
            translator = TranslatorManager.from_config(config.translation, config.character_memory)
            translated_text = translator.traducir_textos([original_text])[0]
        return {
            "bbox": [x, y, bw, bh],
            "original_text": original_text,
            "translated_text": translated_text,
            "source_language": config.translation.idioma_entrada,
            "target_language": config.translation.idioma_salida,
            "ocr_engine": normalize_choice(job.options.transcription_engine, "auto"),
            "translator": normalize_choice(job.options.translator, "llm"),
        }

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        with self._processing_lock:
            self._mark_job(job, status="processing", message="Preparando modelos y recursos…")
            try:
                prepare_runtime()
                if not self._assets_prepared:
                    prepare_assets()
                    self._assets_prepared = True

                config = self._build_config_for_job(job)
                processing = replace(
                    config.processing,
                    ruta_carpeta_entrada=job.input_dir,
                    batch_size=1,
                    usar_paralelismo=False,
                    max_workers=1,
                )
                translation = replace(config.translation, project_dir=job.input_dir)
                config = replace(config, processing=processing, translation=translation)
                set_active_config(config)
                configure_logging(log_file=config.logging.file, level=config.logging.level)

                output_dir = Path(job.output_dir)
                clean_dir = output_dir / "limpieza"
                translation_dir = output_dir / "traduccion"
                corrected_dir = output_dir / "corregida"
                corrections_dir = output_dir / "correcciones"
                for directory in [clean_dir, translation_dir, corrected_dir, corrections_dir]:
                    directory.mkdir(parents=True, exist_ok=True)

                trans_queue = CapturingJsonQueue(clean_dir / "Transcripción.json")
                trad_queue = CapturingJsonQueue(translation_dir / "Traducción.json")
                trans_queue.put({"agregar_entrada": {"Título": job.title, "Páginas": len(job.pages)}})
                trad_queue.put({"agregar_entrada": {"Título": job.title, "Páginas": len(job.pages)}})

                processor = build_image_processor(config)
                for page in job.pages:
                    with self._lock:
                        page.status = "processing"
                        page.message = "Procesando OCR, limpieza, traducción y renderizado…"
                        page.updated_at = time.time()
                        job.active_page = page.index
                        job.updated_at = page.updated_at
                        self._save_manifest(job)
                    try:
                        processor.procesar(
                            job.input_dir,
                            str(clean_dir),
                            str(translation_dir),
                            {page.index: page.source_filename},
                            trans_queue,
                            trad_queue,
                        )
                        trans_queue.put({"ordenar_por_paginas": {"tipo": "Transcripción"}})
                        trad_queue.put({"ordenar_por_paginas": {"tipo": "Traducción"}})
                        if not Path(page.translated_path).exists() or not Path(page.clean_path).exists():
                            raise RuntimeError("El pipeline terminó sin generar las imágenes esperadas.")
                        with self._lock:
                            page.regions = self._merge_page_regions(job, page, trans_queue.data, trad_queue.data)
                            saved_payload = read_corrections_payload(page.corrections_path)
                            corrections = saved_payload.get("regions", [])
                            if corrections:
                                page.regions = self._apply_saved_corrections(page.regions, corrections)
                            page.brush_strokes = saved_payload.get("brush_strokes", [])
                            page.status = "ready"
                            page.message = "Lista para revisión."
                            page.updated_at = time.time()
                            job.processed_count = sum(1 for item in job.pages if item.status == "ready")
                            job.failed_count = sum(1 for item in job.pages if item.status == "failed")
                            job.updated_at = page.updated_at
                            self._save_manifest(job)
                    except Exception as exc:
                        logger.exception("Error procesando página %s del job %s: %s", page.index + 1, job_id, exc)
                        with self._lock:
                            page.status = "failed"
                            page.message = str(exc)
                            page.updated_at = time.time()
                            job.failed_count = sum(1 for item in job.pages if item.status == "failed")
                            job.updated_at = page.updated_at
                            self._save_manifest(job)

                status = "ready" if job.failed_count == 0 else "failed"
                message = "Procesamiento finalizado." if status == "ready" else "Finalizado con páginas fallidas."
                self._mark_job(job, status=status, message=message)
            except Exception as exc:
                logger.exception("Error preparando job %s: %s", job_id, exc)
                self._mark_job(job, status="failed", message=f"Error general: {exc}")
                failure_path = Path(job.root_dir) / "error.log"
                failure_path.write_text(traceback.format_exc(), encoding="utf-8")

    def _merge_page_regions(
        self,
        job: JobState,
        page: PageState,
        trans_data: Optional[Dict[str, Any]] = None,
        trad_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        trans_data = trans_data or self._read_json(Path(job.output_dir) / "limpieza" / "Transcripción.json")
        trad_data = trad_data or self._read_json(Path(job.output_dir) / "traduccion" / "Traducción.json")
        page_no = page.index + 1
        originals = self._page_items(trans_data, "Transcripción", page_no)
        translations = self._page_items(trad_data, "Traducción", page_no)
        by_index: Dict[int, Dict[str, Any]] = {}

        for item in originals:
            idx = int(item.get("Índice", len(by_index)))
            coords = item.get("Coordenadas") or [[0, 0], [0, 0]]
            bbox = self._coords_to_bbox(coords)
            by_index.setdefault(idx, {"index": idx})
            by_index[idx].update(
                {
                    "bbox": bbox,
                    "source_bbox": bbox,
                    "original_text": item.get("Texto", ""),
                    "style": item.get("Estilo", "dialogo"),
                    "type": item.get("Tipo", "dialogue"),
                    "confidence": item.get("Confianza", 0),
                    "restore_original": False,
                    "visible": True,
                    "modified": False,
                    "deleted": False,
                    "auto_font_size": True,
                    "font_size": None,
                    "ui_layout": item.get("Layout UI") or item.get("ui_layout"),
                }
            )
        for item in translations:
            idx = int(item.get("Índice", len(by_index)))
            coords = item.get("Coordenadas") or [[0, 0], [0, 0]]
            bbox = self._coords_to_bbox(coords)
            by_index.setdefault(idx, {"index": idx})
            by_index[idx].update(
                {
                    "bbox": by_index[idx].get("bbox") or bbox,
                    "source_bbox": by_index[idx].get("source_bbox") or bbox,
                    "translated_text": item.get("Texto", ""),
                    "style": item.get("Estilo", by_index[idx].get("style", "dialogo")),
                    "type": item.get("Tipo", by_index[idx].get("type", "dialogue")),
                    "confidence": item.get("Confianza", by_index[idx].get("confidence", 0)),
                    "restore_original": by_index[idx].get("restore_original", False),
                    "visible": by_index[idx].get("visible", True),
                    "modified": by_index[idx].get("modified", False),
                    "deleted": by_index[idx].get("deleted", False),
                    "auto_font_size": by_index[idx].get("auto_font_size", True),
                    "font_size": by_index[idx].get("font_size"),
                    "ui_layout": item.get("Layout UI") or item.get("ui_layout") or by_index[idx].get("ui_layout"),
                }
            )
        return [by_index[idx] for idx in sorted(by_index)]

    @staticmethod
    def _apply_saved_corrections(regions: List[Dict[str, Any]], corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        corrected_by_index = {int(item.get("index", idx)): item for idx, item in enumerate(corrections) if isinstance(item, dict)}
        merged: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for idx, region in enumerate(regions):
            region_index = int(region.get("index", idx))
            correction = corrected_by_index.get(region_index)
            if correction:
                source_bbox = correction.get("source_bbox") or region.get("source_bbox") or region.get("bbox")
                region = {
                    **region,
                    "bbox": correction.get("bbox", region.get("bbox")),
                    "source_bbox": source_bbox,
                    "translated_text": correction.get("text", region.get("translated_text", "")),
                    "style": correction.get("style", region.get("style", "dialogo")),
                    "restore_original": bool(correction.get("restore_original", False)),
                    "visible": bool(correction.get("visible", True)),
                    "modified": bool(correction.get("modified", True)),
                    "manual": bool(correction.get("manual", region.get("manual", False))),
                    "deleted": bool(correction.get("deleted", False)),
                    "auto_font_size": bool(correction.get("auto_font_size", True)),
                    "font_size": correction.get("font_size"),
                    "ui_layout": correction.get("ui_layout") or region.get("ui_layout"),
                }
            else:
                region = {
                    **region,
                    "source_bbox": region.get("source_bbox") or region.get("bbox"),
                    "modified": bool(region.get("modified", False)),
                    "auto_font_size": region.get("auto_font_size", True),
                    "font_size": region.get("font_size"),
                    "ui_layout": region.get("ui_layout"),
                }
            seen.add(region_index)
            merged.append(region)

        for correction_index, correction in sorted(corrected_by_index.items()):
            if correction_index in seen:
                continue
            bbox = correction.get("bbox") or [0, 0, 1, 1]
            merged.append({
                "index": correction_index,
                "bbox": bbox,
                "source_bbox": correction.get("source_bbox") or bbox,
                "original_text": correction.get("original_text", ""),
                "translated_text": correction.get("text", correction.get("translated_text", "")),
                "style": correction.get("style", "dialogo"),
                "type": correction.get("type", "manual"),
                "confidence": correction.get("confidence", 0),
                "restore_original": bool(correction.get("restore_original", False)),
                "visible": bool(correction.get("visible", True)),
                "modified": bool(correction.get("modified", True)),
                "manual": True,
                "deleted": bool(correction.get("deleted", False)),
                "auto_font_size": bool(correction.get("auto_font_size", True)),
                "font_size": correction.get("font_size"),
                "ui_layout": correction.get("ui_layout"),
            })
        return merged

    @staticmethod
    def _page_items(data: Dict[str, Any], key: str, page_no: int) -> List[Dict[str, Any]]:
        pages = data.get(key, []) if isinstance(data, dict) else []
        if not isinstance(pages, list):
            return []
        page = next((item for item in pages if isinstance(item, dict) and item.get("Página") == page_no), None)
        items = page.get("Globos de texto", []) if isinstance(page, dict) else []
        return items if isinstance(items, list) else []

    @staticmethod
    def _coords_to_bbox(coords: Any) -> List[int]:
        try:
            (x1, y1), (x2, y2) = coords
            return [int(x1), int(y1), max(1, int(x2) - int(x1)), max(1, int(y2) - int(y1))]
        except Exception:
            return [0, 0, 1, 1]

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _page(self, job: JobState, page_index: int) -> PageState:
        if page_index < 0 or page_index >= len(job.pages):
            raise IndexError("La página solicitada no existe.")
        return job.pages[page_index]

    def _mark_job(self, job: JobState, *, status: str, message: str) -> None:
        with self._lock:
            job.status = status
            job.message = message
            job.updated_at = time.time()
            self._save_manifest(job)

    def _save_manifest(self, job: JobState) -> None:
        manifest = Path(job.root_dir) / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(job)
        tmp_path = manifest.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(manifest)

    def _load_manifest(self, job_id: str) -> JobState:
        manifest = self.jobs_root / job_id / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError("No existe ese trabajo de UI.")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        pages = [PageState(**page) for page in data.get("pages", [])]
        data["pages"] = pages
        options = data.get("options", {})
        if isinstance(options, dict):
            data["options"] = JobOptions(**{k: v for k, v in options.items() if k in JobOptions.__dataclass_fields__})
        elif not isinstance(options, JobOptions):
            data["options"] = JobOptions()
        return JobState(**data)
