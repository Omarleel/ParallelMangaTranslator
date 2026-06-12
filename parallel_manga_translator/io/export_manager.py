from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from parallel_manga_translator.config.runtime_config import get_active_config
from parallel_manga_translator.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


class ExportManager:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".webp")

    @staticmethod
    def natural_sort_key(filename: str):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(filename).name)]

    @classmethod
    def list_images(cls, folder: str) -> List[str]:
        try:
            files = [f for f in os.listdir(folder) if f.lower().endswith(cls.IMAGE_EXTENSIONS)]
        except FileNotFoundError:
            return []
        return sorted(files, key=cls.natural_sort_key)

    @classmethod
    def export_pdf(cls, translated_dir: str, title: str) -> None:
        export_config = get_active_config().export
        if export_config.skip_pdf or not export_config.pdf:
            logger.info("Exportación PDF omitida por configuración.")
            return
        files = cls.list_images(translated_dir)
        if not files:
            logger.warning("No hay imágenes traducidas para generar PDF.")
            return
        images = []
        try:
            for name in files:
                images.append(Image.open(os.path.join(translated_dir, name)).convert("RGB"))
            pdf_path = os.path.join(translated_dir, f"{title}_Traducido.pdf")
            images[0].save(pdf_path, save_all=True, append_images=images[1:])
            logger.info("PDF generado: %s", pdf_path)
        except Exception as exc:
            logger.error("Error al compilar PDF: %s", exc)
        finally:
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass

    @classmethod
    def export_cbz(cls, translated_dir: str, title: str) -> None:
        if not get_active_config().export.cbz:
            return
        files = cls.list_images(translated_dir)
        if not files:
            logger.warning("No hay imágenes traducidas para generar CBZ.")
            return
        cbz_path = os.path.join(translated_dir, f"{title}_Traducido.cbz")
        try:
            with zipfile.ZipFile(cbz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name in files:
                    zf.write(os.path.join(translated_dir, name), arcname=name)
            logger.info("CBZ generado: %s", cbz_path)
        except Exception as exc:
            logger.error("Error al compilar CBZ: %s", exc)
