"""Artefactos de depuración del inpaint de **una** página.

Esto vivía dentro de `CleanInpaintingPipelineMixin`: diez métodos, ~180 líneas y cuatro
atributos de página —`visual_inpaint_debug_output_root`, `_page_index`, `_filename` y la
lista de registros— que dos setters del puerto empujaban al limpiador antes de cada
página y trece `getattr` leían después. Es decir: un objeto de vida larga haciendo de
cuaderno de notas, y el contrato del puerto cargando con cuatro métodos que solo existían
para eso.

Aquí es lo que es: un escritor **por página**, creado con el destino y la identidad de esa
página, que se pasa a los tres métodos de limpieza que lo necesitan. Sin destino —el caso
normal— queda apagado y todos sus métodos no hacen nada, así que quien lo usa no tiene que
preguntar si la depuración está encendida.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.io.image_io import try_write_image
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.processing.inpainter_runner import InpainterRunner

logger = get_logger(__name__)


class VisualInpaintDebugWriter:
    """Escribe recortes, informes y manifiesto del inpaint de una página.

    Apagado (sin `output_root`) es un objeto nulo: `write_crop` devuelve `None` y
    `write_region_summary` un diccionario vacío, que es justo lo que esperan los sitios
    donde se usa.
    """

    def __init__(
        self,
        output_root: str = "",
        page_index: Optional[int] = None,
        filename: str = "",
        enabled: bool = True,
    ) -> None:
        self.output_root = str(output_root or "")
        self.page_index = page_index
        self.filename = str(filename or "")
        self._enabled = bool(enabled)
        self.records: List[Dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self.output_root)

    # -- rutas -------------------------------------------------------------------------

    @staticmethod
    def _safe_token(value: object, fallback: str = "page") -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
        token = token.strip("._-")
        return token or fallback

    def page_dir(self) -> Optional[Path]:
        if not self.enabled:
            return None

        output_root = Path(self.output_root)
        page_stem = self._safe_token(Path(self.filename or "page").stem, "page")
        if isinstance(self.page_index, int):
            page_number = f"{self.page_index + 1:04d}"
            folder_name = page_number if page_stem == page_number else f"{page_number}_{page_stem}"
        else:
            folder_name = page_stem

        page_dir = output_root / "debug_inpaint" / folder_name
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir

    def relpath(self, path: Path) -> str:
        """Ruta relativa con separadores `/`.

        Estas rutas viajan dentro de JSON que consumen la UI y otras herramientas, así que
        no pueden depender del separador del sistema: en Windows saldrían con `\\`.
        """
        output_root = Path(self.output_root)
        try:
            return path.relative_to(output_root).as_posix()
        except Exception:
            return Path(path).as_posix()

    # -- serialización -----------------------------------------------------------------

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Objeto no serializable: {type(value)!r}")

    @staticmethod
    def _bbox(
        clean_mask: np.ndarray,
        safe_mask: Optional[np.ndarray],
        image_shape,
        padding: int = 12,
    ) -> Tuple[int, int, int, int]:
        bbox_mask = safe_mask if safe_mask is not None and cv2.countNonZero(safe_mask) > 0 else clean_mask
        return InpainterRunner.mask_bounding_rect(bbox_mask, image_shape, padding=padding)

    # -- escritura ---------------------------------------------------------------------

    def write_crop(
        self,
        image_or_mask: Optional[np.ndarray],
        clean_mask: np.ndarray,
        safe_mask: Optional[np.ndarray],
        region_index: Optional[int],
        label: str,
    ) -> Optional[str]:
        page_dir = self.page_dir()
        if page_dir is None or image_or_mask is None or region_index is None:
            return None

        x, y, w, h = self._bbox(clean_mask, safe_mask, image_or_mask.shape, padding=12)
        if w <= 0 or h <= 0:
            return None

        crop = image_or_mask[y:y + h, x:x + w]
        if crop.size == 0:
            return None

        safe_label = self._safe_token(label, "crop")
        path = page_dir / f"r{int(region_index):03d}_{safe_label}.png"
        try_write_image(path, crop, logger=logger)
        return self.relpath(path)

    def write_manifest(self) -> Optional[str]:
        page_dir = self.page_dir()
        if page_dir is None:
            return None

        manifest = {
            "page_index": self.page_index,
            "page_number": int(self.page_index) + 1 if isinstance(self.page_index, int) else None,
            "filename": self.filename,
            "regions": list(self.records),
        }
        manifest_path = page_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=self._json_safe), encoding="utf-8"
        )
        return self.relpath(manifest_path)

    def write_region_summary(
        self,
        *,
        region_index: int,
        region: TextRegion,
        before_image: Optional[np.ndarray],
        after_image: np.ndarray,
        clean_mask: np.ndarray,
        safe_mask: np.ndarray,
        fill_color,
        fill_strategy: str,
        method: str,
        chosen_candidate: str,
        report,
        attempts: List[dict],
    ) -> dict:
        page_dir = self.page_dir()
        if page_dir is None:
            return {}

        files = {
            "before_crop": self.write_crop(before_image, clean_mask, safe_mask, region_index, "before") if before_image is not None else None,
            "chosen_crop": self.write_crop(after_image, clean_mask, safe_mask, region_index, "chosen"),
            "clean_mask": self.write_crop(clean_mask, clean_mask, safe_mask, region_index, "clean_mask"),
            "safe_mask": self.write_crop(safe_mask, clean_mask, safe_mask, region_index, "safe_mask"),
        }
        files = {key: value for key, value in files.items() if value}

        x, y, w, h = region.bbox
        tx, ty, tw, th = region.text_bbox
        payload = {
            "region_index": int(region_index),
            "kind": str(region.kind),
            "confidence": round(float(region.confidence), 4),
            "bbox": [int(x), int(y), int(w), int(h)],
            "text_bbox": [int(tx), int(ty), int(tw), int(th)],
            "fill_strategy": str(fill_strategy),
            "fill_color_bgr": [int(c) for c in fill_color],
            "method": str(method),
            "chosen_candidate": str(chosen_candidate),
            "passed": bool(getattr(report, "passed", True)) if report is not None else None,
            "score": round(float(getattr(report, "score", 0.0)), 4) if report is not None else None,
            "failed_checks": list(getattr(report, "failed_checks", [])) if report is not None else [],
            "attempts": attempts,
            "report": report.to_dict() if report is not None and hasattr(report, "to_dict") else None,
            "files": files,
        }

        report_path = page_dir / f"r{int(region_index):03d}_report.json"
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=self._json_safe), encoding="utf-8"
        )
        payload["report_file"] = self.relpath(report_path)

        self.records.append(payload)
        manifest_path = self.write_manifest()

        return {
            "visual_inpaint_debug_dir": self.relpath(page_dir),
            "visual_inpaint_debug_manifest": manifest_path,
            "visual_inpaint_debug_report": payload["report_file"],
            "visual_inpaint_debug_files": files,
        }


#: Escritor apagado, para cuando no hay depuración pedida. Se comparte porque no guarda
#: nada: sin destino, `records` no se toca nunca.
DEPURACION_APAGADA = VisualInpaintDebugWriter(enabled=False)


__all__ = ["DEPURACION_APAGADA", "VisualInpaintDebugWriter"]
