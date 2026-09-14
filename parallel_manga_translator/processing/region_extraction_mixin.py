from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.models.region_identity import run_region_uid
from parallel_manga_translator.infrastructure.logging_config import get_logger

Box = Tuple[int, int, int, int]
logger = get_logger(__name__)


class RegionExtractionMixin:
    """Extracción y ordenamiento de áreas de interés para OCR."""

    @staticmethod
    def _prepare_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=8)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        # Binarización adaptativa + corrección de texto blanco sobre fondo oscuro.
        block_size = max(15, (min(gray.shape[:2]) // 8) * 2 + 1)
        binaria = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            9,
        )

        pixeles_blancos = cv2.countNonZero(binaria)
        pixeles_totales = binaria.size
        pixeles_negros = pixeles_totales - pixeles_blancos
        if pixeles_negros > pixeles_blancos:
            binaria = cv2.bitwise_not(binaria)

        # Borde blanco para que OCR no pierda caracteres pegados a la caja.
        border = max(6, min(18, int(round(min(binaria.shape[:2]) * 0.06))))
        binaria = cv2.copyMakeBorder(binaria, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

        h, w = binaria.shape[:2]
        min_side = min(h, w)
        max_side = max(h, w)
        scale = 1.0
        if min_side < 96:
            scale = max(scale, min(3.0, 96 / max(1, min_side)))
        if max_side < 420:
            scale = max(scale, min(2.2, 420 / max(1, max_side)))
        if scale > 1.01:
            binaria = cv2.resize(binaria, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

    def _reading_order_key(self, item):
        box = item.bbox if isinstance(item, TextRegion) else item
        return self.reading_order_resolver.key_for_page_box(box)

    def _sort_regions_for_reading(self, regiones: Sequence[TextRegion]) -> List[TextRegion]:
        try:
            if regiones and any("panel_index" in getattr(r, "metadata", {}) for r in regiones):
                return sorted(
                    list(regiones),
                    key=lambda r: (
                        int(r.metadata.get("panel_index", 9999)),
                        int(r.metadata.get("reading_order_index", 9999)),
                        self._reading_order_key(r),
                    ),
                )
            return self.reading_order_resolver.sort_regions(regiones)
        except Exception as exc:
            logger.warning("No se pudo ordenar regiones por lectura; usando fallback: %s", exc)
            return sorted(list(regiones), key=self._reading_order_key)

    def _sort_boxes_for_reading(self, boxes: Sequence[Box]) -> List[Box]:
        try:
            return self.reading_order_resolver.sort_boxes(boxes)
        except Exception as exc:
            logger.warning("No se pudo ordenar cajas por lectura; usando fallback: %s", exc)
            return sorted(list(boxes), key=self._reading_order_key)

    def _masked_region_crop(self, imagen: np.ndarray, region: TextRegion) -> np.ndarray:
        """Recorta la región y blanquea lo que queda fuera de su máscara.

        Separado del preproceso a propósito: medido sobre 38 regiones corregidas a mano,
        este enmascarado vale ~3x más que la elección de motor OCR (CER 0.336 -> 0.116),
        así que conviene poder volcarlo y estudiarlo por separado.

        Para globos usa la máscara YOLO completa: OCR dentro del globo, no dentro de la
        caja OCR antigua. Para texto libre/SFX se conserva su máscara local expandida.
        """
        height_img, width_img = imagen.shape[:2]
        use_text_hint = (
            self.ocr_region_mode in {"text", "text_hint", "tight"}
            and region.detections_count > 0
            and region.kind in {"dialogue", "narration", "unknown"}
        )
        box = region.text_bbox if use_text_hint else region.ocr_bbox
        x, y, w, h = self.geometry.clip_box_to_image(box, width_img, height_img)
        crop = imagen[y:y + h, x:x + w]
        if crop.size == 0:
            return crop

        local_mask = region.mask[y:y + h, x:x + w]
        if local_mask.size and cv2.countNonZero(local_mask) > 0:
            if local_mask.shape[:2] != crop.shape[:2]:
                local_mask = cv2.resize(local_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Deja todo lo que esté fuera del globo en blanco para que OCR no lea arte cercano.
            canvas = np.full_like(crop, 255)
            canvas[local_mask > 0] = crop[local_mask > 0]
            crop = canvas

        return crop

    def _masked_region_crop_for_ocr(self, imagen: np.ndarray, region: TextRegion) -> np.ndarray:
        """El recorte tal como lo recibe el OCR: enmascarado y preprocesado."""
        return self._prepare_crop_for_ocr(self._masked_region_crop(imagen, region))

    def obtener_areas_interes_desde_regiones(self, imagen, regiones, indice_pagina: int = 0):
        cuadros_delimitadores: List[Box] = []
        imagenes_interes = []
        regiones_ordenadas = self._sort_regions_for_reading(list(regiones))
        height_img, width_img = imagen.shape[:2]

        for indice, region in enumerate(regiones_ordenadas):
            enmascarado = self._masked_region_crop(imagen, region)
            area_limpia = self._prepare_crop_for_ocr(enmascarado)
            self._dump_ocr_crop(indice, region, enmascarado, area_limpia, indice_pagina)
            cuadros_delimitadores.append(region.render_bbox)
            imagenes_interes.append(area_limpia)

        return cuadros_delimitadores, imagenes_interes, regiones_ordenadas

    def _dump_ocr_crop(
        self,
        indice: int,
        region: TextRegion,
        enmascarado: np.ndarray,
        preparado: np.ndarray,
        indice_pagina: int = 0,
    ) -> None:
        """Vuelca lo que ve el OCR, si `quality.ocr_crop_debug_dir` lo pide.

        Se guardan las dos versiones: el enmascarado sirve para probar preprocesos
        alternativos sin volver a correr el pipeline, que es lo caro.

        Junto a cada PNG va su geometría. Es lo que permite emparejar un recorte con la
        región corregida a mano **por solapamiento** en vez de por posición en la lista:
        el editor reordena y reasigna regiones al corregir, así que el índice miente.
        """
        destino = str(getattr(self, "ocr_crop_debug_dir", "") or "").strip()
        if not destino:
            return
        try:
            from pathlib import Path as _Path

            # La pagina llega por parametro desde el contexto. Cuando la leia de
            # `self.indice_imagen` con un `getattr` por defecto, quitar ese atributo dejo
            # todos los recortes cayendo en `pagina_0000` sin que nada fallara.
            pagina = int(indice_pagina)
            carpeta = _Path(destino) / f"pagina_{pagina:04d}"
            carpeta.mkdir(parents=True, exist_ok=True)
            if enmascarado is not None and getattr(enmascarado, "size", 0):
                cv2.imwrite(str(carpeta / f"region_{indice:02d}_enmascarado.png"), enmascarado)
            if preparado is not None and getattr(preparado, "size", 0):
                cv2.imwrite(str(carpeta / f"region_{indice:02d}_preparado.png"), preparado)
            import json as _json

            (carpeta / f"region_{indice:02d}.json").write_text(
                _json.dumps(
                    {
                        "indice": int(indice),
                        "region_uid": run_region_uid(pagina + 1, indice),
                        "bbox": [int(v) for v in region.bbox],
                        "text_bbox": [int(v) for v in region.text_bbox],
                        "kind": str(region.kind or ""),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - depuración, nunca debe tumbar la página
            logger.warning("No se pudo volcar el recorte de OCR (%s).", exc)

    def obtener_areas_interes(self, imagen, mascara_capa):
        cuadros_delimitadores: List[Box] = []
        imagenes_interes = []
        height_img, width_img = imagen.shape[:2]

        boxes = self.geometry.mask_to_boxes(mascara_capa)

        for box in self._sort_boxes_for_reading(boxes):
            x, y, w, h = self.geometry.expand_box(box, width_img, height_img)
            area_interes = imagen[y:y + h, x:x + w]
            area_limpia = self._prepare_crop_for_ocr(area_interes)

            cuadros_delimitadores.append((x, y, w, h))
            imagenes_interes.append(area_limpia)

        return cuadros_delimitadores, imagenes_interes
