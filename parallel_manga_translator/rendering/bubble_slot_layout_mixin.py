from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parallel_manga_translator.config.constants import COLOR_BLANCO, COLOR_NEGRO, FACTOR_ESPACIO, RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


class BubbleSlotLayoutMixin:
    """Distribución de texto entre lóbulos/componentes de un globo."""

    @staticmethod
    def _component_records(binary_mask: np.ndarray, min_area: int) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], int]]:
        """Componentes significativos de una máscara binaria local."""
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            np.uint8(binary_mask > 0) * 255,
            connectivity=8,
        )
        records: List[Tuple[np.ndarray, Tuple[int, int, int, int], int]] = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            if w < 8 or h < 8:
                continue
            comp = np.zeros_like(binary_mask, dtype=np.uint8)
            comp[labels == label_idx] = 255
            records.append((comp, (x, y, w, h), area))
        records.sort(key=lambda item: item[2], reverse=True)
        return records

    @staticmethod
    def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _component_centroid(component: np.ndarray, fallback_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        moments = cv2.moments(np.uint8(component > 0), binaryImage=True)
        if abs(moments.get("m00", 0.0)) > 1e-6:
            return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
        x, y, w, h = fallback_box
        return x + w / 2.0, y + h / 2.0

    def _lobe_slots_from_centers(
        self,
        mask: np.ndarray,
        centers: Sequence[Tuple[float, float]],
        width: int,
        height: int,
        style: str,
        *,
        right_to_left: bool = False,
        min_lobe_area_ratio: float = 0.07,
        erosion_radius: int = 8,
    ) -> List[Tuple[int, int, int, int]]:
        centers = list(centers)
        if len(centers) < 2:
            return []

        total_area = int(cv2.countNonZero(mask))
        if total_area <= 0:
            return []

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []

        centers_arr = np.array(centers, dtype=np.float32)
        points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        distances = ((points[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(distances, axis=1)

        slots: List[Tuple[int, int, int, int]] = []
        seen: List[Tuple[int, int, int, int]] = []
        for idx in range(len(centers)):
            lobe = np.zeros_like(mask, dtype=np.uint8)
            selected = nearest == idx
            lobe[ys[selected], xs[selected]] = 255
            lobe_area = int(cv2.countNonZero(lobe))
            if lobe_area < max(80, int(total_area * float(min_lobe_area_ratio))):
                continue

            local_radius = max(2, min(22, int(round(erosion_radius))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (local_radius * 2 + 1, local_radius * 2 + 1))
            lobe_inner = cv2.erode(lobe, kernel, iterations=1)
            if cv2.countNonZero(lobe_inner) < max(50, int(lobe_area * 0.18)):
                lobe_inner = lobe

            slot = self._safe_text_area_from_mask(lobe_inner, width, height, style)
            sx, sy, sw, sh = slot
            if sw < max(18, width * 0.09) or sh < max(18, height * 0.09):
                continue
            if sw * sh < max(220, int(width * height * 0.018)):
                continue

            duplicate = False
            for other in seen:
                ox, oy, ow, oh = other
                inter_x1 = max(sx, ox)
                inter_y1 = max(sy, oy)
                inter_x2 = min(sx + sw, ox + ow)
                inter_y2 = min(sy + sh, oy + oh)
                inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                if inter / max(1, min(sw * sh, ow * oh)) > 0.65:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen.append(slot)
            slots.append(slot)

        if len(slots) < 2:
            return []
        return self._order_slots_for_reading(slots, right_to_left=right_to_left)

    def _distance_lobe_slots_from_mask(
        self,
        mask: np.ndarray,
        width: int,
        height: int,
        style: str,
        *,
        right_to_left: bool = False,
    ) -> List[Tuple[int, int, int, int]]:
        """Fallback para globos conectados en diagonal o con cuello ancho.

        La erosión simple sólo separa bien globos unidos por un cuello muy estrecho.
        En páginas reales, dos globos pueden tocarse con una unión ancha y aun así
        tener dos centros visuales claros. El mapa de distancia encuentra esos
        centros internos y permite repartir la traducción entre los dos lóbulos.
        """
        if style.startswith("onomatopeya"):
            return []

        total_area = int(cv2.countNonZero(mask))
        if total_area < max(180, int(width * height * 0.08)):
            return []

        dist = cv2.distanceTransform(np.uint8(mask > 0), cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        min_side = max(1, min(width, height))
        if max_dist < max(8.0, min_side * 0.055):
            return []

        best_centers: List[Tuple[float, float]] = []
        best_score = -1.0
        # Umbrales altos: buscan núcleos de lóbulos, no toda la masa del globo.
        for frac in (0.72, 0.68, 0.64, 0.60, 0.56, 0.52):
            seed = np.uint8(dist >= max_dist * frac) * 255
            seed = cv2.morphologyEx(
                seed,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            components = self._component_records(seed, min_area=max(20, int(total_area * 0.0025)))
            if len(components) < 2:
                continue

            largest = max(1, components[0][2])
            filtered = [
                comp for comp in components
                if comp[2] >= max(24, int(largest * 0.10), int(total_area * 0.0035))
            ][:3]
            if len(filtered) < 2:
                continue

            centers = [self._component_centroid(comp, box) for comp, box, _area in filtered]
            # Las semillas deben representar centros visuales distintos. Si están
            # demasiado cerca, probablemente son irregularidades de un solo globo.
            min_pair_dist = min(
                float(np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))
                for i in range(len(centers))
                for j in range(i + 1, len(centers))
            )
            if min_pair_dist < max(38.0, min_side * 0.26):
                continue

            areas = [area for _comp, _box, area in filtered]
            balance = min(areas) / max(1, max(areas))
            coverage = sum(areas) / max(1, int(cv2.countNonZero(seed)))
            spread = min(1.0, min_pair_dist / max(1.0, min(width, height)))
            score = len(filtered) * 2.0 + min(0.9, balance) + min(1.0, coverage) + spread
            if score > best_score:
                best_score = score
                best_centers = centers

        if len(best_centers) < 2:
            return []

        return self._lobe_slots_from_centers(
            mask,
            best_centers,
            width,
            height,
            style,
            right_to_left=right_to_left,
            min_lobe_area_ratio=0.06,
            erosion_radius=max(4, min(20, int(round(max_dist * 0.10)))),
        )

    def _order_slots_for_reading(
        self,
        slots: Sequence[Tuple[int, int, int, int]],
        *,
        right_to_left: bool = False,
    ) -> List[Tuple[int, int, int, int]]:
        if len(slots) <= 1:
            return list(slots)
        heights = [max(1, slot[3]) for slot in slots]
        band = max(24.0, float(np.median(heights)) * 0.65)

        def key(slot: Tuple[int, int, int, int]):
            x, y, w, h = slot
            cx, cy = self._box_center(slot)
            row = int(round(cy / band))
            return row, -cx if right_to_left else cx, y, x

        return sorted(list(slots), key=key)

    def _connected_lobe_slots_from_mask(
        self,
        local_mask: Optional[np.ndarray],
        width: int,
        height: int,
        style: str,
        *,
        right_to_left: bool = False,
    ) -> List[Tuple[int, int, int, int]]:
        """Detecta globos conectados y devuelve slots seguros, uno por lóbulo.

        Los modelos de globos a veces devuelven una sola máscara cuando dos globos
        se tocan. Si se usa el rectángulo interior máximo de toda esa máscara, el
        texto traducido cae en un solo lóbulo y el otro queda vacío. Esta rutina
        intenta separar lóbulos por la estrechez del cuello usando erosión; no toca
        onomatopeyas ni máscaras simples.
        """
        if local_mask is None or style.startswith("onomatopeya"):
            return []
        if width < 48 or height < 48:
            return []

        mask = self._normalize_clip_mask(local_mask, width, height)
        total_area = int(cv2.countNonZero(mask))
        if total_area < max(120, int(width * height * 0.08)):
            return []

        min_side = max(1, min(width, height))
        best_components: List[Tuple[np.ndarray, Tuple[int, int, int, int], int]] = []
        best_radius = 0
        best_score = -1.0

        # Probamos varias erosiones: una pequeña no rompe el cuello; una excesiva
        # destruye lóbulos estrechos. Elegimos la separación con mejor cobertura.
        for ratio in (0.055, 0.075, 0.095, 0.12, 0.15):
            radius = max(3, min(36, int(round(min_side * ratio))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            eroded = cv2.erode(mask, kernel, iterations=1)
            if cv2.countNonZero(eroded) < max(80, int(total_area * 0.18)):
                continue
            min_area = max(36, int(total_area * 0.025))
            components = self._component_records(eroded, min_area=min_area)
            if len(components) < 2:
                continue

            largest = max(1, components[0][2])
            filtered = [
                comp for comp in components
                if comp[2] >= max(40, int(largest * 0.20), int(total_area * 0.018))
            ]
            if len(filtered) < 2:
                continue

            # Evita convertir agujeros/trozos pequeños en slots. Para diálogos,
            # dos o tres lóbulos son útiles; más suele ser ruido de colas/bordes.
            filtered = filtered[:3]
            coverage = sum(comp[2] for comp in filtered) / max(1, int(cv2.countNonZero(eroded)))
            balance = min(comp[2] for comp in filtered) / max(1, max(comp[2] for comp in filtered))
            score = len(filtered) * 2.0 + coverage + min(0.75, balance)
            if score > best_score:
                best_score = score
                best_components = filtered
                best_radius = radius

        if len(best_components) < 2:
            return self._distance_lobe_slots_from_mask(
                mask,
                width,
                height,
                style,
                right_to_left=right_to_left,
            )

        centers = [self._box_center(box) for _comp, box, _area in best_components]
        slots = self._lobe_slots_from_centers(
            mask,
            centers,
            width,
            height,
            style,
            right_to_left=right_to_left,
            min_lobe_area_ratio=0.07,
            erosion_radius=max(2, min(18, int(round(best_radius * 0.45)))),
        )
        if len(slots) < 2:
            return self._distance_lobe_slots_from_mask(
                mask,
                width,
                height,
                style,
                right_to_left=right_to_left,
            )
        return slots
