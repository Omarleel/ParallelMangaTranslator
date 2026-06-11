from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Utils.Constantes import COLOR_BLANCO, COLOR_NEGRO, FACTOR_ESPACIO, RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


class TextRenderer:
    """
    Renderiza traducciones dentro de la caja detectada intentando mantenerlas
    legibles y, sobre todo, evitando que el texto se salga del globo.
    """

    def __init__(
        self,
        font_path: str = RUTA_FUENTE,
        min_font_size: int = TAMANIO_MINIMO_FUENTE,
        absolute_min_font_size: int = 7,
        max_font_size: int = 96,
        inner_margin_ratio: float = 0.09,
    ) -> None:
        self.font_path = font_path
        self.min_font_size = max(1, int(min_font_size))
        self.absolute_min_font_size = max(5, min(int(absolute_min_font_size), self.min_font_size))
        self.max_font_size = max(self.min_font_size, int(max_font_size))
        self.inner_margin_ratio = max(0.02, min(0.20, float(inner_margin_ratio)))

    @lru_cache(maxsize=192)
    def _get_font(self, size: int):
        safe_size = max(self.absolute_min_font_size, int(size))
        try:
            return ImageFont.truetype(self.font_path, safe_size)
        except OSError:
            # Evita que el renderizado falle si la fuente aún no fue descargada.
            return ImageFont.load_default()

    @staticmethod
    def _text_width(texto: str, fuente) -> int:
        bbox = fuente.getbbox(texto or " ")
        return max(0, bbox[2] - bbox[0])

    @staticmethod
    def _text_height(texto: str, fuente) -> int:
        bbox = fuente.getbbox(texto or " ")
        return max(1, bbox[3] - bbox[1])

    @staticmethod
    def _line_spacing(fuente) -> float:
        size = getattr(fuente, "size", TAMANIO_MINIMO_FUENTE)
        return max(1.0, min(size * 0.22, size * FACTOR_ESPACIO))

    @staticmethod
    def _normalize_text(texto: str) -> str:
        texto = str(texto or " ").replace("\r", "\n")
        texto = re.sub(r"[ \t\f\v]+", " ", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        return texto.strip() or " "

    def _break_long_token(self, token: str, fuente, max_width: int) -> List[str]:
        """Parte palabras muy largas para que no atraviesen la caja."""
        if self._text_width(token, fuente) <= max_width:
            return [token]

        partes: List[str] = []
        actual = ""
        for char in token:
            candidata = actual + char
            if actual and self._text_width(candidata, fuente) > max_width:
                partes.append(actual)
                actual = char
            else:
                actual = candidata
        if actual:
            partes.append(actual)
        return partes or [token]

    def _split_lines(self, texto: str, fuente, max_width: int) -> List[str]:
        max_width = max(1, int(max_width))
        texto = self._normalize_text(texto)
        lineas: List[str] = []

        for parrafo in texto.split("\n"):
            palabras = parrafo.split()
            if not palabras:
                if lineas:
                    lineas.append("")
                continue

            tokens: List[str] = []
            for palabra in palabras:
                tokens.extend(self._break_long_token(palabra, fuente, max_width))

            linea_actual = ""
            for token in tokens:
                candidata = token if not linea_actual else f"{linea_actual} {token}"
                if linea_actual and self._text_width(candidata, fuente) > max_width:
                    lineas.append(linea_actual)
                    linea_actual = token
                else:
                    linea_actual = candidata
            if linea_actual:
                lineas.append(linea_actual)

        return lineas or [" "]

    def _paragraph_height(self, lineas: Sequence[str], fuente, espacio_entre_lineas: float) -> float:
        if not lineas:
            return 0.0
        alturas = [self._text_height(linea, fuente) for linea in lineas]
        return float(sum(alturas) + max(0, len(lineas) - 1) * espacio_entre_lineas)

    def _fits(self, lineas: Sequence[str], fuente, espacio: float, box_width: int, box_height: int) -> bool:
        if not lineas:
            return True
        ancho_maximo = max((self._text_width(linea, fuente) for linea in lineas), default=0)
        alto = self._paragraph_height(lineas, fuente, espacio)
        return ancho_maximo <= box_width and alto <= box_height

    def _ellipsis_line(self, linea: str, fuente, max_width: int) -> str:
        ellipsis = "…"
        linea = linea.strip()
        if self._text_width(linea + ellipsis, fuente) <= max_width:
            return linea + ellipsis
        while linea and self._text_width(linea + ellipsis, fuente) > max_width:
            linea = linea[:-1].rstrip()
        return (linea + ellipsis) if linea else ellipsis

    def _truncate_to_fit(self, lineas: List[str], fuente, box_width: int, box_height: int, espacio: float) -> List[str]:
        """Último recurso: evita desbordes si ni la fuente mínima alcanza."""
        if not lineas:
            return [" "]

        linea_base = max(1, self._text_height("Ag", fuente))
        max_lineas = max(1, int((box_height + espacio) // (linea_base + espacio)))
        recortadas = list(lineas[:max_lineas])

        if len(lineas) > max_lineas:
            recortadas[-1] = self._ellipsis_line(recortadas[-1], fuente, box_width)

        while self._paragraph_height(recortadas, fuente, espacio) > box_height and len(recortadas) > 1:
            recortadas.pop()
            recortadas[-1] = self._ellipsis_line(recortadas[-1], fuente, box_width)

        recortadas = [self._ellipsis_line(linea, fuente, box_width) if self._text_width(linea, fuente) > box_width else linea for linea in recortadas]
        return recortadas or [" "]

    def _fit_font(self, texto: str, box_width: int, box_height: int, style: str = "dialogo"):
        box_width = max(1, int(box_width))
        box_height = max(1, int(box_height))
        texto = self._normalize_text(texto)

        if style.startswith("onomatopeya"):
            # Los efectos de sonido suelen ocupar más espacio visual y admiten letras grandes.
            factor = 0.82 if style == "onomatopeya_subtitle" else 0.92
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * factor)))
        elif style == "narracion":
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * 0.58)))
        else:
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * 0.70)))

        # Intento principal: fuente más grande que quepa.
        for tamanio in range(start_size, self.absolute_min_font_size - 1, -1):
            fuente = self._get_font(tamanio)
            espacio = self._line_spacing(fuente) * (0.82 if style.startswith("onomatopeya") else 1.0)
            lineas = self._split_lines(texto, fuente, box_width)
            if self._fits(lineas, fuente, espacio, box_width, box_height):
                return fuente, lineas, espacio

        # Último recurso: fuente mínima absoluta + recorte seguro.
        fuente = self._get_font(self.absolute_min_font_size)
        espacio = self._line_spacing(fuente) * (0.82 if style.startswith("onomatopeya") else 1.0)
        lineas = self._split_lines(texto, fuente, box_width)
        lineas = self._truncate_to_fit(lineas, fuente, box_width, box_height, espacio)
        return fuente, lineas, espacio

    @staticmethod
    def _resolve_text_colors(imagen_limpia: np.ndarray, x: int, y: int, w: int, h: int):
        x_margin = max(0, x - 5)
        y_margin = max(0, y - 5)
        w_margin = min(w + 10, imagen_limpia.shape[1] - x_margin)
        h_margin = min(h + 10, imagen_limpia.shape[0] - y_margin)
        region_alrededor = imagen_limpia[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
        promedio_color = cv2.mean(region_alrededor)[:3]
        if np.mean(promedio_color) < 128:
            return COLOR_NEGRO, COLOR_BLANCO
        return COLOR_BLANCO, COLOR_NEGRO

    @staticmethod
    def _to_rgba(color):
        if len(color) == 4:
            return color
        return tuple(color) + (255,)

    @staticmethod
    def _prepare_display_text(texto: str, style: str) -> str:
        texto = str(texto or " ").strip() or " "
        if not style.startswith("onomatopeya"):
            return texto
        if style == "onomatopeya_subtitle" and "\n" in texto:
            original, subtitle = texto.split("\n", 1)
            if re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", subtitle):
                subtitle = subtitle.upper()
            return original.strip() + "\n" + subtitle.strip()
        # En alfabetos latinos las onomatopeyas suelen leerse mejor en mayúsculas.
        if re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", texto):
            texto = texto.upper()
        return texto

    @staticmethod
    def _normalize_clip_mask(clip_mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Devuelve la máscara local del globo como uint8 binaria del tamaño de render."""
        local = np.asarray(clip_mask, dtype=np.uint8)
        if local.ndim == 3:
            local = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
        if local.shape[:2] != (height, width):
            local = cv2.resize(local, (width, height), interpolation=cv2.INTER_NEAREST)
        _, local = cv2.threshold(local, 31, 255, cv2.THRESH_BINARY)
        return local

    @staticmethod
    def _largest_rect_from_row_spans(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calcula un rectángulo interior estable para máscaras convexas/semiconvexas.

        Para globos ovalados, usar toda la bbox permite que las líneas superiores o inferiores
        queden dentro del rectángulo pero fuera de la curva real del globo, y luego la máscara
        las recorta. Este método busca un rectángulo cuyas filas estén dentro de la máscara.
        """
        rows = binary_mask > 0
        height, width = rows.shape[:2]
        lefts = np.full(height, width, dtype=np.int32)
        rights = np.full(height, -1, dtype=np.int32)
        valid_rows = []

        for row_idx in range(height):
            cols = np.flatnonzero(rows[row_idx])
            if cols.size == 0:
                continue
            lefts[row_idx] = int(cols[0])
            rights[row_idx] = int(cols[-1])
            valid_rows.append(row_idx)

        if not valid_rows:
            return 0, 0, width, height

        # Reduce coste en máscaras muy altas sin perder estabilidad visual.
        step = max(1, height // 420)
        candidate_tops = range(valid_rows[0], valid_rows[-1] + 1, step)
        best = (0, int(valid_rows[0]), max(1, width), max(1, valid_rows[-1] - valid_rows[0] + 1))
        best_score = -1.0
        center_x = width / 2.0
        center_y = height / 2.0

        for top in candidate_tops:
            if rights[top] < lefts[top]:
                continue
            left = int(lefts[top])
            right = int(rights[top])
            for bottom in range(top, valid_rows[-1] + 1, step):
                if rights[bottom] < lefts[bottom]:
                    break
                left = max(left, int(lefts[bottom]))
                right = min(right, int(rights[bottom]))
                rect_w = right - left + 1
                rect_h = bottom - top + 1
                if rect_w <= 1 or rect_h <= 1:
                    continue
                area = rect_w * rect_h
                # Cuando hay varias áreas parecidas, preferimos la que queda más centrada
                # en el globo para que el bloque de texto no se pegue a la cola/borde.
                rect_cx = left + rect_w / 2.0
                rect_cy = top + rect_h / 2.0
                center_penalty = 1.0 - min(0.35, (abs(rect_cx - center_x) / max(1.0, width) + abs(rect_cy - center_y) / max(1.0, height)))
                score = area * center_penalty
                if score > best_score:
                    best_score = score
                    best = (left, top, rect_w, rect_h)

        return best

    def _safe_text_area_from_mask(self, local_mask: Optional[np.ndarray], width: int, height: int, style: str) -> Tuple[int, int, int, int]:
        """
        Devuelve la caja interior donde debe ajustarse el texto antes de aplicar clip_mask.

        El clip se mantiene como protección final, pero el ajuste de fuente/líneas usa esta
        caja interior. Así el texto se encoge/redistribuye y no queda comido por la máscara
        curva del globo.
        """
        if local_mask is None or style.startswith("onomatopeya"):
            return 0, 0, width, height

        mask = self._normalize_clip_mask(local_mask, width, height)
        if cv2.countNonZero(mask) < max(12, int(width * height * 0.02)):
            return 0, 0, width, height

        # Protege contra bordes curvos y contra el ancho del stroke. En globos grandes
        # el valor crece poco para no desperdiciar demasiado espacio.
        min_side = max(1, min(width, height))
        if style == "narracion":
            pad_ratio = 0.035
        else:
            pad_ratio = 0.055
        pad = max(2, min(18, int(round(min_side * pad_ratio))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
        inner = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(inner) < max(12, int(cv2.countNonZero(mask) * 0.25)):
            inner = mask

        sx, sy, sw, sh = self._largest_rect_from_row_spans(inner)
        if sw < max(8, width * 0.15) or sh < max(8, height * 0.15):
            x, y, w, h = cv2.boundingRect(inner)
            sx, sy, sw, sh = int(x), int(y), int(w), int(h)

        # Un último margen mínimo evita que el stroke caiga justo sobre el límite del rectángulo seguro.
        extra = max(1, min(4, int(round(min(sw, sh) * 0.025))))
        sx = min(width - 1, max(0, sx + extra))
        sy = min(height - 1, max(0, sy + extra))
        sw = max(1, min(width - sx, sw - 2 * extra))
        sh = max(1, min(height - sy, sh - 2 * extra))
        return sx, sy, sw, sh

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

    @staticmethod
    def _sentence_units(texto: str) -> List[str]:
        texto = TextRenderer._normalize_text(texto)
        # Conserva el signo de puntuación en la unidad. Incluye puntuación española,
        # japonesa y puntos suspensivos para separar frases naturales de un mismo globo doble.
        units = re.findall(r".+?(?:[.!?。！？…]+|$)(?:\s+|$)", texto, flags=re.S)
        units = [re.sub(r"\s+", " ", unit).strip() for unit in units if unit.strip()]
        return units or [texto]

    def _split_text_for_slots(self, texto: str, slot_count: int) -> List[str]:
        slot_count = max(1, int(slot_count))
        texto = self._normalize_text(texto)
        if slot_count == 1:
            return [texto]

        units = self._sentence_units(texto)
        if len(units) >= slot_count:
            chunks: List[str] = []
            remaining_units = list(units)
            for slot_idx in range(slot_count):
                remaining_slots = slot_count - slot_idx
                if remaining_slots == 1:
                    chunks.append(" ".join(remaining_units).strip())
                    break
                remaining_chars = sum(len(unit) for unit in remaining_units)
                target = max(1, remaining_chars / remaining_slots)
                current: List[str] = []
                current_len = 0
                while remaining_units and len(remaining_units) > remaining_slots - 1:
                    next_unit = remaining_units[0]
                    if current and current_len + len(next_unit) > target * 1.18:
                        break
                    current.append(remaining_units.pop(0))
                    current_len += len(next_unit)
                    if current_len >= target * 0.82:
                        break
                chunks.append(" ".join(current).strip())
            return [(chunk or " ") for chunk in chunks[:slot_count]]

        # Si no hay suficientes frases, parte por palabras procurando equilibrio.
        words = texto.split()
        if len(words) < slot_count * 2:
            # Último recurso para idiomas sin espacios: corte por caracteres.
            total = len(texto)
            chunks = []
            start = 0
            for slot_idx in range(slot_count):
                end = total if slot_idx == slot_count - 1 else int(round(total * (slot_idx + 1) / slot_count))
                chunks.append(texto[start:end].strip() or " ")
                start = end
            return chunks

        chunks = []
        start = 0
        for slot_idx in range(slot_count):
            remaining_slots = slot_count - slot_idx
            if remaining_slots == 1:
                chunks.append(" ".join(words[start:]).strip() or " ")
                break
            remaining_words = len(words) - start
            take = max(1, int(round(remaining_words / remaining_slots)))
            chunks.append(" ".join(words[start:start + take]).strip() or " ")
            start += take
        return chunks

    def render(
        self,
        imagen_limpia: np.ndarray,
        cuadros_delimitadores: Sequence[Tuple[int, int, int, int]],
        textos: Sequence[str],
        text_styles: Optional[Sequence[str]] = None,
        clip_masks: Optional[Sequence[np.ndarray]] = None,
        *,
        reading_order_right_to_left: bool = False,
    ) -> np.ndarray:
        # OpenCV trabaja en BGR; PIL trabaja en RGB. Convertir explícitamente evita
        # desplazamientos de color en páginas a color.
        imagen_pil = Image.fromarray(cv2.cvtColor(imagen_limpia, cv2.COLOR_BGR2RGB))
        ancho_img, alto_img = imagen_pil.size
        if text_styles is None:
            text_styles = ["dialogo"] * len(textos)

        if clip_masks is None:
            clip_masks = [None] * len(textos)

        for (x, y, w, h), texto, style, clip_mask in zip(cuadros_delimitadores, textos, text_styles, clip_masks):
            style = style or "dialogo"
            texto = self._prepare_display_text(texto, style)
            x = int(max(0, x))
            y = int(max(0, y))
            w = int(min(max(1, w), ancho_img - x))
            h = int(min(max(1, h), alto_img - y))
            if w <= 2 or h <= 2:
                continue

            split_slots = self._connected_lobe_slots_from_mask(
                clip_mask,
                w,
                h,
                style,
                right_to_left=reading_order_right_to_left,
            )
            if split_slots:
                slot_texts = self._split_text_for_slots(texto, len(split_slots))
                render_blocks = list(zip(split_slots, slot_texts))
            else:
                safe_x, safe_y, safe_w, safe_h = self._safe_text_area_from_mask(clip_mask, w, h, style)
                render_blocks = [((safe_x, safe_y, safe_w, safe_h), texto)]

            color_borde, color_texto = self._resolve_text_colors(imagen_limpia, x, y, w, h)

            # Dibujamos en una capa del tamaño exacto del globo/caja y la pegamos con máscara.
            # Así se garantiza que ningún píxel de texto quede fuera de la región asignada.
            capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(capa)

            for (safe_x, safe_y, safe_w, safe_h), block_text in render_blocks:
                # Margen interno: en diálogos protege esquinas de globos ovalados; en SFX se reduce
                # para que el efecto pueda ocupar más del espacio detectado. Con clip_mask, el margen
                # se aplica sobre la caja interior real del globo, no sobre la bbox rectangular completa.
                margin_ratio = self.inner_margin_ratio if not style.startswith("onomatopeya") else max(0.035, self.inner_margin_ratio * 0.45)
                margen_x = max(2, int(safe_w * margin_ratio))
                margen_y = max(2, int(safe_h * margin_ratio))
                area_x = safe_x + margen_x
                area_y = safe_y + margen_y
                area_w = max(1, safe_w - 2 * margen_x)
                area_h = max(1, safe_h - 2 * margen_y)

                fuente, lineas, espacio_entre_lineas = self._fit_font(block_text or " ", area_w, area_h, style=style)
                alto_parrafo = self._paragraph_height(lineas, fuente, espacio_entre_lineas)
                if style.startswith("onomatopeya"):
                    stroke_width = max(1, min(5, int(getattr(fuente, "size", self.min_font_size) * 0.11)))
                elif style == "narracion":
                    stroke_width = max(1, min(2, int(getattr(fuente, "size", self.min_font_size) * 0.045)))
                else:
                    stroke_width = max(1, min(3, int(getattr(fuente, "size", self.min_font_size) * 0.07)))

                y_texto = area_y + max(0, (area_h - alto_parrafo) / 2)
                for linea in lineas:
                    alto_linea = self._text_height(linea, fuente)
                    ancho_linea = self._text_width(linea, fuente)
                    x_texto = area_x + max(0, (area_w - ancho_linea) / 2)
                    draw.text(
                        (x_texto, y_texto),
                        linea,
                        font=fuente,
                        fill=self._to_rgba(color_texto),
                        stroke_width=stroke_width,
                        stroke_fill=self._to_rgba(color_borde),
                    )
                    y_texto += alto_linea + espacio_entre_lineas

            del draw

            paste_mask = capa
            if clip_mask is not None and not style.startswith("onomatopeya"):
                local = self._normalize_clip_mask(clip_mask, w, h)
                # Suaviza un poco el recorte para que el texto no quede serruchado cerca del borde.
                local = cv2.GaussianBlur(local, (0, 0), sigmaX=0.8, sigmaY=0.8)
                alpha = np.array(capa.getchannel("A"), dtype=np.float32)
                clipped_alpha = np.minimum(alpha, local.astype(np.float32))
                capa.putalpha(Image.fromarray(np.clip(clipped_alpha, 0, 255).astype(np.uint8)))
                paste_mask = capa

            imagen_pil.paste(capa, (x, y), paste_mask)

        return cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
