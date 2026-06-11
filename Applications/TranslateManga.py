from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from Applications.OcrManager import OcrManager
from Applications.OnomatopoeiaManager import OnomatopoeiaManager
from Applications.TextNormalization import OcrTextNormalizer
from Applications.TextRendering import TextRenderer
from Applications.TranslatorManager import TranslatorManager
from Applications.ProcessingModels import TextRegion
from Applications.ReadingOrderResolver import ReadingOrderResolver
from .LoggingConfig import get_logger
from Utils.Constantes import RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


Box = Tuple[int, int, int, int]

logger = get_logger(__name__)


class TranslateManga:
    def __init__(self, idioma_entrada, idioma_salida, metodo_traduccion="Tradicional", groq_api_key="", lore_manga=""):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.idioma_entrada = idioma_entrada
        self.idioma_salida = idioma_salida
        self.metodo_traduccion = metodo_traduccion

        self.translator_manager = TranslatorManager(
            idioma_entrada,
            idioma_salida,
            metodo=metodo_traduccion,
            groq_api_key=groq_api_key,
            lore_manga=lore_manga,
        )
        self.ocr_manager = OcrManager(idioma_entrada=idioma_entrada)
        self.reading_order_resolver = ReadingOrderResolver(idioma_entrada)
        self.text_renderer = TextRenderer(font_path=RUTA_FUENTE, min_font_size=TAMANIO_MINIMO_FUENTE)
        self.text_normalizer = OcrTextNormalizer()
        self.onomatopoeia_manager = OnomatopoeiaManager()
        self.historial_contexto = deque(maxlen=3)
        self.ultimo_estilos_texto = []
        self.ultimas_regiones: List[TextRegion] = []
        self.ultimos_textos_originales: List[str] = []
        self.ultimos_textos_traducidos: List[str] = []
        self.ultimas_asignaciones_hablante: List[Dict[str, Any]] = []
        self.onomatopoeia_mode = os.getenv("PMT_ONOMATOPOEIA_MODE", "translate").strip().lower()
        if os.getenv("PMT_TRANSLATE_ONOMATOPOEIA", "1").strip().lower() in {"0", "false", "no", "off"}:
            self.onomatopoeia_mode = "keep"
        # bubble: OCR sobre el globo completo segmentado; text_hint: recorte más ajustado si hubo OCR global.
        self.ocr_region_mode = os.getenv("PMT_OCR_REGION_MODE", "bubble").strip().lower()
        self.indice_imagen = 0
        self.transcripcion_queue = None
        self.traduccion_queue = None

    def insertar_json_queue(self, indice_imagen, transcripcion_queue, traduccion_queue):
        self.indice_imagen = indice_imagen
        self.transcripcion_queue = transcripcion_queue
        self.traduccion_queue = traduccion_queue

    def traducir_manga(self, imagen, imagen_limpia, mascara_capa, text_regions=None):
        if text_regions:
            cuadros_delimitadores, imagenes_interes, regiones_ordenadas = self.obtener_areas_interes_desde_regiones(imagen, text_regions)
            self.ultimas_regiones = regiones_ordenadas
        else:
            cuadros_delimitadores, imagenes_interes = self.obtener_areas_interes(imagen, mascara_capa)
            self.ultimas_regiones = []
        textos = self.obtener_textos(imagenes_interes)
        return self.incrustar_textos(imagen_limpia, cuadros_delimitadores, textos)

    @staticmethod
    def _rect_from_contour(contour) -> Box:
        x, y, w, h = cv2.boundingRect(contour)
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _box_area(box: Box) -> int:
        return max(0, box[2]) * max(0, box[3])

    @staticmethod
    def _union(a: Box, b: Box) -> Box:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _overlap_ratio_1d(a1: int, a2: int, b1: int, b2: int) -> float:
        inter = max(0, min(a2, b2) - max(a1, b1))
        denom = max(1, min(a2 - a1, b2 - b1))
        return inter / denom

    def _should_merge_boxes(self, a: Box, b: Box) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        gap_x = max(0, max(bx - ax2, ax - bx2))
        gap_y = max(0, max(by - ay2, ay - by2))
        avg_h = max(1, (ah + bh) / 2)
        avg_w = max(1, (aw + bw) / 2)
        x_overlap = self._overlap_ratio_1d(ax, ax2, bx, bx2)
        y_overlap = self._overlap_ratio_1d(ay, ay2, by, by2)

        # Líneas de un mismo globo suelen estar una debajo de otra y comparten rango X.
        if x_overlap >= 0.22 and gap_y <= max(12, avg_h * 1.45):
            return True

        # Texto japonés vertical: columnas cercanas con bastante solape vertical.
        if self.idioma_entrada == "Japonés" and y_overlap >= 0.22 and gap_x <= max(10, avg_w * 1.15):
            return True

        # Fragmentos rotos de una misma palabra/línea.
        if y_overlap >= 0.45 and gap_x <= max(10, avg_h * 0.80):
            return True

        # Onomatopeyas estilizadas: EasyOCR/inpainting puede separar letras enormes o trazos
        # decorativos. Estas reglas unen componentes próximos sin exigir tanto solape.
        if y_overlap >= 0.18 and gap_x <= max(18, avg_h * 1.35, avg_w * 0.80):
            return True
        if x_overlap >= 0.18 and gap_y <= max(18, avg_w * 1.35, avg_h * 0.80):
            return True

        return False

    def _merge_boxes(self, boxes: Sequence[Box]) -> List[Box]:
        merged = list(boxes)
        changed = True
        while changed:
            changed = False
            result: List[Box] = []
            consumed = [False] * len(merged)
            for i, box in enumerate(merged):
                if consumed[i]:
                    continue
                current = box
                consumed[i] = True
                for j in range(i + 1, len(merged)):
                    if consumed[j]:
                        continue
                    if self._should_merge_boxes(current, merged[j]):
                        current = self._union(current, merged[j])
                        consumed[j] = True
                        changed = True
                result.append(current)
            merged = result
        return merged

    def _mask_to_boxes(self, mascara_capa: np.ndarray) -> List[Box]:
        height, width = mascara_capa.shape[:2]
        _, mascara_binaria = cv2.threshold(mascara_capa, 127, 255, cv2.THRESH_BINARY)
        mascara_binaria = np.uint8(mascara_binaria)

        if self.idioma_entrada == "Japonés":
            kernel_h = max(3, round(height * 0.011))
            kernel_w = max(3, round(width * 0.006))
        else:
            kernel_h = max(3, round(height * 0.008))
            kernel_w = max(5, round(width * 0.020))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        mascara_agrupada = cv2.morphologyEx(mascara_binaria, cv2.MORPH_CLOSE, kernel, iterations=1)
        mascara_agrupada = cv2.dilate(mascara_agrupada, kernel, iterations=1)

        contours, _ = cv2.findContours(mascara_agrupada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_img = height * width
        min_area = max(20, int(area_img * 0.00003))
        boxes = []
        for contour in contours:
            x, y, w, h = self._rect_from_contour(contour)
            if w < 3 or h < 3 or self._box_area((x, y, w, h)) < min_area:
                continue
            boxes.append((x, y, w, h))

        return self._merge_boxes(boxes)

    @staticmethod
    def _expand_box(box: Box, width_img: int, height_img: int) -> Box:
        x, y, w, h = box
        # Margen proporcional: más grande en globos complejos, pero acotado para no invadir viñetas vecinas.
        aspect = max(w, h) / max(1, min(w, h))
        if aspect >= 3.2:
            # SFX/onomatopeyas largas necesitan algo más de aire para no cortar contornos.
            pad_x = int(min(42, max(5, round(w * 0.10))))
            pad_y = int(min(38, max(5, round(h * 0.14))))
        else:
            pad_x = int(min(30, max(6, round(w * 0.18))))
            pad_y = int(min(26, max(6, round(h * 0.22))))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width_img, x + w + pad_x)
        y2 = min(height_img, y + h + pad_y)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

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

    @staticmethod
    def _clip_box_to_image(box: Box, width_img: int, height_img: int) -> Box:
        x, y, w, h = box
        x = int(max(0, min(width_img - 1, x)))
        y = int(max(0, min(height_img - 1, y)))
        x2 = int(max(x + 1, min(width_img, x + max(1, w))))
        y2 = int(max(y + 1, min(height_img, y + max(1, h))))
        return x, y, x2 - x, y2 - y

    def _masked_region_crop_for_ocr(self, imagen: np.ndarray, region: TextRegion) -> np.ndarray:
        """Prepara un recorte de OCR desde la región detectada.

        Para globos usa la máscara profesional completa: OCR dentro del globo, no dentro de la
        caja OCR antigua. Para texto libre/SFX se conserva su máscara local expandida.
        """
        height_img, width_img = imagen.shape[:2]
        use_text_hint = (
            self.ocr_region_mode in {"text", "text_hint", "tight"}
            and region.detections_count > 0
            and region.kind in {"dialogue", "narration", "unknown"}
        )
        box = region.text_bbox if use_text_hint else region.ocr_bbox
        x, y, w, h = self._clip_box_to_image(box, width_img, height_img)
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

        return self._prepare_crop_for_ocr(crop)

    def obtener_areas_interes_desde_regiones(self, imagen, regiones):
        cuadros_delimitadores: List[Box] = []
        imagenes_interes = []
        regiones_ordenadas = self._sort_regions_for_reading(list(regiones))
        height_img, width_img = imagen.shape[:2]

        for region in regiones_ordenadas:
            area_limpia = self._masked_region_crop_for_ocr(imagen, region)
            cuadros_delimitadores.append(region.render_bbox)
            imagenes_interes.append(area_limpia)

        return cuadros_delimitadores, imagenes_interes, regiones_ordenadas

    def obtener_areas_interes(self, imagen, mascara_capa):
        cuadros_delimitadores: List[Box] = []
        imagenes_interes = []
        height_img, width_img = imagen.shape[:2]

        boxes = self._mask_to_boxes(mascara_capa)

        for box in self._sort_boxes_for_reading(boxes):
            x, y, w, h = self._expand_box(box, width_img, height_img)
            area_interes = imagen[y:y + h, x:x + w]
            area_limpia = self._prepare_crop_for_ocr(area_interes)

            cuadros_delimitadores.append((x, y, w, h))
            imagenes_interes.append(area_limpia)

        return cuadros_delimitadores, imagenes_interes

    def obtener_textos(self, imagenes_interes):
        textos = self.ocr_manager.extract_texts(imagenes_interes)
        return [self.normalizar_texto_ocr(texto) for texto in textos]

    def reemplazar_caracter_especial(self, texto):
        return self.text_normalizer.replace_special_characters(texto)

    def suprimir_caracteres_repetidos(self, texto, min_reps=3):
        return self.text_normalizer.suppress_repeated_characters(texto, min_reps=min_reps)

    def suprimir_simbolos_y_espacios(self, texto):
        return self.text_normalizer.suppress_symbols_and_spaces(texto)

    def normalizar_texto_ocr(self, texto: str) -> str:
        return self.text_normalizer.normalize_ocr_text(texto)

    def _es_onomatopeya_de_texto_libre(self, indice: int, texto: str) -> bool:
        """Detecta SFX/onomatopeyas que el detector dejó como free_text.

        En ese caso se conserva el original y no se envía al traductor normal/LLM,
        porque las onomatopeyas grandes fuera de globo suelen ser parte del arte.
        """
        if not (self.ultimas_regiones and indice < len(self.ultimas_regiones)):
            return False
        region = self.ultimas_regiones[indice]
        if region.kind != "free_text":
            return False

        candidatos = [texto, getattr(region, "source_text_hint", "")]
        for candidato in candidatos:
            if self.onomatopoeia_manager.is_free_text_onomatopoeia(candidato, self.idioma_entrada):
                if hasattr(region, "metadata"):
                    match = self.onomatopoeia_manager.similar_semantic_key(candidato, self.idioma_entrada)
                    if match:
                        _key, score, source = match
                        region.metadata["free_text_onomatopoeia_keep"] = True
                        region.metadata["free_text_onomatopoeia_similarity"] = round(float(score), 4)
                        region.metadata["free_text_onomatopoeia_source"] = source
                return True
        return False

    def _clasificar_estilos_texto(self, textos: Sequence[str]) -> List[str]:
        estilos = [
            self.onomatopoeia_manager.render_style(texto, self.idioma_entrada)
            for texto in textos
        ]
        if self.ultimas_regiones and len(self.ultimas_regiones) == len(estilos):
            for i, region in enumerate(self.ultimas_regiones):
                if region.kind in {"sfx", "onomatopoeia"}:
                    estilos[i] = "onomatopeya"
                elif region.kind == "free_text" and self._es_onomatopeya_de_texto_libre(i, textos[i]):
                    estilos[i] = "onomatopeya"
                elif region.kind == "narration":
                    estilos[i] = "narracion"
        if self.onomatopoeia_mode == "subtitle":
            estilos = ["onomatopeya_subtitle" if e == "onomatopeya" else e for e in estilos]
        return estilos

    def _traducir_onomatopeyas_con_diccionario(self, textos: Sequence[str]):
        """
        Devuelve una lista parcial de traducciones para onomatopeyas conocidas.
        Las posiciones con None quedan para el traductor normal o LLM.
        """
        parciales = []
        for indice, texto in enumerate(textos):
            if self._es_onomatopeya_de_texto_libre(indice, texto):
                parciales.append(texto)
                continue
            if self.onomatopoeia_mode in {"keep", "original", "none", "off"}:
                parciales.append(texto if self.onomatopoeia_manager.is_onomatopoeia(texto, self.idioma_entrada) else None)
                continue
            traduccion = self.onomatopoeia_manager.translate(
                texto,
                idioma_entrada=self.idioma_entrada,
                idioma_salida=self.idioma_salida,
            )
            if traduccion is not None and self.onomatopoeia_mode == "subtitle":
                traduccion = f"{texto}\n({traduccion})"
            parciales.append(traduccion)
        return parciales

    def _region_metadata_for_translation(self, textos: Sequence[str]) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        for idx, texto in enumerate(textos):
            region = self.ultimas_regiones[idx] if idx < len(self.ultimas_regiones) else None
            row: Dict[str, Any] = {"source_index": idx, "max_chars": max(16, min(120, int(len(str(texto or "")) * 1.8 + 18)))}
            if region is not None:
                x, y, w, h = region.bbox
                row.update({
                    "kind": region.kind,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": round(float(region.confidence), 4),
                    "reading_order_index": region.metadata.get("reading_order_index"),
                })
            else:
                row.update({"kind": "dialogue", "bbox": None, "confidence": 0.0})
            metadata.append(row)
        return metadata

    def _metadata_with_speaker_assignments(
        self,
        base_metadata: Sequence[Mapping[str, Any]],
        assignments: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_id = {int(row.get("text_id")): row for row in assignments if isinstance(row, Mapping) and str(row.get("text_id", "")).lstrip("-").isdigit()}
        memory_chars = self.translator_manager.character_memory_snapshot().get("characters", []) if self.metodo_traduccion == "LLM" else []
        styles_by_speaker = {
            str(char.get("id")): str(char.get("speech_style") or "")
            for char in memory_chars
            if isinstance(char, Mapping)
        }
        enriched: List[Dict[str, Any]] = []
        for idx, row in enumerate(base_metadata):
            merged = dict(row)
            assignment = by_id.get(idx)
            if assignment:
                speaker_id = str(assignment.get("speaker_id") or "unknown")
                merged.update({
                    "speaker_id": speaker_id,
                    "speaker_confidence": round(float(assignment.get("confidence") or 0.0), 4),
                    "is_narration": bool(assignment.get("is_narration")),
                    "speech_style": styles_by_speaker.get(speaker_id, ""),
                })
            enriched.append(merged)
        return enriched

    def traducir_textos(self, textos: Sequence[str]):
        textos_limpios = [self.normalizar_texto_ocr(texto) for texto in textos]
        self.ultimo_estilos_texto = self._clasificar_estilos_texto(textos_limpios)
        base_metadata = self._region_metadata_for_translation(textos_limpios)

        self.ultimas_asignaciones_hablante = []
        metadata_con_hablantes = list(base_metadata)
        if self.metodo_traduccion == "LLM":
            self.ultimas_asignaciones_hablante = self.translator_manager.analyze_character_memory(
                textos_limpios,
                page_index=self.indice_imagen,
                region_metadata=base_metadata,
                contexto_previo=list(self.historial_contexto),
            )
            metadata_con_hablantes = self._metadata_with_speaker_assignments(base_metadata, self.ultimas_asignaciones_hablante)

        traducciones_onomatopeyas = self._traducir_onomatopeyas_con_diccionario(textos_limpios)
        indices_por_traducir = [
            i for i, traduccion in enumerate(traducciones_onomatopeyas)
            if traduccion is None
        ]

        textos_traducidos_brutos = list(textos_limpios)

        if indices_por_traducir:
            payload = [textos_limpios[i] for i in indices_por_traducir]
            if self.metodo_traduccion == "LLM":
                payload_metadata = [dict(metadata_con_hablantes[i], source_index=i) for i in indices_por_traducir]
                traducidos_payload = self.translator_manager.traducir_textos_llm(
                    payload,
                    contexto_previo=list(self.historial_contexto),
                    items_metadata=payload_metadata,
                    character_memory=self.translator_manager.character_memory_snapshot(),
                )
            else:
                traducidos_payload = self.translator_manager.traducir_textos_tradicional(payload)

            for idx, texto_traducido in zip(indices_por_traducir, traducidos_payload):
                textos_traducidos_brutos[idx] = texto_traducido

        for idx, traduccion in enumerate(traducciones_onomatopeyas):
            if traduccion is not None:
                textos_traducidos_brutos[idx] = traduccion

        textos_traducidos_limpios = [
            self.text_normalizer.normalize_translated_text(texto_traducido, estilo)
            for texto_traducido, estilo in zip(textos_traducidos_brutos, self.ultimo_estilos_texto)
        ]

        if self.metodo_traduccion == "LLM":
            contexto_bilingue = [
                f"{orig} -> {trad}"
                for orig, trad in zip(textos_limpios, textos_traducidos_limpios)
                if orig.strip() and trad.strip()
            ]
            if contexto_bilingue:
                self.historial_contexto.append(contexto_bilingue)

        return textos_traducidos_limpios

    def _push_original_texts_to_queue(self, cuadros_delimitadores, textos):
        if self.transcripcion_queue is None:
            return
        for idx, ((x, y, w, h), texto) in enumerate(zip(cuadros_delimitadores, textos)):
            region = self.ultimas_regiones[idx] if idx < len(self.ultimas_regiones) else None
            elemento = {
                "Índice": idx,
                "Coordenadas": [[x, y], [x + w, y + h]],
                "Texto": texto,
            }
            if region is not None:
                elemento.update({
                    "Tipo": region.kind,
                    "Confianza": round(float(region.confidence), 4),
                    "Coordenadas texto original": [[region.text_bbox[0], region.text_bbox[1]], [region.text_bbox[0] + region.text_bbox[2], region.text_bbox[1] + region.text_bbox[3]]],
                    "Fuente máscara": region.metadata.get("mask_source", ""),
                })
            if idx < len(self.ultimas_asignaciones_hablante):
                speaker = self.ultimas_asignaciones_hablante[idx]
                elemento.update({
                    "Hablante": speaker.get("speaker_id", "unknown"),
                    "Confianza hablante": round(float(speaker.get("confidence") or 0.0), 4),
                    "Evidencia hablante": speaker.get("evidence", ""),
                })
            self.transcripcion_queue.put({
                "agregar_a_sublista": {
                    "clave_lista": "Transcripción",
                    "pagina": self.indice_imagen + 1,
                    "clave_sublista": "Globos de texto",
                    "elemento_sublista": elemento,
                }
            })

    def _push_translated_texts_to_queue(self, cuadros_delimitadores, textos_traducidos):
        if self.traduccion_queue is None:
            return
        for idx, ((x, y, w, h), texto_traducido) in enumerate(zip(cuadros_delimitadores, textos_traducidos)):
            region = self.ultimas_regiones[idx] if idx < len(self.ultimas_regiones) else None
            elemento = {
                "Índice": idx,
                "Coordenadas": [[x, y], [x + w, y + h]],
                "Texto": texto_traducido,
                "Estilo": self.ultimo_estilos_texto[idx] if idx < len(self.ultimo_estilos_texto) else "dialogo",
            }
            if region is not None:
                elemento.update({
                    "Tipo": region.kind,
                    "Confianza": round(float(region.confidence), 4),
                    "Fuente máscara": region.metadata.get("mask_source", ""),
                })
            if idx < len(self.ultimas_asignaciones_hablante):
                speaker = self.ultimas_asignaciones_hablante[idx]
                elemento.update({
                    "Hablante": speaker.get("speaker_id", "unknown"),
                    "Confianza hablante": round(float(speaker.get("confidence") or 0.0), 4),
                })
            self.traduccion_queue.put({
                "agregar_a_sublista": {
                    "clave_lista": "Traducción",
                    "pagina": self.indice_imagen + 1,
                    "clave_sublista": "Globos de texto",
                    "elemento_sublista": elemento,
                }
            })

    def incrustar_textos(self, imagen_limpia, cuadros_delimitadores, textos):
        textos_limpios = [self.normalizar_texto_ocr(texto) for texto in textos]
        self.ultimos_textos_originales = textos_limpios
        textos_traducidos = self.traducir_textos(textos_limpios)
        self.ultimos_textos_traducidos = textos_traducidos
        self._push_original_texts_to_queue(cuadros_delimitadores, textos_limpios)
        self._push_translated_texts_to_queue(cuadros_delimitadores, textos_traducidos)
        clip_masks = [region.local_mask() for region in self.ultimas_regiones] if self.ultimas_regiones and len(self.ultimas_regiones) == len(cuadros_delimitadores) else None
        return self.text_renderer.render(
            imagen_limpia,
            cuadros_delimitadores,
            textos_traducidos,
            text_styles=self.ultimo_estilos_texto,
            clip_masks=clip_masks,
        )
