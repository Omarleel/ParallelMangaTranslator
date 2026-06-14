from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from parallel_manga_translator.detection.yolo_bubble_detector import YoloBubbleCandidate
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import Box, TextRegion

logger = get_logger(__name__)
BUBBLE_SPLIT_DEBUG_VERSION = "v7_bubble_onomatopoeia_translation_2026_06_11"


class BubbleTextRulesMixin:
    """Reglas de clasificación de texto, SFX y agrupación OCR."""

    @staticmethod
    def _label_is_sfx(label: str) -> bool:
        normalized = (label or "").strip().lower().replace("_", " ").replace("-", " ")
        return any(token in normalized for token in {"sfx", "sound", "effect", "onomato", "onomatopoeia", "text free"})

    @staticmethod
    def _label_is_narration(label: str) -> bool:
        normalized = (label or "").strip().lower().replace("_", " ").replace("-", " ")
        return any(token in normalized for token in {"narration", "caption", "box", "thought"})

    def _kind_from_yolo_candidate(self, candidate: YoloBubbleCandidate, fallback_sfx: bool = False) -> str:
        if fallback_sfx or self._label_is_sfx(candidate.label):
            return "sfx"
        if self._label_is_narration(candidate.label):
            return "narration"
        return "dialogue"

    def _looks_like_sfx(self, detections: Sequence) -> bool:
        text = "".join(self._text(det) for det in detections).strip()
        if self.onomatopoeia_manager.is_onomatopoeia_candidate(text, self.idioma_entrada):
            return True
        boxes = [self._to_rect(det) for det in detections]
        if not boxes:
            return False
        merged = boxes[0]
        for box in boxes[1:]:
            merged = self._union(merged, box)
        _x, _y, w, h = merged
        aspect = max(w, h) / max(1, min(w, h))
        compact_text = re.sub(r"\s+", "", text)
        compact_len = len(compact_text)
        # Dígitos/puntuación sueltos sobre ojos, botones o tramas no son SFX.
        # Las onomatopeyas reales ya se aceptaron arriba por diccionario.
        if not re.search(r"[A-Za-z\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u3400-\u9fff\uac00-\ud7af]", compact_text):
            return False

        # El fallback por forma era demasiado agresivo: una columna japonesa vertical
        # de diálogo libre (por ejemplo 「すごい演技力ね」) también tiene aspect ratio alto.
        # Por proporción sola sólo aceptamos trazos/renglones muy horizontales; los
        # SFX verticales reales deben entrar por diccionario/similitud/heurística.
        horizontalish = w >= max(1, h) * 1.35
        if not horizontalish:
            return False

        # Una línea muy horizontal y corta puede ser un SFX, pero en páginas de
        # notas los renglones japoneses largos también son muy anchos. Si contiene
        # hiragana/kanji suficientes para parecer frase, no lo clasifiques como SFX
        # solo por proporción.
        looks_sentence_like = compact_len >= 7 and bool(re.search(r"[\u3040-\u309f\u3400-\u9fff]", compact_text))
        return aspect >= 4.0 and compact_len <= 10 and not looks_sentence_like

    def _free_text_onomatopoeia_metadata(self, text: str) -> Dict[str, object]:
        """Devuelve metadata estable para textos libres que parecen SFX/onomatopeya.

        Flujo deliberado:
        1) diccionario exacto/normalizado,
        2) similitud contra el diccionario,
        3) heurística débil sólo para candidatos free_text/SFX.
        """
        text = str(text or "").strip()
        if not text:
            return {}
        match = self.onomatopoeia_manager.candidate_semantic_key(
            text,
            self.idioma_entrada,
            allow_similarity=True,
            allow_heuristic=True,
        )
        if not match:
            return {}
        key, score, source, method = match
        return {
            "free_text_onomatopoeia": True,
            "onomatopoeia": True,
            "onomatopoeia_key": key,
            "free_text_onomatopoeia_similarity": round(float(score), 4),
            "free_text_onomatopoeia_source": source,
            "free_text_onomatopoeia_method": method,
        }

    @staticmethod
    def _compact_visible_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or ""))

    @staticmethod
    def _bubble_symbol_preserve_metadata(compact: str, region_bbox: Box) -> Dict[str, object]:
        """Protege expresiones visuales/símbolos dentro de globos.

        Los detectores de texto a veces leen signos como ``!?`` como una sílaba
        katakana aislada (por ejemplo ``パ``). Si marcamos eso como onomatopeya
        o lo tratamos como diálogo normal, el limpiador borra el símbolo original.
        Esta regla sólo cubre casos muy cortos y aislados para no bloquear frases.
        """
        compact = str(compact or "").strip()
        if not compact:
            return {}
        
        area = region_bbox[2] * region_bbox[3]
        
        # 2. Si el globo es grande (ajusta este umbral según tus necesidades)
        # permitimos que el OCR falle sin que la regla lo proteja.
        if area > len(compact) * 1000: 
            return {}
        
        if len(compact) > 2:
            return {}
        
        punctuation_expression = bool(re.fullmatch(r"[!！?？⁉⁈‼…｡。・･、,\.~〜\-♪♫♥♡☆★]+", compact))
        single_katakana_hint = bool(re.fullmatch(r"[ァ-ヿ]", compact))
        if not (punctuation_expression or single_katakana_hint):
            return {}
        return {
            "visual_expression": True,
            "non_translatable_expression": True,
            "skip_cleanup_translation": True,
            "visual_expression_source": "short_bubble_symbol_or_single_katakana_hint",
            "visual_expression_hint": compact,
        }

    def _bubble_visual_expression_metadata(self, region: TextRegion) -> Dict[str, object]:
        """Materializa onomatopeyas cortas detectadas dentro de globos.

        Las onomatopeyas dentro de globos de diálogo deben seguir el flujo normal
        de limpieza, OCR especializado, traducción y render. Por eso esta
        inferencia sólo etiqueta la región para estilo/debug; no activa
        ``skip_cleanup_translation`` ni ``free_text_onomatopoeia_keep``.
        """
        if region.kind not in {"dialogue", "narration", "unknown"}:
            return {}

        metadata = getattr(region, "metadata", {}) or {}
        if metadata.get("free_text_onomatopoeia") or metadata.get("onomatopoeia"):
            return {}

        candidates = [getattr(region, "source_text_hint", "")]
        for key in ("text", "ocr_text", "source_text", "source_text_hint", "ocr_group_text"):
            value = metadata.get(key)
            if value is not None:
                candidates.append(str(value))

        for candidate in candidates:
            compact = self._compact_visible_text(candidate)
            # Sólo usamos pistas muy cortas. Una frase normal que EasyOCR leyó a
            # medias no debe saltarse por contener una sílaba tipo パ.
            if not compact or len(compact) > 4:
                continue

            preserve = self._bubble_symbol_preserve_metadata(compact, region.bbox)
            if preserve:
                return preserve

            # Una sola sílaba katakana es demasiado poco fiable dentro de globos:
            # ``!?`` puede venir de Paddle/EasyOCR como ``パ``. Evita convertirla en
            # onomatopeya interna y borrar el símbolo original. No bloquees kana
            # no katakana que el diccionario sí reconozca explícitamente.
            if len(compact) < 2 and re.fullmatch(r"[ァ-ヿ]", compact):
                continue

            result = self._free_text_onomatopoeia_metadata(compact)
            if not result:
                continue
            if result.get("free_text_onomatopoeia_method") == "heuristic":
                continue
            try:
                score = float(result.get("free_text_onomatopoeia_similarity", 0.0))
            except Exception:
                score = 0.0
            if score < 0.88:
                continue
            result.update({
                "bubble_onomatopoeia": True,
                "translate_inside_bubble": True,
                "visual_expression_source": "short_bubble_ocr_hint",
                "visual_expression_hint": compact,
            })
            return result
        return {}

    def _annotate_bubble_visual_expressions(self, regions: Sequence[TextRegion]) -> None:
        for region in regions or []:
            metadata = getattr(region, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            inferred = self._bubble_visual_expression_metadata(region)
            if inferred:
                metadata.update(inferred)

    def _region_onomatopoeia_debug_metadata(self, region: TextRegion) -> Dict[str, object]:
        metadata = getattr(region, "metadata", {}) or {}
        # No marques todo ``kind=sfx`` como onomatopeya en debug: un falso SFX por
        # proporción no debe terminar como ``is_free_text_onomatopoeia=true``. Sólo
        # reporta onomatopeya cuando hay metadata explícita de diccionario/similitud.
        if metadata.get("free_text_onomatopoeia") or metadata.get("onomatopoeia"):
            result = {
                "free_text_onomatopoeia": bool(metadata.get("free_text_onomatopoeia")),
                "onomatopoeia": bool(metadata.get("onomatopoeia")),
            }
            for key in ("onomatopoeia_key", "free_text_onomatopoeia_similarity", "free_text_onomatopoeia_source", "free_text_onomatopoeia_keep", "bubble_onomatopoeia", "translate_inside_bubble", "visual_expression", "skip_cleanup_translation", "visual_expression_source", "visual_expression_hint"):
                if key in metadata:
                    result[key] = metadata[key]
            return result

        result = self._bubble_visual_expression_metadata(region)
        if result:
            return result
        return {}

    @staticmethod
    def _free_text_signal_stats(text: str) -> Dict[str, int]:
        """Cuenta señales OCR útiles sin tratar números sueltos como texto real.

        EasyOCR/Paddle pueden devolver dígitos o símbolos sobre ojos, botones y
        tramas de ropa.  Para texto libre fuera de globos somos más estrictos que
        para los globos detectados por IA: un número aislado no debe generar una
        región que luego se limpie/traduzca.
        """
        text = str(text or "")
        kana = len(re.findall(r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff]", text))
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        hangul = len(re.findall(r"[\uac00-\ud7af]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        digits = len(re.findall(r"[0-9]", text))
        meaningful_without_digits = kana + cjk + hangul + latin
        meaningful = meaningful_without_digits + digits
        visible = len(re.sub(r"\s+", "", text))
        symbols = max(0, visible - meaningful)
        return {
            "kana": kana,
            "cjk": cjk,
            "hangul": hangul,
            "latin": latin,
            "digits": digits,
            "meaningful_without_digits": meaningful_without_digits,
            "meaningful": meaningful,
            "visible": visible,
            "symbols": symbols,
        }

    @classmethod
    def _has_meaningful_text_signal(cls, text: str) -> bool:
        """Devuelve True si el OCR contiene letras/CJK reales.

        Los dígitos aislados ya no cuentan como señal textual suficiente.  En las
        páginas de manga suelen ser falsos positivos sobre botones, ojos, fondos o
        tramas; conservarlos terminaba creando regiones de texto libre donde no
        había nada que traducir.
        """
        stats = cls._free_text_signal_stats(text)
        if stats["meaningful_without_digits"] > 0:
            return True
        return stats["digits"] >= 2 and stats["visible"] == stats["digits"]

    def _should_keep_free_text_group(
        self,
        text_box: Box,
        text_hint: str,
        confidence: float,
        image_shape,
        looks_sfx: bool,
    ) -> Tuple[bool, str]:
        img_height, img_width = image_shape[:2]
        img_area = max(1, img_height * img_width)
        bx, by, bw, bh = text_box
        box_area = max(1, bw * bh)
        area_ratio = box_area / img_area
        width_ratio = bw / max(1, img_width)
        height_ratio = bh / max(1, img_height)
        stats = self._free_text_signal_stats(text_hint)
        has_signal = self._has_meaningful_text_signal(text_hint)

        # Corte duro para el caso que estaba generando falsos positivos: varias
        # letras cercanas se fusionan en una sola columna de texto libre y el
        # limpiador termina borrando una franja enorme de la página. Una región
        # OCR vertical fuera de globos que ocupa más de media página no es una
        # unidad de texto válida para limpiar/traducir; aunque tenga caracteres
        # CJK y buena confianza, debe descartarse antes del retry de OCR.
        if height_ratio > 0.50 and bh > bw * 1.20:
            return False, "altura_vertical_imposible"

        symbols = stats["symbols"]
        visible = max(1, stats["visible"])
        symbol_ratio = symbols / visible
        meaningful_without_digits = stats["meaningful_without_digits"]
        only_digits_or_symbols = meaningful_without_digits == 0
        weak_short_signal = meaningful_without_digits <= 1 and stats["meaningful"] <= 2
        cjk_vertical_ocr_retry = (
            self.idioma_entrada in {"Japonés", "Chino", "Coreano"}
            and (stats["kana"] + stats["cjk"] + stats["hangul"]) >= 1
            and bh >= max(48, int(round(img_height * 0.035)))
            and bw >= 10
            and bh / max(1, bw) >= 2.15
            and bw / max(1, img_width) <= 0.22
            and area_ratio <= min(self.free_text_hard_max_area_ratio, 0.060)
            and symbol_ratio <= 0.55
        )

        # Nada que parezca texto y además baja confianza: probablemente ruido.
        if not has_signal and not looks_sfx and confidence < self.free_text_min_confidence:
            return False, "sin_senal_textual_y_baja_confianza"

        # Filtro específico para texto libre: no conviertas números/símbolos
        # solitarios en regiones a limpiar. Los globos reales ya están cubiertos por
        # el detector YOLO; aquí solo queremos texto huérfano confiable.
        if not looks_sfx:
            if only_digits_or_symbols:
                return False, "solo_numeros_o_simbolos_ocr_ruido"

            # EasyOCR/Paddle fallan mucho al reconocer columnas japonesas/chinas:
            # pueden detectar bien la caja, pero devolver una sola letra como texto
            # (por ejemplo ``書`` para una frase vertical completa). En esos casos
            # conservamos la caja como free_text para que el OCR especializado sobre
            # el recorte vuelva a leerla; la pista global se usará sólo como bbox,
            # no como contenido definitivo ni como señal de onomatopeya.
            if cjk_vertical_ocr_retry:
                return True, "cjk_vertical_bbox_retry_ocr"

            if weak_short_signal and confidence < max(0.35, self.free_text_min_confidence):
                return False, "senal_textual_demasiado_debil"
            if area_ratio > 0.018 and meaningful_without_digits < 3 and symbol_ratio > 0.40:
                return False, "region_grande_con_ocr_ruidoso"
            if area_ratio > 0.018 and meaningful_without_digits < 3 and confidence < 0.45:
                return False, "region_grande_con_senal_textual_debil"
            if symbol_ratio > 0.50 and meaningful_without_digits < 3 and confidence < 0.45:
                return False, "ocr_ruidoso_con_demasiados_simbolos"
            if area_ratio > 0.08 and meaningful_without_digits < 8:
                cjk_vertical_text = (
                    self.idioma_entrada in {"Japonés", "Chino", "Coreano"}
                    and (stats["kana"] + stats["cjk"] + stats["hangul"]) >= 4
                    and confidence >= 0.70
                    and bh / max(1, bw) >= 3.0
                )
                if not cjk_vertical_text:
                    return False, "free_text_grande_con_demasiado_poco_texto"

        # Límites duros para regiones gigantes; estas cajas suelen ser fondos, paneles
        # o dibujos completos detectados como texto.
        if area_ratio > self.free_text_hard_max_area_ratio:
            return False, "area_gigante"

        # Límites blandos: solo se rechazan si la caja grande no tiene señal textual
        # suficiente. Así se conservan líneas horizontales largas de páginas de notas.
        is_large = (
            area_ratio > self.free_text_max_area_ratio
            or width_ratio > self.free_text_max_width_ratio
            or height_ratio > self.free_text_max_height_ratio
        )
        if is_large and (not has_signal or confidence < self.free_text_large_min_confidence):
            return False, "region_grande_sin_senal_ocr_fiable"

        return True, "aceptado"

    def _detection_group_merge_decision(self, a: Box, b: Box) -> Dict[str, object]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        gap_x = max(0, max(bx - ax2, ax - bx2))
        gap_y = max(0, max(by - ay2, ay - by2))
        avg_h = max(1.0, (ah + bh) / 2)
        avg_w = max(1.0, (aw + bw) / 2)
        x_overlap = self._overlap_ratio_1d(ax, ax2, bx, bx2)
        y_overlap = self._overlap_ratio_1d(ay, ay2, by, by2)

        merge = False
        reason = "separados"
        max_vertical_gap = max(8.0, avg_h * self.ocr_merge_y_gap_ratio)
        max_cjk_horizontal_gap = max(6.0, avg_w * self.ocr_merge_cjk_x_gap_ratio)
        # Antes este límite usaba avg_h, lo que en texto vertical permitía saltos enormes
        # entre columnas. Debe depender del ancho de los fragmentos y, por defecto, solo
        # aplica a texto horizontal.
        max_line_gap = max(6.0, avg_w * self.ocr_merge_line_x_gap_ratio)
        a_horizontalish = aw >= ah * 0.90
        b_horizontalish = bw >= bh * 0.90
        both_horizontalish = a_horizontalish and b_horizontalish

        # Reglas conservadoras: evita unir globos/textos cercanos; solo une fragmentos
        # con solape claro y separación pequeña. Los umbrales son ajustables desde config.yaml.
        if x_overlap >= self.ocr_merge_x_overlap and gap_y <= max_vertical_gap:
            merge = True
            reason = "fragmentos_verticales_mismo_bloque"
        elif (
            self.ocr_merge_cjk_columns
            and self.idioma_entrada in {"Japonés", "Chino", "Coreano"}
            and y_overlap >= self.ocr_merge_cjk_y_overlap
            and gap_x <= max_cjk_horizontal_gap
        ):
            merge = True
            reason = "columnas_cjk_mismo_bloque"
        elif (
            (not self.ocr_merge_line_horizontal_only or both_horizontalish)
            and y_overlap >= self.ocr_merge_line_y_overlap
            and gap_x <= max_line_gap
        ):
            merge = True
            reason = "fragmentos_horizontales_misma_linea"

        return {
            "merge": merge,
            "reason": reason,
            "a": list(map(int, a)),
            "b": list(map(int, b)),
            "gap_x": int(gap_x),
            "gap_y": int(gap_y),
            "x_overlap": round(float(x_overlap), 4),
            "y_overlap": round(float(y_overlap), 4),
            "thresholds": {
                "x_overlap": self.ocr_merge_x_overlap,
                "vertical_gap": round(float(max_vertical_gap), 3),
                "cjk_y_overlap": self.ocr_merge_cjk_y_overlap,
                "cjk_horizontal_gap": round(float(max_cjk_horizontal_gap), 3),
                "line_y_overlap": self.ocr_merge_line_y_overlap,
                "line_gap": round(float(max_line_gap), 3),
                "cjk_columns_enabled": self.ocr_merge_cjk_columns,
                "line_horizontal_only": self.ocr_merge_line_horizontal_only,
            },
            "orientation": {
                "a_horizontalish": bool(a_horizontalish),
                "b_horizontalish": bool(b_horizontalish),
            },
        }

    def _should_merge_detection_groups(self, a: Box, b: Box) -> bool:
        return bool(self._detection_group_merge_decision(a, b)["merge"])

    def _group_detections(self, detections: Sequence, trace_decisions: Optional[List[Dict[str, object]]] = None) -> List[List]:
        groups: List[Tuple[Box, List]] = []
        for det_idx, det in enumerate(detections):
            try:
                rect = self._to_rect(det)
            except Exception:
                continue
            placed = False
            for i, (box, items) in enumerate(groups):
                decision = self._detection_group_merge_decision(box, rect)
                if trace_decisions is not None and len(trace_decisions) < self.merge_debug_pair_limit:
                    trace_item = dict(decision)
                    trace_item.update({
                        "stage": "asignacion_inicial",
                        "group_index": i,
                        "group_items": len(items),
                        "detection_index": det_idx,
                        "detection_text": self._text(det),
                    })
                    trace_decisions.append(trace_item)
                if decision["merge"]:
                    groups[i] = (self._union(box, rect), items + [det])
                    placed = True
                    break
            if not placed:
                groups.append((rect, [det]))

        changed = True
        pass_index = 0
        while changed:
            pass_index += 1
            changed = False
            merged: List[Tuple[Box, List]] = []
            used = [False] * len(groups)
            for i, (box, items) in enumerate(groups):
                if used[i]:
                    continue
                current_box = box
                current_items = list(items)
                used[i] = True
                for j in range(i + 1, len(groups)):
                    if used[j]:
                        continue
                    other_box, other_items = groups[j]
                    decision = self._detection_group_merge_decision(current_box, other_box)
                    if trace_decisions is not None and len(trace_decisions) < self.merge_debug_pair_limit:
                        trace_item = dict(decision)
                        trace_item.update({
                            "stage": "consolidacion",
                            "pass": pass_index,
                            "group_a": i,
                            "group_b": j,
                            "group_a_items": len(current_items),
                            "group_b_items": len(other_items),
                        })
                        trace_decisions.append(trace_item)
                    if decision["merge"]:
                        current_box = self._union(current_box, other_box)
                        current_items.extend(other_items)
                        used[j] = True
                        changed = True
                merged.append((current_box, current_items))
            groups = merged
        return [items for _box, items in groups]
