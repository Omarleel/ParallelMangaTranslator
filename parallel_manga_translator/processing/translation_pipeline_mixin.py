from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


from parallel_manga_translator.infrastructure.logging_config import get_logger

Box = Tuple[int, int, int, int]
logger = get_logger(__name__)


class TranslationPipelineMixin:
    """Traducción de regiones y reglas de onomatopeyas."""

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

        metadata = getattr(region, "metadata", {}) or {}
        if metadata.get("free_text_onomatopoeia") or metadata.get("onomatopoeia"):
            metadata["free_text_onomatopoeia_keep"] = True
            return True

        candidatos = [texto, getattr(region, "source_text_hint", "")]
        for candidato in candidatos:
            if self.onomatopoeia_manager.is_free_text_onomatopoeia(candidato, self.idioma_entrada):
                if hasattr(region, "metadata"):
                    match = self.onomatopoeia_manager.similar_semantic_key(candidato, self.idioma_entrada)
                    if match:
                        key, score, source = match
                        region.metadata["free_text_onomatopoeia_keep"] = True
                        region.metadata["free_text_onomatopoeia"] = True
                        region.metadata["onomatopoeia"] = True
                        region.metadata["onomatopoeia_key"] = key
                        region.metadata["free_text_onomatopoeia_similarity"] = round(float(score), 4)
                        region.metadata["free_text_onomatopoeia_source"] = source
                return True
        return False

    def _onomatopoeia_keep_requested(self) -> bool:
        return (
            self.onomatopoeia_mode in {"keep", "original", "none", "off"}
            or not getattr(self, "translate_onomatopoeia", True)
        )

    def _should_keep_original_onomatopoeia(self, indice: int, texto: str) -> bool:
        region = self.ultimas_regiones[indice] if self.ultimas_regiones and indice < len(self.ultimas_regiones) else None
        # Una onomatopeya dentro de un globo es contenido de diálogo/reacción:
        # se limpia, se transcribe y se traduce incluso si el modo global conserva
        # SFX externos.
        if region is not None and region.kind in {"dialogue", "narration", "unknown"}:
            return False
        if not self._onomatopoeia_keep_requested():
            return False
        if region is not None and region.kind in {"sfx", "onomatopoeia"}:
            return True
        if region is not None and region.kind == "free_text" and self._es_onomatopeya_de_texto_libre(indice, texto):
            return True
        return self.onomatopoeia_manager.is_onomatopoeia(texto, self.idioma_entrada)

    def _clasificar_estilos_texto(self, textos: Sequence[str]) -> List[str]:
        # Clasificación final conservadora:
        # - diálogo normal: sólo diccionario exacto/normalizado,
        # - free_text/SFX: puede usar similitud o heurística previa de candidato,
        # - la heurística nunca convierte un diálogo normal en onomatopeya.
        estilos = [
            "onomatopeya" if self.onomatopoeia_manager.is_onomatopoeia(texto, self.idioma_entrada) else "dialogo"
            for texto in textos
        ]
        if self.ultimas_regiones and len(self.ultimas_regiones) == len(estilos):
            for i, region in enumerate(self.ultimas_regiones):
                if region.kind in {"sfx", "onomatopoeia"} or region.kind == "free_text" and self._es_onomatopeya_de_texto_libre(i, textos[i]):
                    estilos[i] = "onomatopeya"
                elif region.kind == "narration":
                    estilos[i] = "narracion"
                elif region.kind == "dialogue":
                    # Diálogo queda como diálogo salvo coincidencia exacta de diccionario.
                    estilos[i] = "onomatopeya" if self.onomatopoeia_manager.is_onomatopoeia(textos[i], self.idioma_entrada) else "dialogo"
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
            if self._should_keep_original_onomatopoeia(indice, texto):
                parciales.append(texto)
                continue
            if self.onomatopoeia_mode in {"keep", "original", "none", "off"}:
                parciales.append(None)
                continue
            if self._es_onomatopeya_de_texto_libre(indice, texto):
                parciales.append(texto)
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

    def resolver_textos_para_render(
        self,
        textos_limpios: Sequence[str],
        textos_traducidos: Sequence[str],
    ) -> List[str]:
        """Decide qué texto se dibuja realmente en cada región ya traducida.

        Queda vacío lo que el filtro de idioma de origen descartó y las onomatopeyas
        que deben conservar el arte original. La retraducción de la UI reutiliza esta
        misma regla para producir la misma imagen que habría producido el pipeline.
        """
        return [
            "" if (idx < len(self.ultimos_source_language_flags) and not self.ultimos_source_language_flags[idx])
            else ("" if self._should_keep_original_onomatopoeia(idx, original) else traducido)
            for idx, (original, traducido) in enumerate(zip(textos_limpios, textos_traducidos))
        ]

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
        source_language_flags = self._source_language_flags_for_texts(textos_limpios)
        self.ultimos_source_language_flags = source_language_flags
        self.ultimo_estilos_texto = self._clasificar_estilos_texto(textos_limpios)
        self.ultimo_estilos_texto = [
            estilo if source_language_flags[idx] else "omitido_idioma_origen"
            for idx, estilo in enumerate(self.ultimo_estilos_texto)
        ]
        base_metadata = self._region_metadata_for_translation(textos_limpios)

        keep_onomatopoeia_flags = [
            self._should_keep_original_onomatopoeia(i, texto) if source_language_flags[i] else False
            for i, texto in enumerate(textos_limpios)
        ]

        self.ultimas_asignaciones_hablante = []
        metadata_con_hablantes = list(base_metadata)
        if self.metodo_traduccion == "LLM":
            # No gastamos tokens de memoria de personajes en onomatopeyas conservadas ni en textos ajenos al idioma de origen.
            memory_texts = ["" if (keep or not source_ok) else texto for keep, source_ok, texto in zip(keep_onomatopoeia_flags, source_language_flags, textos_limpios)]
            self.ultimas_asignaciones_hablante = self.translator_manager.analyze_character_memory(
                memory_texts,
                page_index=self.indice_imagen,
                region_metadata=base_metadata,
                contexto_previo=list(self.historial_contexto),
            )
            for idx, keep in enumerate(keep_onomatopoeia_flags):
                if keep:
                    assignment = {
                        "text_id": idx,
                        "speaker_id": "sfx",
                        "confidence": 0.95,
                        "is_narration": False,
                        "evidence": "onomatopeya_conservada_sin_llm",
                    }
                    if idx < len(self.ultimas_asignaciones_hablante):
                        self.ultimas_asignaciones_hablante[idx] = assignment
                    else:
                        self.ultimas_asignaciones_hablante.append(assignment)
            metadata_con_hablantes = self._metadata_with_speaker_assignments(base_metadata, self.ultimas_asignaciones_hablante)

        traducciones_onomatopeyas = self._traducir_onomatopeyas_con_diccionario(textos_limpios)
        indices_por_traducir = [
            i for i, traduccion in enumerate(traducciones_onomatopeyas)
            if source_language_flags[i] and traduccion is None
        ]

        textos_traducidos_brutos = [texto if source_language_flags[i] else "" for i, texto in enumerate(textos_limpios)]

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
            if source_language_flags[idx] and traduccion is not None:
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

        return [texto if source_language_flags[idx] else "" for idx, texto in enumerate(textos_traducidos_limpios)]
