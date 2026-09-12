"""Clasificación semántica (y refinamiento opcional) de regiones ya localizadas.

Divide el trabajo como lo divide la arquitectura: el detector dice **dónde**, el OCR
dice **qué pone**, y el VLM dice **qué es** —diálogo, pensamiento, narración u
onomatopeya— mirando la página entera, que es el contexto que ni el detector ni el OCR
tienen al trabajar por recortes.

El refinamiento de transcripción viene apagado a propósito. La transcripción actual
(MangaOCR/Paddle) mide CER 0.0015 en el banco: sustituirla por lo que lea un modelo
generalista sobre una página reescalada es muy fácil que empeore. Cuando se activa, solo
rellena huecos: nunca pisa una transcripción que el OCR sí produjo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.vision.annotated_page import PageAnnotator
from parallel_manga_translator.vision.groq_vision_client import NullVisionClient, VisionClientPort

logger = get_logger(__name__)

#: Tipos que el VLM puede devolver. Cualquier otra cosa se descarta en vez de propagarse.
SEMANTIC_KINDS = ("dialogue", "thought", "narration", "sfx")

#: Cómo aterriza cada tipo semántico en el `kind` que ya entiende el pipeline.
KIND_TO_REGION_KIND = {
    "dialogue": "dialogue",
    "thought": "dialogue",
    "narration": "narration",
    "sfx": "sfx",
}

PROMPT = """Analiza esta página de cómic. Las cajas numeradas marcan bloques de texto ya localizados.

Para CADA número visible:
1. Transcribe el texto exacto del bloque, respetando saltos de línea.
2. Clasifícalo en uno de: dialogue, thought, narration, sfx.
   - dialogue: habla dentro de un globo normal.
   - thought: pensamiento (globo de nube, borde discontinuo o burbujas).
   - narration: cajas rectangulares de narración, rótulos y letreros.
   - sfx: onomatopeyas y efectos dibujados sobre el arte.

No inventes números que no veas ni añadas bloques nuevos. No des coordenadas.
Responde solo con JSON: {"regions": [{"id": 1, "type": "dialogue", "text": "..."}]}"""


@dataclass(frozen=True)
class RegionSemantics:
    """Lo que el VLM aporta sobre una región."""

    kind: str
    text: str = ""


class VlmRegionSemantics:
    """Implementa `RegionSemanticsPort` con un cliente de visión inyectado."""

    def __init__(
        self,
        client: VisionClientPort | None = None,
        *,
        annotator: PageAnnotator | None = None,
        prompt: str = PROMPT,
        max_crops: int = 6,
        refine_transcription: bool = False,
        min_regions: int = 1,
    ) -> None:
        self.client = client if client is not None else NullVisionClient()
        self.annotator = annotator if annotator is not None else PageAnnotator()
        self.prompt = prompt
        self.max_crops = max(0, int(max_crops))
        self.refine_transcription = bool(refine_transcription)
        self.min_regions = max(1, int(min_regions))

    @property
    def enabled(self) -> bool:
        return not isinstance(self.client, NullVisionClient)

    def refine(
        self,
        image: np.ndarray,
        regions: Sequence[TextRegion],
        transcriptions: Sequence[str],
    ) -> Mapping[int, Mapping[str, Any]]:
        """Devuelve, por `region_id`, el tipo y el texto que propone el VLM."""
        if not self.enabled or len(regions) < self.min_regions:
            return {}
        if image is None or getattr(image, "size", 0) == 0:
            return {}

        crop_ids = self._crops_worth_sending(regions, transcriptions)
        page = self.annotator.build(image, regions, crop_ids=crop_ids)
        if not page.region_ids:
            return {}

        images: List[np.ndarray] = [page.image]
        prompt = self.prompt
        if page.crops:
            images.extend(crop for _, crop in page.crops)
            enviados = ", ".join(str(region_id) for region_id, _ in page.crops)
            prompt = f"{prompt}\n\nTras la página van recortes ampliados de los bloques: {enviados}."

        try:
            raw = self.client.analyze(images, prompt)
        except Exception as exc:  # pragma: no cover - depende del proveedor
            # Una caída del VLM no puede tumbar la página: es refinamiento, no la
            # fuente de verdad. El pipeline sigue con lo que ya tenía.
            logger.warning("El VLM no pudo clasificar la página (%s); se conserva la clasificación del detector.", exc)
            return {}

        parsed = self._parse(raw, set(page.region_ids))
        logger.info("VLM: %s de %s bloques clasificados.", len(parsed), len(page.region_ids))
        return parsed

    # -- selección de recortes --------------------------------------------------

    def _crops_worth_sending(self, regions: Sequence[TextRegion], transcriptions: Sequence[str]) -> List[int]:
        """Recorta solo lo difícil: bloques sin transcripción o fuera de globo.

        Mandar un recorte por región multiplicaría el coste sin ganar nada en los globos
        normales, que el modelo ya lee bien en la página completa.
        """
        if self.max_crops <= 0:
            return []
        candidates: List[tuple[float, int]] = []
        for position, region in enumerate(regions):
            region_id = int(region.metadata.get("region_id", position + 1))
            texto = str(transcriptions[position]) if position < len(transcriptions) else ""
            sin_texto = not texto.strip()
            fuera_de_globo = region.metadata.get("structural_kind") == "out_of_bubble" or region.kind in {"sfx", "free_text"}
            if not sin_texto and not fuera_de_globo:
                continue
            # Prioridad: primero lo que no tiene texto, luego lo de fuera de globo.
            candidates.append((0.0 if sin_texto else 1.0, region_id))
        candidates.sort()
        return [region_id for _, region_id in candidates[: self.max_crops]]

    # -- parseo -----------------------------------------------------------------

    def _parse(self, raw: str, valid_ids: set[int]) -> Dict[int, Dict[str, Any]]:
        payload = self._loads(raw)
        if not isinstance(payload, Mapping):
            return {}
        rows = payload.get("regions")
        if not isinstance(rows, Sequence):
            return {}

        result: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                region_id = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            # Un id que no se dibujó en la página es una alucinación: se descarta.
            if region_id not in valid_ids:
                continue
            kind = str(row.get("type") or "").strip().lower()
            if kind not in SEMANTIC_KINDS:
                continue
            entry: Dict[str, Any] = {"kind": kind, "region_kind": KIND_TO_REGION_KIND[kind]}
            if self.refine_transcription:
                text = str(row.get("text") or "").strip()
                if text:
                    entry["text"] = text
            result[region_id] = entry
        return result

    @staticmethod
    def _loads(raw: str) -> Any:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Algunos modelos envuelven el JSON en ```json ... ``` pese a pedir json_object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def apply_semantics(
    regions: Sequence[TextRegion],
    transcriptions: List[str],
    semantics: Mapping[int, Mapping[str, Any]],
    *,
    refine_transcription: bool = False,
) -> int:
    """Vuelca lo que dijo el VLM sobre las regiones. Devuelve cuántas cambiaron.

    Está fuera de la clase porque es una función pura sobre el dominio: así el pipeline
    puede aplicarla sin depender del cliente de visión.
    """
    if not semantics:
        return 0
    touched = 0
    for position, region in enumerate(regions):
        region_id = int(region.metadata.get("region_id", position + 1))
        entry = semantics.get(region_id)
        if not entry:
            continue
        kind = str(entry.get("kind") or "")
        if kind:
            region.metadata["vlm_kind"] = kind
            region.metadata["kind_before_vlm"] = region.kind
            region.kind = str(entry.get("region_kind") or region.kind)
            touched += 1
        if refine_transcription:
            text = str(entry.get("text") or "").strip()
            # Solo rellena huecos: la transcripción del OCR especializado manda.
            if text and position < len(transcriptions) and not str(transcriptions[position]).strip():
                transcriptions[position] = text
                region.metadata["vlm_filled_transcription"] = True
    return touched
