from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.processing.translate_manga import TranslateManga

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]


@dataclass
class RetranslatedRegion:
    """Resultado de retraducir una región ya transcrita."""

    index: int
    bbox: List[int]
    original_text: str
    translated_text: str
    rendered_text: str
    style: str
    rotation_angle: float = 0.0
    source_language_ok: bool = True


def region_is_retranslatable(region: Any) -> bool:
    """Una región sirve para retraducir si conserva su transcripción y no fue borrada."""
    if not isinstance(region, dict) or region.get("deleted"):
        return False
    return bool(str(region.get("original_text") or "").strip())


def _safe_bbox(raw: Any, width: int, height: int) -> Box:
    try:
        x, y, w, h = [int(round(float(value))) for value in list(raw)[:4]]
    except Exception as exc:  # bbox corrupto en un manifiesto antiguo
        raise ValueError(f"La región tiene un bbox inválido: {raw!r}") from exc
    x = max(0, min(x, max(0, width - 1)))
    y = max(0, min(y, max(0, height - 1)))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


class JobRetranslator:
    """Vuelve a traducir páginas ya procesadas sin repetir detección, OCR ni inpainting.

    Parte de la imagen limpia y de la transcripción guardada en el manifiesto, y delega
    toda la política de traducción (onomatopeyas, filtro de idioma de origen, memoria de
    personajes, estilos y tipografía) en `TranslateManga`, la misma clase que usa el
    pipeline. Así una retraducción produce lo que habría producido el pipeline con el
    traductor elegido, sin duplicar reglas en la capa de UI.

    Reutiliza una única instancia para todas las páginas del trabajo: el historial de
    contexto bilingüe del LLM depende de procesarlas en orden, igual que en el pipeline.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.translator = TranslateManga(
            config.translation.idioma_entrada,
            config.translation.idioma_salida,
            metodo_traduccion=config.translation.metodo_traduccion,
            groq_api_key=config.translation.groq_api_key,
            lore_manga=config.translation.lore_manga,
            ocr_config=config.ocr,
            translation_config=config.translation,
            quality_config=config.quality,
            onomatopoeia_config=config.onomatopoeia,
            character_memory_config=config.character_memory,
        )

    def retranslate_page(
        self,
        *,
        page_index: int,
        clean_path: str | Path,
        output_path: str | Path,
        regions: Sequence[Dict[str, Any]],
    ) -> List[RetranslatedRegion]:
        """Retraduce una página y reescribe su imagen traducida automática."""
        clean_image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if clean_image is None:
            raise ValueError(f"No se pudo leer la imagen limpia: {clean_path}")
        height, width = clean_image.shape[:2]

        usable = [region for region in regions if region_is_retranslatable(region)]
        if not usable:
            raise ValueError("La página no conserva transcripción que se pueda retraducir.")

        text_regions = self._build_text_regions(usable, width, height)
        originales = [
            self.translator.normalizar_texto_ocr(str(region.get("original_text") or ""))
            for region in usable
        ]

        self.translator.indice_imagen = page_index
        self.translator.ultimas_regiones = text_regions
        self.translator.ultimas_asignaciones_hablante = []
        traducidos = self.translator.traducir_textos(originales)
        textos_para_render = self.translator.resolver_textos_para_render(originales, traducidos)
        estilos = list(self.translator.ultimo_estilos_texto)
        flags = list(self.translator.ultimos_source_language_flags)
        rotaciones = [float(region.get("rotation_angle") or 0.0) for region in usable]
        cajas = [region.bbox for region in text_regions]
        layouts = [region.get("ui_layout") if isinstance(region.get("ui_layout"), dict) else None for region in usable]

        # El pipeline rotula dentro de la máscara del globo (`clip_mask`), que no se
        # persiste. Lo que sí se guardó es el `ui_layout` que salió de esa máscara: sus
        # bloques llevan el área segura y el reparto en lóbulos. Reutilizarlo hace que el
        # texto nuevo se ajuste al mismo hueco; renderizar contra el bbox entero agranda
        # la fuente y descuadra los globos partidos.
        dibujables = [
            posicion
            for posicion, texto in enumerate(textos_para_render)
            if str(texto).strip()
        ]
        imagen = self.translator.text_renderer.render_with_layouts(
            clean_image,
            [cajas[posicion] for posicion in dibujables],
            [textos_para_render[posicion] for posicion in dibujables],
            ui_layouts=[layouts[posicion] for posicion in dibujables],
            text_styles=[estilos[posicion] for posicion in dibujables],
            rotation_angles=[rotaciones[posicion] for posicion in dibujables],
        )
        self._write_image(output_path, imagen)

        resultados: List[RetranslatedRegion] = []
        for position, region in enumerate(usable):
            estilo = estilos[position] if position < len(estilos) else "dialogo"
            resultados.append(
                RetranslatedRegion(
                    index=int(region.get("index", position)),
                    bbox=list(cajas[position]),
                    original_text=originales[position],
                    translated_text=traducidos[position] if position < len(traducidos) else "",
                    rendered_text=textos_para_render[position] if position < len(textos_para_render) else "",
                    style=estilo,
                    rotation_angle=rotaciones[position],
                    source_language_ok=bool(flags[position]) if position < len(flags) else True,
                )
            )
        return resultados

    def _build_text_regions(self, regions: Sequence[Dict[str, Any]], width: int, height: int) -> List[TextRegion]:
        """Reconstruye regiones ligeras para que las reglas del pipeline vean su contexto.

        Las máscaras reales no se persisten, y aquí no hacen falta: sin recortes de
        limpieza ni OCR, `mask` solo se consulta por su forma. Se comparte una única
        máscara de página para no reservar una imagen completa por región.
        """
        shared_mask = np.zeros((height, width), dtype=np.uint8)
        text_regions: List[TextRegion] = []
        for position, region in enumerate(regions):
            bbox = _safe_bbox(region.get("bbox"), width, height)
            metadata: Dict[str, Any] = {
                "reading_order_index": position,
                "text_rotation_angle": float(region.get("rotation_angle") or 0.0),
                "text_rotation_confidence": float(region.get("rotation_confidence") or 0.0),
            }
            text_regions.append(
                TextRegion(
                    bbox=bbox,
                    text_bbox=bbox,
                    mask=shared_mask,
                    kind=str(region.get("type") or "dialogue"),
                    confidence=float(region.get("confidence") or 0.0),
                    source_text_hint=str(region.get("original_text") or ""),
                    metadata=metadata,
                )
            )
        return text_regions

    @property
    def llm_fallback_reason(self) -> str:
        """Motivo por el que una traducción LLM acabó resolviéndose con el tradicional.

        El proveedor cae al traductor tradicional en silencio (sin clave, tras agotar
        reintentos o al quedarse sin tokens). Para el pipeline es lo correcto; para quien
        pidió LLM a propósito sobre un trabajo ya terminado, no: recibiría páginas de
        otro motor sin enterarse.
        """
        provider = getattr(self.translator.translator_manager, "provider", None)
        return str(getattr(provider, "llm_fallback_reason", "") or "")

    @staticmethod
    def _write_image(output_path: str | Path, image: np.ndarray) -> None:
        # La imagen traducida es la señal de recuperación de una página lista: se
        # escribe aparte y se reemplaza de golpe para no dejar un archivo a medias.
        destino = Path(output_path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_name(f"{destino.stem}.retranslate.tmp{destino.suffix}")
        if not cv2.imwrite(str(temporal), image):
            raise ValueError(f"No se pudo escribir la página retraducida: {destino}")
        os.replace(temporal, destino)
