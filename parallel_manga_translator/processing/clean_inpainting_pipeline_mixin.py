from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from parallel_manga_translator.processing.bubble_fill_policy import BubbleFillPolicy, strategy_honored
from parallel_manga_translator.processing.inpainter_runner import InpainterRunner
from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.page_context import PageContext
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.processing.visual_inpaint_debug import DEPURACION_APAGADA, VisualInpaintDebugWriter

logger = get_logger(__name__)
Detection = Tuple[Sequence[Sequence[float]], str, float]


class CleanInpaintingPipelineMixin:
    """Orquestación del proceso de limpieza e inpainting."""

    #: Tipos de región que se limpian rellenando el interior del globo.
    BUBBLE_KINDS = frozenset({"dialogue", "narration", "unknown"})

    def regiones_a_limpiar(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> Tuple[List[TextRegion], List[TextRegion]]:
        """Separa las regiones que realmente se borran: globos y texto libre/SFX.

        Es la única definición de "esto se borra" y por eso es pública: el arnés de
        evaluación la usa para medir la limpieza sobre exactamente las mismas regiones
        que el pipeline decide limpiar. Medir sobre todas las regiones contaría como
        fallo el arte que se conserva a propósito (onomatopeyas en modo ``keep``).
        """
        globos = [
            region for region in regiones or []
            if region.kind in self.BUBBLE_KINDS and self._region_matches_source_language(region)
        ]
        libres = [
            region for region in regiones or []
            if region.kind not in self.BUBBLE_KINDS
            and self._region_matches_source_language(region)
            and self._should_clean_non_bubble_region(region, imagen)
        ]
        return globos, libres


    def limpiar_manga(self, ctx: PageContext) -> None:
        """Detecta las regiones con texto y borra su tinta. Deja todo en el contexto.

        El escritor de depuración se crea aquí, para **esta** página, y se pasa a los tres
        métodos que lo usan. Antes eran cuatro atributos del limpiador que dos métodos del
        puerto empujaban antes de cada página y trece `getattr` leían después.
        """
        imagen = ctx.imagen
        debug = (
            VisualInpaintDebugWriter(
                output_root=ctx.debug_root,
                page_index=ctx.indice_pagina,
                filename=ctx.nombre_archivo,
            )
            if self.visual_inpaint_debug
            else DEPURACION_APAGADA
        )
        # El detector tiene su propio contexto de depuración, y solo corre aquí dentro:
        # acotarlo a esta llamada es más estrecho que hacerlo desde el orquestador.
        self.bubble_detector.set_debug_page_context(
            ctx.indice_pagina,
            source_filename=ctx.archivo_origen or None,
            output_filename=ctx.nombre_archivo or None,
        )
        try:
            # Flujo YOLO: primero detectar todos los globos con el segmentador entrenado.
            # El OCR global se ejecuta después solo para asociar pistas, onomatopeyas y texto libre.
            regiones_primarias = self.bubble_detector.detect_primary_bubble_regions(imagen)
            resultados = self.obtener_cuadros_delimitadores(imagen)
            regiones = self.bubble_detector.build_regions_from_bubbles_and_text(imagen, regiones_primarias, resultados)
        finally:
            self.bubble_detector.clear_debug_page_context()
        regiones = self._filter_regions_by_source_language(regiones)
        regiones = self._filter_regions_by_specialized_ocr_guard(imagen, regiones)
        regiones = self.mask_strategy.attach_clean_masks(imagen, regiones)

        # Esta es la máscara de limpieza/tinta, no la máscara completa de globo.
        # La máscara de globo se conserva en region.mask como zona segura para OCR/render.
        mascara_capa = BubbleDetector.compose_clean_mask(regiones, imagen.shape) if regiones else np.zeros(imagen.shape[:2], dtype=np.uint8)

        ctx.mascara_capa = mascara_capa
        ctx.imagen_limpia = self._clean_with_regions(imagen, mascara_capa, resultados, regiones, debug)
        ctx.regiones = list(regiones)

    


    def _clean_with_regions(
        self,
        imagen: np.ndarray,
        mascara_capa: np.ndarray,
        resultados,
        regiones: Sequence[TextRegion],
        debug: VisualInpaintDebugWriter,
    ) -> np.ndarray:
        if not regiones:
            return imagen.copy()

        # El modo YOLO mantiene dos máscaras distintas: region.mask es la zona segura
        # del globo; region.clean_mask es la tinta/texto original que se borra.
        imagen_base = imagen.copy()
        # Texto libre y onomatopeyas se limpian con inpainting, no con relleno plano de globo.
        # Si el usuario eligió conservar onomatopeyas, las regiones SFX se dejan intactas
        # para no borrar arte original ni reinsertarlo como fuente plana.
        bubble_regions, sfx_regions = self.regiones_a_limpiar(imagen, regiones)

        if self.bubble_fill and self.inpaint_mode in {"auto", "fast", "bubble_only", "quality", "sfx"}:
            imagen_base = self._fill_bubble_interiors(imagen_base, bubble_regions, debug)

        if self.inpaint_mode == "bubble_only":
            return imagen_base

        imagen_base = self._clean_free_text_regions(
            imagen_base, sfx_regions, debug, debug_index_offset=len(bubble_regions)
        )

        if self.inpaint_mode == "quality" and not self.bubble_fill:
            res_impainting = self.inpainter_runner.ejecutar_inpainting(imagen, mascara_capa, resultados)
            return self.convertir_a_imagen_limpia(res_impainting, imagen)

        return imagen_base



    @staticmethod
    def _dominant_fill_color(region_img: np.ndarray, local_mask: np.ndarray, exclude_mask: Optional[np.ndarray] = None):
        """Estima el color de fondo de una región segura."""
        if region_img.size == 0 or local_mask is None or getattr(local_mask, "size", 0) == 0:
            return (255, 255, 255)

        sample_mask = BubbleFillPolicy.background_sample_mask(local_mask, exclude_mask)
        if sample_mask.size == 0 or cv2.countNonZero(sample_mask) == 0:
            return (255, 255, 255)

        pixels = region_img[sample_mask > 0]
        if pixels.size == 0:
            return (255, 255, 255)

        median = np.median(pixels.reshape(-1, 3), axis=0)
        return tuple(int(min(255, max(0, round(float(c))))) for c in median.tolist())



    def _apply_bubble_cleaning_with_visual_verifier(
        self,
        imagen: np.ndarray,
        clean_mask: np.ndarray,
        safe_mask: np.ndarray,
        fill_color,
        *,
        fill_strategy: str,
        background_variation: float,
        variation_threshold: float,
        sigma: float,
        debug: VisualInpaintDebugWriter,
        debug_region_index: Optional[int] = None,
    ) -> tuple[np.ndarray, str, Optional[object], List[dict], str]:
        should_use_inpaint = fill_strategy == "inpaint" or (
            fill_strategy == "auto" and background_variation >= variation_threshold
        )
        initial_candidate = str(getattr(self, "inpaint_model", "auto") or "auto") if should_use_inpaint else "solid"
        initial_candidate = self.fill_policy.normalize_inpaint_candidate(initial_candidate)
        if initial_candidate == "auto":
            initial_candidate = self.fill_policy.normalize_inpaint_candidate(
                self.fill_policy.auto_inpaint_candidate(imagen, background_variation, variation_threshold)
            )
        if initial_candidate == "solid" and should_use_inpaint:
            initial_candidate = "opencv-tela"

        verifier = getattr(self, "visual_inpaint_verifier", None)
        # Sobre fondo con textura ningún relleno domina: medido en 181 regiones, cada
        # candidato es el mejor en 38-50 de ellas, y quedarse con el primero que aprueba
        # deja un 24 % de score sobre la mesa. Ahí se prueban todos y gana el mejor; sobre
        # fondo plano no compensa multiplicar por cuatro el coste de GPU.
        explorar_todos = bool(getattr(self, "visual_inpaint_best_of_textured", False)) and (
            background_variation >= variation_threshold
        )
        attempts: List[dict] = []
        best_image: Optional[np.ndarray] = None
        best_method = "visual_verifier_no_candidate"
        best_candidate = "unknown"
        best_report = None
        best_score = float("inf")

        for candidate in self.fill_policy.visual_retry_candidates(initial_candidate):
            candidate = self.fill_policy.normalize_inpaint_candidate(candidate)
            candidate_image, method = self.inpainter_runner.apply_visual_inpaint_candidate(
                imagen, clean_mask, safe_mask, fill_color, candidate, sigma=sigma
            )
            if candidate_image is None:
                attempts.append({"candidate": candidate, "method": method, "accepted": False, "skipped": True})
                continue

            if verifier is None:
                return candidate_image, method, None, attempts, candidate

            report = verifier.evaluate(imagen, candidate_image, clean_mask, context_mask=safe_mask)
            attempt = {
                "candidate": candidate,
                "method": method,
                "accepted": bool(report.passed),
                "score": round(float(report.score), 4),
                "failed_checks": report.failed_checks,
            }
            debug_crop = debug.write_crop(
                candidate_image,
                clean_mask,
                safe_mask,
                debug_region_index,
                f"attempt_{len(attempts) + 1:02d}_{candidate}",
            )
            if debug_crop:
                attempt["debug_crop"] = debug_crop
            attempts.append(attempt)
            if float(report.score) < best_score:
                best_score = float(report.score)
                best_image = candidate_image
                best_method = method
                best_candidate = candidate
                best_report = report
            if report.passed and not explorar_todos:
                return candidate_image, method, report, attempts, candidate

        if best_image is not None:
            sufijo = "best_of_candidates" if bool(getattr(best_report, "passed", False)) else "best_failed_visual_score"
            return best_image, f"{best_method}:{sufijo}", best_report, attempts, best_candidate

        fallback = InpainterRunner.apply_solid_fill(imagen, clean_mask, fill_color, sigma=sigma)
        if verifier is not None:
            best_report = verifier.evaluate(imagen, fallback, clean_mask, context_mask=safe_mask)
        fallback_attempt = {
            "candidate": "solid",
            "method": "solid_color:last_resort",
            "accepted": bool(getattr(best_report, "passed", True)),
            "score": round(float(getattr(best_report, "score", 0.0)), 4),
            "failed_checks": getattr(best_report, "failed_checks", []),
        }
        debug_crop = debug.write_crop(
            fallback,
            clean_mask,
            safe_mask,
            debug_region_index,
            f"attempt_{len(attempts) + 1:02d}_solid_last_resort",
        )
        if debug_crop:
            fallback_attempt["debug_crop"] = debug_crop
        attempts.append(fallback_attempt)
        return fallback, "solid_color:last_resort", best_report, attempts, "solid"

    def _fill_bubble_interiors(
        self, imagen: np.ndarray, regiones: Sequence[TextRegion], debug: VisualInpaintDebugWriter
    ) -> np.ndarray:
        salida = imagen.copy()
        for region_index, region in enumerate(regiones):
            safe_mask = self.mask_strategy.safe_bubble_mask(region.mask, salida.shape, self.bubble_fill_edge_margin)
            if cv2.countNonZero(safe_mask) == 0:
                continue

            clean_mask = getattr(region, "clean_mask", None)
            if clean_mask is None or getattr(clean_mask, "size", 0) == 0:
                clean_mask, _source = self.mask_strategy.build_clean_mask_for_region(salida, region)
                region.clean_mask = self.mask_strategy.binary_mask(clean_mask, salida.shape)
            else:
                clean_mask = self.mask_strategy.binary_mask(clean_mask, salida.shape)

            if cv2.countNonZero(clean_mask) == 0:
                continue

            salida = self._clean_region_with_masks(salida, region, region_index, clean_mask, safe_mask, debug)
        return salida


    @staticmethod
    def _free_text_context_mask(clean_mask: np.ndarray, image_shape) -> np.ndarray:
        """Anillo de fondo alrededor de la tinta a borrar, para texto libre y SFX.

        Un globo aporta su interior como zona de referencia; el texto libre no tiene
        ninguna. Sin este anillo, el color de relleno, la medida de variación del fondo
        y el verificador visual se quedan sin muestra y cualquier candidato parece
        igual de bueno. El anillo se mantiene ancho porque el verificador muestrea una
        banda de hasta 28 px alrededor de la máscara.
        """
        ink = (clean_mask > 0).astype(np.uint8) * 255
        points = cv2.findNonZero(ink)
        if points is None:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        _x, _y, width, height = cv2.boundingRect(points)
        ring = max(32, min(72, int(round(min(width, height) * 0.45))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
        return cv2.dilate(ink, kernel, iterations=1)

    def _clean_free_text_regions(
        self,
        imagen: np.ndarray,
        regiones: Sequence[TextRegion],
        debug: VisualInpaintDebugWriter,
        *,
        debug_index_offset: int = 0,
    ) -> np.ndarray:
        """Limpia texto libre y onomatopeyas región a región, no de una pasada.

        Antes se componía una única máscara con todas estas regiones y se inpaintaba
        la página entera de golpe. Con manchas grandes y separadas eso arrasa el fondo
        —``cv2.inpaint`` difunde a lo largo de toda la máscara— y además no pasaba
        nunca por el verificador visual, así que un resultado malo se aceptaba igual.
        Aquí cada región usa su propio recorte, sus propios candidatos y su propia
        verificación, exactamente igual que los interiores de globo.
        """
        salida = imagen
        for offset, region in enumerate(regiones or []):
            clean_mask = self.mask_strategy.binary_mask(getattr(region, "clean_mask", None), salida.shape)
            if cv2.countNonZero(clean_mask) == 0:
                continue
            context_mask = self._free_text_context_mask(clean_mask, salida.shape)
            salida = self._clean_region_with_masks(
                salida, region, debug_index_offset + offset, clean_mask, context_mask, debug
            )
        return salida

    def _clean_region_with_masks(
        self,
        salida: np.ndarray,
        region: TextRegion,
        region_index: int,
        clean_mask: np.ndarray,
        safe_mask: np.ndarray,
        debug: VisualInpaintDebugWriter,
    ) -> np.ndarray:
        """Borra la tinta de una región y verifica el resultado.

        ``clean_mask`` es la tinta que se borra; ``safe_mask`` es la zona de referencia
        de la que se muestrea el fondo (interior del globo, o anillo alrededor del texto
        libre). Nunca se pintan los píxeles de ``safe_mask`` que no estén en
        ``clean_mask``.
        """
        fill_color = self._dominant_fill_color(salida, safe_mask, clean_mask)
        background_variation = self.fill_policy.background_variation_score(salida, safe_mask, clean_mask)
        variation_threshold = float(getattr(self, "bubble_fill_background_std_threshold", 18.0) or 18.0)
        fill_strategy = self.fill_policy.normalized_fill_strategy()

        metadata = getattr(region, "metadata", None)
        if isinstance(metadata, dict):
            metadata["fill_color_source"] = "safe_region_minus_clean_mask"
            metadata["fill_color_bgr"] = tuple(int(c) for c in fill_color)
            metadata["background_variation_score"] = round(float(background_variation), 3)
            metadata["background_variation_threshold"] = float(variation_threshold)
            metadata["bubble_fill_strategy"] = fill_strategy

        sigma = float(getattr(self, "bubble_fill_feather", 1.0) or 1.0)
        if not str((region.metadata or {}).get("clean_mask_source", "")).endswith("opt_in"):
            sigma = min(sigma, 0.65)

        if bool(getattr(self, "visual_inpaint_verifier_enabled", False)):
            debug_before_image = salida.copy() if debug.enabled else None
            salida_verificada, method, report, attempts, chosen_candidate = self._apply_bubble_cleaning_with_visual_verifier(
                salida,
                clean_mask,
                safe_mask,
                fill_color,
                fill_strategy=fill_strategy,
                background_variation=background_variation,
                variation_threshold=variation_threshold,
                sigma=sigma,
                debug=debug,
                debug_region_index=region_index,
            )
            salida = salida_verificada
            if isinstance(metadata, dict):
                metadata["bubble_fill_method"] = method
                # El verificador puede sustituir el candidato que pidió la estrategia.
                # Se deja constancia por región para que sea auditable en el JSON.
                metadata["bubble_fill_strategy_honored"] = strategy_honored(
                    requested=fill_strategy, method=method
                )
                metadata["visual_inpaint_candidate"] = chosen_candidate
                metadata["visual_inpaint_retries"] = max(0, len([a for a in attempts if not a.get("skipped")]) - 1)
                if report is not None:
                    metadata["visual_inpaint_passed"] = bool(report.passed)
                    metadata["visual_inpaint_score"] = round(float(report.score), 4)
                    metadata["visual_inpaint_failed_checks"] = report.failed_checks
                    if bool(getattr(self, "visual_inpaint_debug", False)):
                        metadata["visual_inpaint_report"] = report.to_dict()
                if bool(getattr(self, "visual_inpaint_debug", False)):
                    metadata["visual_inpaint_attempts"] = attempts
                    debug_metadata = debug.write_region_summary(
                        region_index=region_index,
                        region=region,
                        before_image=debug_before_image,
                        after_image=salida,
                        clean_mask=clean_mask,
                        safe_mask=safe_mask,
                        fill_color=fill_color,
                        fill_strategy=fill_strategy,
                        method=method,
                        chosen_candidate=chosen_candidate,
                        report=report,
                        attempts=attempts,
                    )
                    metadata.update(debug_metadata)
            return salida

        should_use_configured_inpaint = fill_strategy == "inpaint" or (
            fill_strategy == "auto" and background_variation >= variation_threshold
        )
        if should_use_configured_inpaint:
            salida_inpaint, method = self.inpainter_runner.run_configured_inpaint_on_mask(salida, clean_mask, safe_mask)
            if method.startswith("configured_inpaint"):
                salida = salida_inpaint
                if isinstance(metadata, dict):
                    metadata["bubble_fill_method"] = "configured_inpaint"
                    metadata["bubble_fill_strategy_honored"] = strategy_honored(
                        requested=fill_strategy, method="configured_inpaint"
                    )
                    metadata["bubble_fill_inpaint_model"] = str(getattr(self, "inpaint_model", ""))
                return salida
            if isinstance(metadata, dict):
                metadata["bubble_fill_inpaint_fallback"] = method

        salida = InpainterRunner.apply_solid_fill(salida, clean_mask, fill_color, sigma=sigma)
        if isinstance(metadata, dict):
            metadata["bubble_fill_method"] = "solid_color"
            metadata["bubble_fill_strategy_honored"] = strategy_honored(
                requested=fill_strategy, method="solid_color"
            )
        return salida



    def convertir_a_imagen_limpia(self, res_impainting: np.ndarray, imagen: np.ndarray) -> np.ndarray:
        pil_image_camuflada_limpieza = Image.fromarray(cv2.cvtColor(res_impainting, cv2.COLOR_BGR2RGB))
        pil_image_limpieza = Image.new("RGB", (imagen.shape[1], imagen.shape[0]))
        pil_image_limpieza.paste(pil_image_camuflada_limpieza, (0, 0))
        imagen_limpia = np.asarray(pil_image_limpieza)
        return cv2.cvtColor(imagen_limpia, cv2.COLOR_RGB2BGR)