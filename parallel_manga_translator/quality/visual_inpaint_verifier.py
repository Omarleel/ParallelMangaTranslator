from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualInpaintIssue:
    """Resultado de una comprobación visual del inpainting."""

    name: str
    score: float
    threshold: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(float(self.score), 4),
            "threshold": round(float(self.threshold), 4),
            "passed": bool(self.passed),
            "details": self.details,
        }


@dataclass(frozen=True)
class VisualInpaintReport:
    """Informe agregado para decidir si aceptar o reintentar una limpieza."""

    passed: bool
    score: float
    issues: List[VisualInpaintIssue]
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def failed_checks(self) -> List[str]:
        return [issue.name for issue in self.issues if not issue.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "score": round(float(self.score), 4),
            "failed_checks": self.failed_checks,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": {key: round(float(value), 4) for key, value in self.metrics.items()},
        }


class VisualInpaintVerifier:
    """
    Verificador heurístico para detectar artefactos típicos al borrar texto en manga.

    Está pensado para ejecutarse por región y ser barato: usa solo OpenCV/Numpy. Las
    métricas son comparativas, no un clasificador perfecto; por eso devuelven un score
    y permiten escoger el mejor candidato cuando todos fallan.
    """

    def __init__(
        self,
        *,
        accept_score: float = 1.0,
        halo_threshold: float = 1.15,
        ink_threshold: float = 1.10,
        texture_threshold: float = 1.00,
        flat_patch_threshold: float = 0.72,
        edge_threshold: float = 1.00,
    ) -> None:
        self.accept_score = float(accept_score)
        self.halo_threshold = float(halo_threshold)
        self.ink_threshold = float(ink_threshold)
        self.texture_threshold = float(texture_threshold)
        self.flat_patch_threshold = float(flat_patch_threshold)
        self.edge_threshold = float(edge_threshold)

    def evaluate(
        self,
        before: np.ndarray,
        after: np.ndarray,
        clean_mask: np.ndarray,
        *,
        context_mask: Optional[np.ndarray] = None,
    ) -> VisualInpaintReport:
        mask = self._binary_mask(clean_mask, before.shape[:2])
        if cv2.countNonZero(mask) == 0:
            return VisualInpaintReport(True, 0.0, [], {})

        before_bgr = self._ensure_bgr(before)
        after_bgr = self._ensure_bgr(after)
        context = self._binary_mask(context_mask, before.shape[:2]) if context_mask is not None else None
        if context is not None and cv2.countNonZero(context) == 0:
            context = None

        sample_mask = self._background_sample_mask(mask, context)
        if cv2.countNonZero(sample_mask) == 0:
            sample_mask = self._outer_ring(mask, before.shape[:2], inner=7, outer=21, context_mask=None)
        if cv2.countNonZero(sample_mask) == 0:
            # Sin contexto fiable, devolvemos una aceptación débil en vez de forzar reintentos.
            return VisualInpaintReport(True, 0.0, [], {"sample_pixels": 0.0})

        after_luma = self._luma(after_bgr)
        before_luma = self._luma(before_bgr)
        sample_luma = after_luma[sample_mask > 0].astype(np.float32)
        bg_median = float(np.median(sample_luma))
        bg_std = float(np.std(sample_luma))
        bg_lap = self._texture_energy(after_bgr, sample_mask)
        mask_lap = self._texture_energy(after_bgr, mask)
        mask_luma = after_luma[mask > 0].astype(np.float32)
        mask_std = float(np.std(mask_luma)) if mask_luma.size else 0.0

        issues: List[VisualInpaintIssue] = []
        metrics: Dict[str, float] = {
            "sample_pixels": float(cv2.countNonZero(sample_mask)),
            "mask_pixels": float(cv2.countNonZero(mask)),
            "background_luma_median": bg_median,
            "background_luma_std": bg_std,
            "background_texture": bg_lap,
            "patch_texture": mask_lap,
            "patch_luma_std": mask_std,
        }

        halo_score, halo_details = self._halo_score(after_luma, mask, sample_mask, bg_median, bg_std, context)
        issues.append(self._issue("halo", halo_score, self.halo_threshold, halo_details))
        metrics["halo_score"] = halo_score

        ink_score, ink_details = self._ink_residue_score(after_luma, mask, sample_mask, bg_median, bg_std)
        issues.append(self._issue("ink_residue", ink_score, self.ink_threshold, ink_details))
        metrics["ink_residue_score"] = ink_score

        texture_score, texture_details = self._texture_mismatch_score(after_luma, mask, sample_mask, bg_std, bg_lap, mask_lap)
        issues.append(self._issue("texture_mismatch", texture_score, self.texture_threshold, texture_details))
        metrics["texture_mismatch_score"] = texture_score

        flat_score, flat_details = self._flat_patch_score(bg_lap, mask_lap, bg_std, mask_std)
        issues.append(self._issue("flat_patch", flat_score, self.flat_patch_threshold, flat_details))
        metrics["flat_patch_score"] = flat_score

        edge_score, edge_details = self._bubble_edge_damage_score(before_luma, after_luma, mask, context)
        issues.append(self._issue("bubble_edge_damage", edge_score, self.edge_threshold, edge_details))
        metrics["bubble_edge_damage_score"] = edge_score

        # Score agregado: max(normalizado) para que un fallo visible fuerce reintento.
        normalized_scores = [issue.score / max(issue.threshold, 1e-6) for issue in issues]
        aggregate = float(max(normalized_scores, default=0.0))
        passed = aggregate <= self.accept_score and all(issue.passed for issue in issues)
        return VisualInpaintReport(passed, aggregate, issues, metrics)

    def _issue(self, name: str, score: float, threshold: float, details: Dict[str, Any]) -> VisualInpaintIssue:
        return VisualInpaintIssue(name=name, score=float(score), threshold=float(threshold), passed=float(score) <= float(threshold), details=details)

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return image[:, :, :3]
        return image

    @staticmethod
    def _binary_mask(mask: Optional[np.ndarray], shape: Sequence[int]) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        if mask is None or not getattr(mask, "size", 0):
            return np.zeros((height, width), dtype=np.uint8)
        out = mask
        if out.ndim == 3:
            out = out[:, :, 0]
        if out.shape[:2] != (height, width):
            out = cv2.resize(out, (width, height), interpolation=cv2.INTER_NEAREST)
        return (out > 0).astype(np.uint8) * 255

    @staticmethod
    def _luma(image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image.astype(np.float32))
        return b * 0.114 + g * 0.587 + r * 0.299

    @staticmethod
    def _kernel(radius: int) -> np.ndarray:
        size = max(3, int(radius) * 2 + 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _outer_ring(
        self,
        mask: np.ndarray,
        shape: Sequence[int],
        *,
        inner: int,
        outer: int,
        context_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        outer_mask = cv2.dilate(mask, self._kernel(outer), iterations=1)
        inner_mask = cv2.dilate(mask, self._kernel(inner), iterations=1)
        ring = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
        if context_mask is not None:
            ring = cv2.bitwise_and(ring, context_mask)
        return ring

    def _background_sample_mask(self, mask: np.ndarray, context_mask: Optional[np.ndarray]) -> np.ndarray:
        # Fondo inmediato pero separado de la tinta borrada; dentro del globo si existe máscara segura.
        ring = self._outer_ring(mask, mask.shape, inner=9, outer=28, context_mask=context_mask)
        if cv2.countNonZero(ring) >= 24:
            return ring
        if context_mask is None:
            return ring
        excluded = cv2.dilate(mask, self._kernel(9), iterations=1)
        sample = cv2.bitwise_and(context_mask, cv2.bitwise_not(excluded))
        return sample

    def _halo_score(
        self,
        after_luma: np.ndarray,
        mask: np.ndarray,
        sample_mask: np.ndarray,
        bg_median: float,
        bg_std: float,
        context_mask: Optional[np.ndarray],
    ) -> tuple[float, Dict[str, Any]]:
        halo_band = self._outer_ring(mask, mask.shape, inner=1, outer=5, context_mask=context_mask)
        if cv2.countNonZero(halo_band) < 12:
            return 0.0, {"pixels": int(cv2.countNonZero(halo_band))}
        halo_values = after_luma[halo_band > 0].astype(np.float32)
        sample_values = after_luma[sample_mask > 0].astype(np.float32)
        halo_mean_delta = float(abs(np.mean(halo_values) - bg_median))
        sample_mean_delta = float(abs(np.mean(sample_values) - bg_median))
        halo_std = float(np.std(halo_values))
        # La banda alrededor del texto debe parecerse al fondo: penaliza diferencia media y variación anómala.
        contrast_score = max(0.0, halo_mean_delta - sample_mean_delta) / max(8.0, bg_std + 4.0)
        std_score = max(0.0, halo_std - bg_std * 1.35) / max(10.0, bg_std + 6.0)
        score = float(contrast_score + 0.45 * std_score)
        return score, {
            "pixels": int(cv2.countNonZero(halo_band)),
            "mean_delta": round(halo_mean_delta, 3),
            "std": round(halo_std, 3),
        }

    def _ink_residue_score(
        self,
        after_luma: np.ndarray,
        mask: np.ndarray,
        sample_mask: np.ndarray,
        bg_median: float,
        bg_std: float,
    ) -> tuple[float, Dict[str, Any]]:
        values = after_luma[mask > 0].astype(np.float32)
        if values.size < 8:
            return 0.0, {"pixels": int(values.size)}
        # Restos de tinta: píxeles mucho más oscuros/claros que el fondo local o demasiado contrastados.
        dark_margin = max(32.0, bg_std * 2.0)
        light_margin = max(42.0, bg_std * 2.4)
        dark_ratio = float(np.mean(values < (bg_median - dark_margin)))
        light_ratio = float(np.mean(values > (bg_median + light_margin)))
        mad = float(np.mean(np.abs(values - bg_median)))
        contrast_score = max(0.0, mad - max(6.0, bg_std * 0.65)) / max(18.0, bg_std + 8.0)
        ratio_score = max(dark_ratio, light_ratio * 0.75) * 3.0
        score = float(max(contrast_score, ratio_score))
        return score, {
            "pixels": int(values.size),
            "dark_ratio": round(dark_ratio, 4),
            "light_ratio": round(light_ratio, 4),
            "mean_abs_delta": round(mad, 3),
        }

    @staticmethod
    def _texture_energy(image_or_luma: np.ndarray, mask: np.ndarray) -> float:
        if cv2.countNonZero(mask) == 0:
            return 0.0
        if image_or_luma.ndim == 3:
            luma = VisualInpaintVerifier._luma(image_or_luma)
        else:
            luma = image_or_luma.astype(np.float32)
        lap = cv2.Laplacian(luma, cv2.CV_32F, ksize=3)
        values = np.abs(lap[mask > 0])
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, 75))

    def _texture_mismatch_score(
        self,
        after_luma: np.ndarray,
        mask: np.ndarray,
        sample_mask: np.ndarray,
        bg_std: float,
        bg_lap: float,
        mask_lap: float,
    ) -> tuple[float, Dict[str, Any]]:
        mask_values = after_luma[mask > 0].astype(np.float32)
        sample_values = after_luma[sample_mask > 0].astype(np.float32)
        if mask_values.size < 8 or sample_values.size < 8:
            return 0.0, {"patch_pixels": int(mask_values.size), "sample_pixels": int(sample_values.size)}
        patch_std = float(np.std(mask_values))
        sample_std = float(np.std(sample_values))
        # Ratio logarítmico: detecta tanto trama demasiado lisa como ruido/trama inventada.
        lap_delta = abs(np.log((mask_lap + 1.0) / (bg_lap + 1.0)))
        std_delta = abs(np.log((patch_std + 1.0) / (sample_std + 1.0)))
        score = float((lap_delta + 0.65 * std_delta) / 1.65)
        return score, {
            "patch_std": round(patch_std, 3),
            "sample_std": round(sample_std, 3),
            "patch_texture": round(mask_lap, 3),
            "sample_texture": round(bg_lap, 3),
        }

    @staticmethod
    def _flat_patch_score(bg_lap: float, mask_lap: float, bg_std: float, mask_std: float) -> tuple[float, Dict[str, Any]]:
        if bg_lap < 5.0 and bg_std < 9.0:
            return 0.0, {
                "background_texture": round(bg_lap, 3),
                "patch_texture": round(mask_lap, 3),
                "background_std": round(bg_std, 3),
                "patch_std": round(mask_std, 3),
            }
        texture_drop = max(0.0, bg_lap - mask_lap) / max(bg_lap, 1.0)
        std_drop = max(0.0, bg_std - mask_std) / max(bg_std, 1.0)
        texture_weight = min(1.5, max(0.0, bg_lap / 12.0))
        score = float((0.72 * texture_drop + 0.28 * std_drop) * texture_weight)
        return score, {
            "background_texture": round(bg_lap, 3),
            "patch_texture": round(mask_lap, 3),
            "background_std": round(bg_std, 3),
            "patch_std": round(mask_std, 3),
        }

    def _bubble_edge_damage_score(
        self,
        before_luma: np.ndarray,
        after_luma: np.ndarray,
        mask: np.ndarray,
        context_mask: Optional[np.ndarray],
    ) -> tuple[float, Dict[str, Any]]:
        if context_mask is None or cv2.countNonZero(context_mask) == 0:
            return 0.0, {"pixels": 0, "reason": "no_context_mask"}
        eroded = cv2.erode(context_mask, self._kernel(3), iterations=1)
        edge_band = cv2.bitwise_and(context_mask, cv2.bitwise_not(eroded))
        protected_mask = cv2.dilate(mask, self._kernel(3), iterations=1)
        edge_band = cv2.bitwise_and(edge_band, cv2.bitwise_not(protected_mask))
        edge_pixels = cv2.countNonZero(edge_band)
        if edge_pixels < 12:
            return 0.0, {"pixels": int(edge_pixels)}
        diff = np.abs(after_luma - before_luma)[edge_band > 0].astype(np.float32)
        mean_diff = float(np.mean(diff))
        changed_ratio = float(np.mean(diff > 18.0))
        score = float(max(mean_diff / 14.0, changed_ratio * 3.2))
        return score, {
            "pixels": int(edge_pixels),
            "mean_diff": round(mean_diff, 3),
            "changed_ratio": round(changed_ratio, 4),
        }
