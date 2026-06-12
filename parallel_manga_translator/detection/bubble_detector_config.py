from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.config.runtime_config import get_active_config


@dataclass(frozen=True)
class BubbleSplitSettings:
    enabled: bool
    min_ocr_groups: int
    min_gap_px: int
    gap_ratio: float
    cluster_min_gap_px: int
    cluster_gap_ratio: float
    group_pad_x: float
    group_pad_y: float
    group_min_pad: int

    @classmethod
    def from_quality_config(cls, quality: QualityConfig) -> "BubbleSplitSettings":
        return cls(
            enabled=quality.split_merged_bubbles,
            min_ocr_groups=quality.bubble_split_min_ocr_groups,
            min_gap_px=quality.bubble_split_min_gap_px,
            gap_ratio=quality.bubble_split_gap_ratio,
            cluster_min_gap_px=quality.bubble_split_cluster_min_gap_px,
            cluster_gap_ratio=quality.bubble_split_cluster_gap_ratio,
            group_pad_x=quality.bubble_split_pad_x,
            group_pad_y=quality.bubble_split_pad_y,
            group_min_pad=quality.bubble_split_min_pad,
        )


@dataclass(frozen=True)
class OcrGroupMergeSettings:
    x_overlap: float
    y_gap_ratio: float
    cjk_y_overlap: float
    cjk_x_gap_ratio: float
    cjk_columns: bool
    line_y_overlap: float
    line_x_gap_ratio: float
    line_horizontal_only: bool

    @classmethod
    def from_quality_config(cls, quality: QualityConfig) -> "OcrGroupMergeSettings":
        return cls(
            x_overlap=quality.ocr_group_merge_x_overlap,
            y_gap_ratio=quality.ocr_group_merge_y_gap_ratio,
            cjk_y_overlap=quality.ocr_group_merge_cjk_y_overlap,
            cjk_x_gap_ratio=quality.ocr_group_merge_cjk_x_gap_ratio,
            cjk_columns=quality.ocr_group_merge_cjk_columns,
            line_y_overlap=quality.ocr_group_merge_line_y_overlap,
            line_x_gap_ratio=quality.ocr_group_merge_line_x_gap_ratio,
            line_horizontal_only=quality.ocr_group_merge_line_horizontal_only,
        )


@dataclass(frozen=True)
class FreeTextSettings:
    max_area_ratio: float
    hard_max_area_ratio: float
    max_width_ratio: float
    max_height_ratio: float
    min_confidence: float
    large_min_confidence: float
    gap_recovery: bool
    gap_max_px: int
    gap_min_y_overlap: float
    gap_min_density: float
    gap_max_density: float

    @classmethod
    def from_quality_config(cls, quality: QualityConfig) -> "FreeTextSettings":
        return cls(
            max_area_ratio=quality.free_text_max_area_ratio,
            hard_max_area_ratio=quality.free_text_hard_max_area_ratio,
            max_width_ratio=quality.free_text_max_width_ratio,
            max_height_ratio=quality.free_text_max_height_ratio,
            min_confidence=quality.free_text_min_confidence,
            large_min_confidence=quality.free_text_large_min_confidence,
            gap_recovery=quality.free_text_gap_recovery,
            gap_max_px=quality.free_text_gap_max_px,
            gap_min_y_overlap=quality.free_text_gap_min_y_overlap,
            gap_min_density=quality.free_text_gap_min_ink_density,
            gap_max_density=quality.free_text_gap_max_ink_density,
        )


@dataclass(frozen=True)
class MergeDebugSettings:
    enabled: bool
    pair_limit: int
    directory: Path

    @classmethod
    def from_quality_config(cls, quality: QualityConfig, processing: ProcessingConfig) -> "MergeDebugSettings":
        default_debug_dir = Path(processing.ruta_carpeta_salida) / "debug_globos"
        debug_dir_raw = str(quality.bubble_merge_debug_dir or "").strip()
        return cls(
            enabled=quality.bubble_merge_debug,
            pair_limit=quality.bubble_merge_debug_pair_limit,
            directory=Path(debug_dir_raw or default_debug_dir),
        )


@dataclass(frozen=True)
class BubbleDetectorSettings:
    enabled: bool
    split: BubbleSplitSettings
    ocr_merge: OcrGroupMergeSettings
    free_text: FreeTextSettings
    merge_debug: MergeDebugSettings

    @classmethod
    def from_config(
        cls,
        quality: QualityConfig | None = None,
        processing: ProcessingConfig | None = None,
    ) -> "BubbleDetectorSettings":
        active = get_active_config() if quality is None or processing is None else None
        quality = quality or active.quality  # type: ignore[union-attr]
        processing = processing or active.processing  # type: ignore[union-attr]
        return cls(
            enabled=quality.bubble_detection,
            split=BubbleSplitSettings.from_quality_config(quality),
            ocr_merge=OcrGroupMergeSettings.from_quality_config(quality),
            free_text=FreeTextSettings.from_quality_config(quality),
            merge_debug=MergeDebugSettings.from_quality_config(quality, processing),
        )
