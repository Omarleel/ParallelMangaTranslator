from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from Applications.CacheManager import env_flag
from Applications.Environment import env_float, env_int


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
    def from_env(cls) -> "BubbleSplitSettings":
        return cls(
            enabled=env_flag("PMT_SPLIT_MERGED_BUBBLES", True),
            min_ocr_groups=env_int("PMT_BUBBLE_SPLIT_MIN_OCR_GROUPS", 2),
            min_gap_px=env_int("PMT_BUBBLE_SPLIT_MIN_GAP_PX", 18),
            gap_ratio=env_float("PMT_BUBBLE_SPLIT_GAP_RATIO", 0.70),
            cluster_min_gap_px=env_int("PMT_BUBBLE_SPLIT_CLUSTER_MIN_GAP_PX", 12),
            cluster_gap_ratio=env_float("PMT_BUBBLE_SPLIT_CLUSTER_GAP_RATIO", 0.35),
            group_pad_x=env_float("PMT_BUBBLE_SPLIT_PAD_X", 0.85),
            group_pad_y=env_float("PMT_BUBBLE_SPLIT_PAD_Y", 1.05),
            group_min_pad=env_int("PMT_BUBBLE_SPLIT_MIN_PAD", 18),
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
    def from_env(cls) -> "OcrGroupMergeSettings":
        return cls(
            x_overlap=env_float("PMT_OCR_GROUP_MERGE_X_OVERLAP", 0.52),
            y_gap_ratio=env_float("PMT_OCR_GROUP_MERGE_Y_GAP_RATIO", 0.55),
            cjk_y_overlap=env_float("PMT_OCR_GROUP_MERGE_CJK_Y_OVERLAP", 0.80),
            cjk_x_gap_ratio=env_float("PMT_OCR_GROUP_MERGE_CJK_X_GAP_RATIO", 0.35),
            cjk_columns=env_flag("PMT_OCR_GROUP_MERGE_CJK_COLUMNS", False),
            line_y_overlap=env_float("PMT_OCR_GROUP_MERGE_LINE_Y_OVERLAP", 0.72),
            line_x_gap_ratio=env_float("PMT_OCR_GROUP_MERGE_LINE_X_GAP_RATIO", 0.20),
            line_horizontal_only=env_flag("PMT_OCR_GROUP_MERGE_LINE_HORIZONTAL_ONLY", True),
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
    def from_env(cls) -> "FreeTextSettings":
        return cls(
            max_area_ratio=env_float("PMT_FREE_TEXT_MAX_AREA_RATIO", 0.12),
            hard_max_area_ratio=env_float("PMT_FREE_TEXT_HARD_MAX_AREA_RATIO", 0.22),
            max_width_ratio=env_float("PMT_FREE_TEXT_MAX_WIDTH_RATIO", 0.96),
            max_height_ratio=env_float("PMT_FREE_TEXT_MAX_HEIGHT_RATIO", 0.60),
            min_confidence=env_float("PMT_FREE_TEXT_MIN_CONFIDENCE", 0.08),
            large_min_confidence=env_float("PMT_FREE_TEXT_LARGE_MIN_CONFIDENCE", 0.16),
            gap_recovery=env_flag("PMT_FREE_TEXT_GAP_RECOVERY", True),
            gap_max_px=env_int("PMT_FREE_TEXT_GAP_MAX_PX", 220),
            gap_min_y_overlap=env_float("PMT_FREE_TEXT_GAP_MIN_Y_OVERLAP", 0.45),
            gap_min_density=env_float("PMT_FREE_TEXT_GAP_MIN_INK_DENSITY", 0.025),
            gap_max_density=env_float("PMT_FREE_TEXT_GAP_MAX_INK_DENSITY", 0.90),
        )


@dataclass(frozen=True)
class MergeDebugSettings:
    enabled: bool
    pair_limit: int
    directory: Path

    @classmethod
    def from_env(cls) -> "MergeDebugSettings":
        default_debug_dir = Path(os.getenv("PMT_PROJECT_DIR", "Dataset")) / "Outputs" / "DebugGlobos"
        debug_dir_raw = os.getenv("PMT_BUBBLE_MERGE_DEBUG_DIR", "").strip()
        return cls(
            enabled=env_flag("PMT_BUBBLE_MERGE_DEBUG", False),
            pair_limit=env_int("PMT_BUBBLE_MERGE_DEBUG_PAIR_LIMIT", 160),
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
    def from_env(cls) -> "BubbleDetectorSettings":
        return cls(
            enabled=env_flag("PMT_BUBBLE_DETECTION", True),
            split=BubbleSplitSettings.from_env(),
            ocr_merge=OcrGroupMergeSettings.from_env(),
            free_text=FreeTextSettings.from_env(),
            merge_debug=MergeDebugSettings.from_env(),
        )
