from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from parallel_manga_translator.config.app_config import (
    ApplicationConfig,
    CharacterMemoryConfig,
    ExportConfig,
    LlmConfig,
    LoggingConfig,
    OcrConfig,
    OnomatopoeiaConfig,
    ProcessingConfig,
    QualityConfig,
    TranslationConfig,
)
from parallel_manga_translator.config.constants import MODELOS_INPAINT
from parallel_manga_translator.config.environment import bool_value, float_value, int_value


class ConfigManager:
    """Carga configuración funcional exclusivamente desde YAML/JSON.

    Reglas:
    - `config.yaml` contiene motores, flags, rutas, umbrales y opciones funcionales.
    - `.env` queda reservado para secretos (`GROQ_API_KEY`, `DEEPL_API_KEY`, etc.).
    - No se leen variables de entorno para configuración funcional.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = Path(config_path or "config.yaml")
        self.data = self._load_file(self.config_path)

    @staticmethod
    def _load_yaml_fallback(path: Path) -> Dict[str, Any]:
        root: Dict[str, Any] = {}
        current: Optional[str] = None
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            if not raw_line.startswith((" ", "\t")) and raw_line.rstrip().endswith(":"):
                current = raw_line.strip()[:-1]
                root.setdefault(current, {})
                continue
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"\'')
            parsed = ConfigManager._parse_scalar(value)
            if current and raw_line.startswith((" ", "\t")):
                root.setdefault(current, {})[key] = parsed
            else:
                root[key] = parsed
        return root

    @staticmethod
    def _parse_scalar(value: str) -> Any:
        lower = value.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    @classmethod
    def _load_file(cls, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            if path.suffix.lower() == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
            try:
                import yaml  # type: ignore

                with path.open("r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                return loaded if isinstance(loaded, dict) else {}
            except Exception:
                return cls._load_yaml_fallback(path)
        except Exception:
            return {}

    def _section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(name, {}) if isinstance(self.data, Mapping) else {}
        return value if isinstance(value, Mapping) else {}

    def build_application_config(self) -> ApplicationConfig:
        method = str(self._section("translation").get("method", "LLM"))
        processing = self._build_processing_config(method)
        translation = self._build_translation_config(project_dir=processing.ruta_carpeta_entrada)
        return ApplicationConfig(
            translation=translation,
            processing=processing,
            ocr=self._build_ocr_config(),
            quality=self._build_quality_config(),
            onomatopoeia=self._build_onomatopoeia_config(),
            character_memory=self._build_character_memory_config(),
            export=self._build_export_config(),
            logging=self._build_logging_config(),
        )

    def _build_translation_config(self, project_dir: str) -> TranslationConfig:
        translation_section = self._section("translation")
        llm_section = self._section("llm")
        model_default = MODELOS_INPAINT[1] if len(MODELOS_INPAINT) > 1 else MODELOS_INPAINT[0]
        return TranslationConfig(
            idioma_entrada=str(translation_section.get("source_language", "Japonés")),
            idioma_salida=str(translation_section.get("target_language", "Español")),
            metodo_traduccion=str(translation_section.get("method", "LLM")),
            modelo_inpaint=str(translation_section.get("inpaint_model", model_default)),
            lore_manga=str(translation_section.get("lore", "")),
            glossary_path=str(translation_section.get("glossary", "")),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            deepl_api_key=os.getenv("DEEPL_API_KEY", ""),
            traditional_provider=str(translation_section.get("traditional_provider", "auto")),
            project_dir=project_dir,
            llm=LlmConfig(
                provider=str(llm_section.get("provider", "groq")),
                model=str(llm_section.get("model", "llama-3.3-70b-versatile")),
                strict_json_schema=bool_value(llm_section.get("strict_json_schema", True), True),
                seed=int_value(llm_section.get("seed", 7), 7),
                max_retries=max(1, int_value(llm_section.get("max_retries", 3), 3)),
            ),
        )

    def _build_ocr_config(self) -> OcrConfig:
        ocr_section = self._section("ocr")
        processing_section = self._section("processing")
        return OcrConfig(
            detection_engine=str(ocr_section.get("detection_engine", "auto")),
            transcription_engine=str(ocr_section.get("transcription_engine")),
            gpu=bool_value(ocr_section.get("gpu", False), False),
            paddle_subprocess=str(ocr_section.get("paddle_subprocess", "auto")),
            fast_mode=bool_value(ocr_section.get("fast_mode", processing_section.get("fast_mode", False)), False),
        )

    def _build_processing_config(self, translation_method: str) -> ProcessingConfig:
        default_parallel = translation_method != "LLM"
        processing_section = self._section("processing")
        max_workers_raw = processing_section.get("max_workers")
        max_workers = None if max_workers_raw in {None, "", "null", "None"} else max(1, int_value(max_workers_raw, 1))
        return ProcessingConfig(
            ruta_carpeta_entrada=str(processing_section.get("input_dir", "dataset")),
            batch_size=int_value(processing_section.get("batch_size", 8), 8),
            usar_paralelismo=bool_value(processing_section.get("parallel", default_parallel), default_parallel),
            cache=bool_value(processing_section.get("cache", True), True),
            cache_dir=str(processing_section.get("cache_dir", ".cache")),
            max_workers=max_workers,
        )

    def _build_quality_config(self) -> QualityConfig:
        q = self._section("quality")

        # Los pesos del detector YOLO se configuran únicamente con bubble_model_*.
        bubble_model_path = str(q.get("bubble_model_path", "")).strip()
        bubble_model_repo = str(q.get("bubble_model_repo", "huyvux3005/manga109-segmentation-bubble")).strip()
        bubble_model_file = str(q.get("bubble_model_file", "best.pt")).strip()

        return QualityConfig(
            bubble_detection=bool_value(q.get("bubble_detection", True), True),
            bubble_fill=bool_value(q.get("bubble_fill", True), True),
            bubble_fill_whole_interior=bool_value(q.get("bubble_fill_whole_interior", False), False),
            bubble_fill_edge_margin=int_value(q.get("bubble_fill_edge_margin", 5), 5),
            bubble_fill_text_dilate=int_value(q.get("bubble_fill_text_dilate", 2), 2),
            bubble_fill_feather=float_value(q.get("bubble_fill_feather", 1.0), 1.0),
            bubble_fill_flat_max_rectangularity=float_value(q.get("bubble_fill_flat_max_rectangularity", 0.86), 0.86),
            bubble_fill_strategy=str(q.get("bubble_fill_strategy", "inpaint")).strip().lower(),
            bubble_fill_background_std_threshold=float_value(q.get("bubble_fill_background_std_threshold", 18.0), 18.0),
            bubble_fill_inpaint_padding=int_value(q.get("bubble_fill_inpaint_padding", 18), 18),
            bubble_detector=str(q.get("bubble_detector", "yolo11-seg")),
            require_yolo=bool_value(q.get("require_yolo", True), True),
            bubble_first=bool_value(q.get("bubble_first", True), True),
            ocr_region_mode=str(q.get("ocr_region_mode", "bubble")),
            bubble_model_repo=bubble_model_repo,
            bubble_model_file=bubble_model_file,
            bubble_model_path=bubble_model_path,
            bubble_device=str(q.get("bubble_device", "")),
            bubble_confidence=float_value(q.get("bubble_confidence", 0.35), 0.35),
            bubble_img_size=int_value(q.get("bubble_img_size", 1024), 1024),
            bubble_retina_masks=bool_value(q.get("bubble_retina_masks", True), True),
            bubble_model_classes=str(q.get("bubble_model_classes", "")),
            bubble_include_labels=str(q.get("bubble_include_labels", "")),
            bubble_exclude_labels=str(q.get("bubble_exclude_labels", "ignore_art,panel,page,background")),
            bubble_max_area_ratio=float_value(q.get("bubble_max_area_ratio", 0.55), 0.55),
            inpaint_mode=str(q.get("inpaint_mode", "auto")),
            split_merged_bubbles=bool_value(q.get("split_merged_bubbles", True), True),
            bubble_split_min_ocr_groups=int_value(q.get("bubble_split_min_ocr_groups", 2), 2),
            bubble_split_min_gap_px=int_value(q.get("bubble_split_min_gap_px", 18), 18),
            bubble_split_gap_ratio=float_value(q.get("bubble_split_gap_ratio", 0.70), 0.70),
            bubble_split_cluster_min_gap_px=int_value(q.get("bubble_split_cluster_min_gap_px", 12), 12),
            bubble_split_cluster_gap_ratio=float_value(q.get("bubble_split_cluster_gap_ratio", 0.35), 0.35),
            bubble_split_pad_x=float_value(q.get("bubble_split_pad_x", 0.85), 0.85),
            bubble_split_pad_y=float_value(q.get("bubble_split_pad_y", 1.05), 1.05),
            bubble_split_min_pad=int_value(q.get("bubble_split_min_pad", 18), 18),
            ocr_group_merge_x_overlap=float_value(q.get("ocr_group_merge_x_overlap", 0.52), 0.52),
            ocr_group_merge_y_gap_ratio=float_value(q.get("ocr_group_merge_y_gap_ratio", 0.55), 0.55),
            ocr_group_merge_cjk_y_overlap=float_value(q.get("ocr_group_merge_cjk_y_overlap", 0.80), 0.80),
            ocr_group_merge_cjk_x_gap_ratio=float_value(q.get("ocr_group_merge_cjk_x_gap_ratio", 0.35), 0.35),
            ocr_group_merge_cjk_columns=bool_value(q.get("ocr_group_merge_cjk_columns", False), False),
            ocr_group_merge_line_y_overlap=float_value(q.get("ocr_group_merge_line_y_overlap", 0.72), 0.72),
            ocr_group_merge_line_x_gap_ratio=float_value(q.get("ocr_group_merge_line_x_gap_ratio", 0.20), 0.20),
            ocr_group_merge_line_horizontal_only=bool_value(q.get("ocr_group_merge_line_horizontal_only", True), True),
            free_text_max_area_ratio=float_value(q.get("free_text_max_area_ratio", 0.12), 0.12),
            free_text_hard_max_area_ratio=float_value(q.get("free_text_hard_max_area_ratio", 0.22), 0.22),
            free_text_max_width_ratio=float_value(q.get("free_text_max_width_ratio", 0.96), 0.96),
            free_text_max_height_ratio=float_value(q.get("free_text_max_height_ratio", 0.60), 0.60),
            free_text_min_confidence=float_value(q.get("free_text_min_confidence", 0.08), 0.08),
            free_text_large_min_confidence=float_value(q.get("free_text_large_min_confidence", 0.16), 0.16),
            free_text_gap_recovery=bool_value(q.get("free_text_gap_recovery", True), True),
            free_text_gap_max_px=int_value(q.get("free_text_gap_max_px", 220), 220),
            free_text_gap_min_y_overlap=float_value(q.get("free_text_gap_min_y_overlap", 0.45), 0.45),
            free_text_gap_min_ink_density=float_value(q.get("free_text_gap_min_ink_density", 0.025), 0.025),
            free_text_gap_max_ink_density=float_value(q.get("free_text_gap_max_ink_density", 0.90), 0.90),
            fine_text_detection=bool_value(q.get("fine_text_detection", True), True),
            fine_text_mask_dilate=int_value(q.get("fine_text_mask_dilate", 2), 2),
            ink_mask_refinement=bool_value(q.get("ink_mask_refinement", True), True),
            ink_mask_min_component_area=int_value(q.get("ink_mask_min_component_area", 3), 3),
            ink_mask_component_anchor_overlap=float_value(q.get("ink_mask_component_anchor_overlap", 0.03), 0.03),
            ink_mask_component_anchor_max_gap_ratio=float_value(q.get("ink_mask_component_anchor_max_gap_ratio", 0.45), 0.45),
            panel_aware_reading_order=bool_value(q.get("panel_aware_reading_order", True), True),
            panel_detection_min_area_ratio=float_value(q.get("panel_detection_min_area_ratio", 0.015), 0.015),
            panel_detection_max_area_ratio=float_value(q.get("panel_detection_max_area_ratio", 0.96), 0.96),
            panel_detection_gutter_px=int_value(q.get("panel_detection_gutter_px", 10), 10),
            typography_smart_wrap=bool_value(q.get("typography_smart_wrap", True), True),
            typography_hyphenation=bool_value(q.get("typography_hyphenation", True), True),
            typography_balance_lines=bool_value(q.get("typography_balance_lines", True), True),
            typography_line_spacing_factor=float_value(q.get("typography_line_spacing_factor", 1.0), 1.0),
            bubble_merge_debug=bool_value(q.get("bubble_merge_debug", False), False),
            bubble_merge_debug_dir=str(q.get("bubble_merge_debug_dir", "")),
            bubble_merge_debug_pair_limit=int_value(q.get("bubble_merge_debug_pair_limit", 160), 160),
        )

    def _build_onomatopoeia_config(self) -> OnomatopoeiaConfig:
        section = self._section("onomatopoeia")
        mode = str(section.get("mode", "translate"))
        translate = bool_value(section.get("translate", True), True)
        default_clean = translate and mode.strip().lower() not in {"keep", "original", "none", "off"}
        return OnomatopoeiaConfig(
            mode=mode,
            translate=translate,
            clean=bool_value(section.get("clean", default_clean), default_clean),
        )

    def _build_character_memory_config(self) -> CharacterMemoryConfig:
        section = self._section("character_memory")
        return CharacterMemoryConfig(
            enabled=bool_value(section.get("enabled", True), True),
            path=str(section.get("path", "")),
            max_context_pages=int_value(section.get("max_context_pages", 8), 8),
        )

    def _build_export_config(self) -> ExportConfig:
        section = self._section("export")
        return ExportConfig(
            pdf=bool_value(section.get("pdf", True), True),
            cbz=bool_value(section.get("cbz", False), False),
            skip_pdf=bool_value(section.get("skip_pdf", False), False),
        )

    def _build_logging_config(self) -> LoggingConfig:
        section = self._section("logging")
        return LoggingConfig(
            level=str(section.get("level", "INFO")),
            file=str(section.get("file", "debug.log")),
        )
