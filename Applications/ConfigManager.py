from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from Applications.AppConfig import ApplicationConfig, ProcessingConfig, TranslationConfig
from Applications.Environment import env_bool, env_int, publish_env_defaults
from Utils.Constantes import MODELOS_INPAINT


@dataclass(frozen=True)
class EnvBinding:
    """Vincula una clave YAML con una variable PMT_* usada por módulos legados."""

    env_name: str
    section: str
    key: str


ENV_BINDINGS: tuple[EnvBinding, ...] = (
    EnvBinding("PMT_CACHE", "processing", "cache"),
    EnvBinding("PMT_CACHE_DIR", "processing", "cache_dir"),
    EnvBinding("PMT_GLOSSARY", "translation", "glossary"),
    EnvBinding("PMT_OCR_ENGINE", "ocr", "engine"),
    EnvBinding("PMT_OCR_GPU", "ocr", "gpu"),
    EnvBinding("PMT_PADDLE_SUBPROCESS", "ocr", "paddle_subprocess"),
    EnvBinding("PMT_BUBBLE_DETECTION", "quality", "bubble_detection"),
    EnvBinding("PMT_BUBBLE_FILL", "quality", "bubble_fill"),
    EnvBinding("PMT_BUBBLE_FILL_WHOLE_INTERIOR", "quality", "bubble_fill_whole_interior"),
    EnvBinding("PMT_BUBBLE_FILL_EDGE_MARGIN", "quality", "bubble_fill_edge_margin"),
    EnvBinding("PMT_BUBBLE_FILL_TEXT_DILATE", "quality", "bubble_fill_text_dilate"),
    EnvBinding("PMT_BUBBLE_FILL_FEATHER", "quality", "bubble_fill_feather"),
    EnvBinding("PMT_BUBBLE_FILL_FLAT_MAX_RECTANGULARITY", "quality", "bubble_fill_flat_max_rectangularity"),
    EnvBinding("PMT_BUBBLE_FILL_SHAPE_AWARE", "quality", "bubble_fill_shape_aware"),
    EnvBinding("PMT_BUBBLE_DETECTOR", "quality", "bubble_detector"),
    EnvBinding("PMT_BUBBLE_FIRST", "quality", "bubble_first"),
    EnvBinding("PMT_OCR_REGION_MODE", "quality", "ocr_region_mode"),
    EnvBinding("PMT_BUBBLE_MODEL_REPO", "quality", "bubble_model_repo"),
    EnvBinding("PMT_BUBBLE_MODEL_FILE", "quality", "bubble_model_file"),
    EnvBinding("PMT_BUBBLE_MODEL_PATH", "quality", "bubble_model_path"),
    EnvBinding("PMT_BUBBLE_DEVICE", "quality", "bubble_device"),
    EnvBinding("PMT_REQUIRE_PROFESSIONAL_BUBBLE", "quality", "require_professional"),
    EnvBinding("PMT_BUBBLE_CONF", "quality", "bubble_confidence"),
    EnvBinding("PMT_BUBBLE_IMGSZ", "quality", "bubble_img_size"),
    EnvBinding("PMT_INPAINT_MODE", "quality", "inpaint_mode"),
    EnvBinding("PMT_SPLIT_MERGED_BUBBLES", "quality", "split_merged_bubbles"),
    EnvBinding("PMT_BUBBLE_SPLIT_MIN_OCR_GROUPS", "quality", "bubble_split_min_ocr_groups"),
    EnvBinding("PMT_BUBBLE_SPLIT_MIN_GAP_PX", "quality", "bubble_split_min_gap_px"),
    EnvBinding("PMT_BUBBLE_SPLIT_GAP_RATIO", "quality", "bubble_split_gap_ratio"),
    EnvBinding("PMT_BUBBLE_SPLIT_CLUSTER_MIN_GAP_PX", "quality", "bubble_split_cluster_min_gap_px"),
    EnvBinding("PMT_BUBBLE_SPLIT_CLUSTER_GAP_RATIO", "quality", "bubble_split_cluster_gap_ratio"),
    EnvBinding("PMT_BUBBLE_SPLIT_PAD_X", "quality", "bubble_split_pad_x"),
    EnvBinding("PMT_BUBBLE_SPLIT_PAD_Y", "quality", "bubble_split_pad_y"),
    EnvBinding("PMT_BUBBLE_SPLIT_MIN_PAD", "quality", "bubble_split_min_pad"),
    EnvBinding("PMT_OCR_GROUP_MERGE_X_OVERLAP", "quality", "ocr_group_merge_x_overlap"),
    EnvBinding("PMT_OCR_GROUP_MERGE_Y_GAP_RATIO", "quality", "ocr_group_merge_y_gap_ratio"),
    EnvBinding("PMT_OCR_GROUP_MERGE_CJK_Y_OVERLAP", "quality", "ocr_group_merge_cjk_y_overlap"),
    EnvBinding("PMT_OCR_GROUP_MERGE_CJK_X_GAP_RATIO", "quality", "ocr_group_merge_cjk_x_gap_ratio"),
    EnvBinding("PMT_OCR_GROUP_MERGE_CJK_COLUMNS", "quality", "ocr_group_merge_cjk_columns"),
    EnvBinding("PMT_OCR_GROUP_MERGE_LINE_Y_OVERLAP", "quality", "ocr_group_merge_line_y_overlap"),
    EnvBinding("PMT_OCR_GROUP_MERGE_LINE_X_GAP_RATIO", "quality", "ocr_group_merge_line_x_gap_ratio"),
    EnvBinding("PMT_OCR_GROUP_MERGE_LINE_HORIZONTAL_ONLY", "quality", "ocr_group_merge_line_horizontal_only"),
    EnvBinding("PMT_FREE_TEXT_MAX_AREA_RATIO", "quality", "free_text_max_area_ratio"),
    EnvBinding("PMT_FREE_TEXT_HARD_MAX_AREA_RATIO", "quality", "free_text_hard_max_area_ratio"),
    EnvBinding("PMT_FREE_TEXT_MAX_WIDTH_RATIO", "quality", "free_text_max_width_ratio"),
    EnvBinding("PMT_FREE_TEXT_MAX_HEIGHT_RATIO", "quality", "free_text_max_height_ratio"),
    EnvBinding("PMT_FREE_TEXT_MIN_CONFIDENCE", "quality", "free_text_min_confidence"),
    EnvBinding("PMT_FREE_TEXT_LARGE_MIN_CONFIDENCE", "quality", "free_text_large_min_confidence"),
    EnvBinding("PMT_FREE_TEXT_GAP_RECOVERY", "quality", "free_text_gap_recovery"),
    EnvBinding("PMT_FREE_TEXT_GAP_MAX_PX", "quality", "free_text_gap_max_px"),
    EnvBinding("PMT_FREE_TEXT_GAP_MIN_Y_OVERLAP", "quality", "free_text_gap_min_y_overlap"),
    EnvBinding("PMT_FREE_TEXT_GAP_MIN_INK_DENSITY", "quality", "free_text_gap_min_ink_density"),
    EnvBinding("PMT_FREE_TEXT_GAP_MAX_INK_DENSITY", "quality", "free_text_gap_max_ink_density"),
    EnvBinding("PMT_BUBBLE_MERGE_DEBUG", "quality", "bubble_merge_debug"),
    EnvBinding("PMT_BUBBLE_MERGE_DEBUG_PAIR_LIMIT", "quality", "bubble_merge_debug_pair_limit"),
    EnvBinding("PMT_BUBBLE_MERGE_DEBUG_DIR", "quality", "bubble_merge_debug_dir"),
    EnvBinding("PMT_ONOMATOPOEIA_MODE", "onomatopoeia", "mode"),
    EnvBinding("PMT_TRANSLATE_ONOMATOPOEIA", "onomatopoeia", "translate"),
    EnvBinding("PMT_EXPORT_PDF", "export", "pdf"),
    EnvBinding("PMT_EXPORT_CBZ", "export", "cbz"),
    EnvBinding("PMT_SKIP_PDF", "export", "skip_pdf"),
)


class ConfigManager:
    """Carga configuración desde YAML/JSON y variables de entorno.

    El YAML es la fuente principal de configuración no privada. Las variables de entorno
    conservan prioridad para compatibilidad, automatización y secretos.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = Path(config_path or os.getenv("PMT_CONFIG", "config.yaml"))
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
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return int(value)
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

    def _section_value(self, section_name: str, key: str, default: Any = None) -> Any:
        return self._section(section_name).get(key, default)

    def build_application_config(self) -> ApplicationConfig:
        translation = self._build_translation_config()
        processing = self._build_processing_config(translation.metodo_traduccion)
        self._publish_legacy_environment(processing)
        return ApplicationConfig(translation=translation, processing=processing)

    def _build_translation_config(self) -> TranslationConfig:
        method = os.getenv("PMT_TRANSLATION_METHOD", str(self._section_value("translation", "method", "LLM")))
        model_default = MODELOS_INPAINT[1] if len(MODELOS_INPAINT) > 1 else MODELOS_INPAINT[0]
        return TranslationConfig(
            idioma_entrada=os.getenv("PMT_SOURCE_LANG", str(self._section_value("translation", "source_language", "Japonés"))),
            idioma_salida=os.getenv("PMT_TARGET_LANG", str(self._section_value("translation", "target_language", "Español"))),
            metodo_traduccion=method,
            modelo_inpaint=os.getenv("PMT_INPAINT_MODEL", str(self._section_value("translation", "inpaint_model", model_default))),
            lore_manga=str(self._section_value("translation", "lore", "")),
            groq_api_key=os.getenv("GROQ_API_KEY", str(self._section_value("translation", "groq_api_key", ""))),
        )

    def _build_processing_config(self, translation_method: str) -> ProcessingConfig:
        default_parallel = translation_method != "LLM"
        return ProcessingConfig(
            ruta_carpeta_entrada=os.getenv("PMT_INPUT_DIR", str(self._section_value("processing", "input_dir", "Dataset"))),
            batch_size=env_int("PMT_BATCH_SIZE", int(self._section_value("processing", "batch_size", 8))),
            usar_paralelismo=env_bool("PMT_PARALLEL", bool(self._section_value("processing", "parallel", default_parallel))),
        )

    def _publish_legacy_environment(self, processing: ProcessingConfig) -> None:
        defaults = {"PMT_PROJECT_DIR": processing.ruta_carpeta_entrada}
        for binding in ENV_BINDINGS:
            defaults[binding.env_name] = self._section_value(binding.section, binding.key)
        publish_env_defaults(defaults)
