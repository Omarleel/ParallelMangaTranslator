from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from Applications.AppConfig import ApplicationConfig, ProcessingConfig, TranslationConfig
from Utils.Constantes import MODELOS_INPAINT


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class ConfigManager:
    """Carga configuración desde YAML/JSON + variables de entorno.

    Archivo por defecto: config.yaml. También se puede usar PMT_CONFIG=/ruta/config.yaml.
    Las variables de entorno PMT_* tienen prioridad sobre el archivo.
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
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    parsed = value
            if current and raw_line.startswith((" ", "\t")):
                root.setdefault(current, {})[key] = parsed
            else:
                root[key] = parsed
        return root

    @classmethod
    def _load_file(cls, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            if path.suffix.lower() in {".json"}:
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

    @staticmethod
    def _get(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
        return mapping.get(key, default)

    def build_application_config(self) -> ApplicationConfig:
        t = self._section("translation")
        p = self._section("processing")
        q = self._section("quality")
        o = self._section("onomatopoeia")
        e = self._section("export")
        ocr = self._section("ocr")

        metodo = os.getenv("PMT_TRANSLATION_METHOD", str(self._get(t, "method", "LLM")))
        groq_api_key = os.getenv("GROQ_API_KEY", str(self._get(t, "groq_api_key", "")))
        default_parallel = metodo != "LLM"

        modelo_default = MODELOS_INPAINT[1] if len(MODELOS_INPAINT) > 1 else MODELOS_INPAINT[0]
        translation = TranslationConfig(
            idioma_entrada=os.getenv("PMT_SOURCE_LANG", str(self._get(t, "source_language", "Japonés"))),
            idioma_salida=os.getenv("PMT_TARGET_LANG", str(self._get(t, "target_language", "Español"))),
            metodo_traduccion=metodo,
            modelo_inpaint=os.getenv("PMT_INPAINT_MODEL", str(self._get(t, "inpaint_model", modelo_default))),
            lore_manga=str(self._get(t, "lore", "")),
            groq_api_key=groq_api_key,
        )

        processing = ProcessingConfig(
            ruta_carpeta_entrada=os.getenv("PMT_INPUT_DIR", str(self._get(p, "input_dir", "Dataset"))),
            batch_size=_env_int("PMT_BATCH_SIZE", int(self._get(p, "batch_size", 8))),
            usar_paralelismo=_env_bool("PMT_PARALLEL", bool(self._get(p, "parallel", default_parallel))),
        )

        # Publica rutas/opciones para módulos que mantienen firmas antiguas.
        os.environ.setdefault("PMT_PROJECT_DIR", processing.ruta_carpeta_entrada)
        if "cache" in p and not os.getenv("PMT_CACHE"):
            os.environ["PMT_CACHE"] = str(p["cache"])
        if "cache_dir" in p and not os.getenv("PMT_CACHE_DIR"):
            os.environ["PMT_CACHE_DIR"] = str(p["cache_dir"])
        if "glossary" in t and not os.getenv("PMT_GLOSSARY"):
            os.environ["PMT_GLOSSARY"] = str(t["glossary"])
        env_defaults = {
            "PMT_OCR_ENGINE": ocr.get("engine"),
            "PMT_OCR_GPU": ocr.get("gpu"),
            "PMT_PADDLE_SUBPROCESS": ocr.get("paddle_subprocess"),
            "PMT_BUBBLE_DETECTION": q.get("bubble_detection"),
            "PMT_BUBBLE_FILL": q.get("bubble_fill"),
            "PMT_BUBBLE_FILL_WHOLE_INTERIOR": q.get("bubble_fill_whole_interior"),
            "PMT_BUBBLE_FILL_EDGE_MARGIN": q.get("bubble_fill_edge_margin"),
            "PMT_BUBBLE_FILL_TEXT_DILATE": q.get("bubble_fill_text_dilate"),
            "PMT_BUBBLE_FILL_FEATHER": q.get("bubble_fill_feather"),
            "PMT_BUBBLE_FILL_FLAT_MAX_RECTANGULARITY": q.get("bubble_fill_flat_max_rectangularity"),
            "PMT_BUBBLE_FILL_SHAPE_AWARE": q.get("bubble_fill_shape_aware"),
            "PMT_BUBBLE_DETECTOR": q.get("bubble_detector"),
            "PMT_BUBBLE_FIRST": q.get("bubble_first"),
            "PMT_OCR_REGION_MODE": q.get("ocr_region_mode"),
            "PMT_BUBBLE_MODEL_REPO": q.get("bubble_model_repo"),
            "PMT_BUBBLE_MODEL_FILE": q.get("bubble_model_file"),
            "PMT_BUBBLE_MODEL_PATH": q.get("bubble_model_path"),
            "PMT_BUBBLE_DEVICE": q.get("bubble_device"),
            "PMT_REQUIRE_PROFESSIONAL_BUBBLE": q.get("require_professional"),
            "PMT_BUBBLE_CONF": q.get("bubble_confidence"),
            "PMT_BUBBLE_IMGSZ": q.get("bubble_img_size"),
            "PMT_INPAINT_MODE": q.get("inpaint_mode"),
            "PMT_SPLIT_MERGED_BUBBLES": q.get("split_merged_bubbles"),
            "PMT_BUBBLE_SPLIT_MIN_OCR_GROUPS": q.get("bubble_split_min_ocr_groups"),
            "PMT_BUBBLE_SPLIT_MIN_GAP_PX": q.get("bubble_split_min_gap_px"),
            "PMT_BUBBLE_SPLIT_GAP_RATIO": q.get("bubble_split_gap_ratio"),
            "PMT_BUBBLE_SPLIT_CLUSTER_MIN_GAP_PX": q.get("bubble_split_cluster_min_gap_px"),
            "PMT_BUBBLE_SPLIT_CLUSTER_GAP_RATIO": q.get("bubble_split_cluster_gap_ratio"),
            "PMT_BUBBLE_SPLIT_PAD_X": q.get("bubble_split_pad_x"),
            "PMT_BUBBLE_SPLIT_PAD_Y": q.get("bubble_split_pad_y"),
            "PMT_BUBBLE_SPLIT_MIN_PAD": q.get("bubble_split_min_pad"),
            "PMT_OCR_GROUP_MERGE_X_OVERLAP": q.get("ocr_group_merge_x_overlap"),
            "PMT_OCR_GROUP_MERGE_Y_GAP_RATIO": q.get("ocr_group_merge_y_gap_ratio"),
            "PMT_OCR_GROUP_MERGE_CJK_Y_OVERLAP": q.get("ocr_group_merge_cjk_y_overlap"),
            "PMT_OCR_GROUP_MERGE_CJK_X_GAP_RATIO": q.get("ocr_group_merge_cjk_x_gap_ratio"),
            "PMT_OCR_GROUP_MERGE_CJK_COLUMNS": q.get("ocr_group_merge_cjk_columns"),
            "PMT_OCR_GROUP_MERGE_LINE_Y_OVERLAP": q.get("ocr_group_merge_line_y_overlap"),
            "PMT_OCR_GROUP_MERGE_LINE_X_GAP_RATIO": q.get("ocr_group_merge_line_x_gap_ratio"),
            "PMT_OCR_GROUP_MERGE_LINE_HORIZONTAL_ONLY": q.get("ocr_group_merge_line_horizontal_only"),
            "PMT_FREE_TEXT_MAX_AREA_RATIO": q.get("free_text_max_area_ratio"),
            "PMT_FREE_TEXT_HARD_MAX_AREA_RATIO": q.get("free_text_hard_max_area_ratio"),
            "PMT_FREE_TEXT_MAX_WIDTH_RATIO": q.get("free_text_max_width_ratio"),
            "PMT_FREE_TEXT_MAX_HEIGHT_RATIO": q.get("free_text_max_height_ratio"),
            "PMT_FREE_TEXT_MIN_CONFIDENCE": q.get("free_text_min_confidence"),
            "PMT_FREE_TEXT_LARGE_MIN_CONFIDENCE": q.get("free_text_large_min_confidence"),
            "PMT_FREE_TEXT_GAP_RECOVERY": q.get("free_text_gap_recovery"),
            "PMT_FREE_TEXT_GAP_MAX_PX": q.get("free_text_gap_max_px"),
            "PMT_FREE_TEXT_GAP_MIN_Y_OVERLAP": q.get("free_text_gap_min_y_overlap"),
            "PMT_FREE_TEXT_GAP_MIN_INK_DENSITY": q.get("free_text_gap_min_ink_density"),
            "PMT_FREE_TEXT_GAP_MAX_INK_DENSITY": q.get("free_text_gap_max_ink_density"),
            "PMT_BUBBLE_MERGE_DEBUG": q.get("bubble_merge_debug"),
            "PMT_BUBBLE_MERGE_DEBUG_PAIR_LIMIT": q.get("bubble_merge_debug_pair_limit"),
            "PMT_BUBBLE_MERGE_DEBUG_DIR": q.get("bubble_merge_debug_dir"),
            "PMT_ONOMATOPOEIA_MODE": o.get("mode"),
            "PMT_TRANSLATE_ONOMATOPOEIA": o.get("translate"),
            "PMT_EXPORT_PDF": e.get("pdf"),
            "PMT_EXPORT_CBZ": e.get("cbz"),
            "PMT_SKIP_PDF": e.get("skip_pdf"),
        }
        for key, value in env_defaults.items():
            if value is not None and os.getenv(key) is None:
                os.environ[key] = str(value)
        return ApplicationConfig(translation=translation, processing=processing)
