from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OcrConfig:
    """Configuración no secreta del motor OCR.

    Esta configuración pertenece a config.yaml. El .env queda reservado para secretos
    de proveedores externos, como claves privadas de APIs.
    """

    # OCR de localización: detecta bounding boxes de texto para limpieza, pistas y división de globos.
    detection_engine: str = "auto"
    # OCR de transcripción: lee el texto final dentro de cada región/globo.
    transcription_engine: str = "auto"
    gpu: bool = False
    paddle_subprocess: str = "auto"
    fast_mode: bool = False


@dataclass(frozen=True)
class LlmConfig:
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    strict_json_schema: bool = True
    seed: int = 7
    max_retries: int = 3


@dataclass(frozen=True)
class TranslationConfig:
    idioma_entrada: str
    idioma_salida: str
    metodo_traduccion: str = "Tradicional"
    modelo_inpaint: str = "lama_mpe"
    lore_manga: str = ""
    glossary_path: str = ""
    groq_api_key: str = ""
    deepl_api_key: str = ""
    traditional_provider: str = "auto"
    llm: LlmConfig = field(default_factory=LlmConfig)
    project_dir: Optional[str] = None


@dataclass(frozen=True)
class ProcessingConfig:
    ruta_carpeta_entrada: str = "dataset"
    batch_size: int = 8
    usar_paralelismo: bool = True
    # Solapa la preparación GPU de la página N+1 con OCR/traducción/render de N.
    # No crea otra copia de los modelos CUDA y, por tanto, es segura para GPUs de 16 GB.
    cpu_gpu_pipeline: bool = True
    pipeline_prefetch: int = 2
    cache: bool = True
    cache_dir: str = ".cache"
    max_workers: Optional[int] = None

    @property
    def ruta_carpeta_salida(self) -> str:
        return str(Path(self.ruta_carpeta_entrada) / "outputs")

    @property
    def ruta_carpeta_limpieza(self) -> str:
        return str(Path(self.ruta_carpeta_salida) / "limpieza")

    @property
    def ruta_carpeta_traduccion(self) -> str:
        return str(Path(self.ruta_carpeta_salida) / "traduccion")


@dataclass(frozen=True)
class QualityConfig:
    bubble_detection: bool = True
    bubble_fill: bool = True
    bubble_fill_whole_interior: bool = False
    bubble_fill_edge_margin: int = 5
    bubble_fill_text_dilate: int = 2
    bubble_ink_without_ocr_max_ratio: float = 0.35
    bubble_fill_feather: float = 1.0
    bubble_fill_flat_max_rectangularity: float = 0.86
    bubble_fill_strategy: str = "inpaint"
    bubble_fill_background_std_threshold: float = 18.0
    bubble_fill_inpaint_padding: int = 18
    visual_inpaint_verifier: bool = True
    visual_inpaint_retry: bool = True
    visual_inpaint_retry_models: str = "solid,opencv-tela,lama_mpe,aot"
    visual_inpaint_max_retries: int = 4
    visual_inpaint_accept_score: float = 1.0
    visual_inpaint_best_of_textured: bool = False
    visual_inpaint_debug: bool = False
    bubble_detector: str = "yolo11-seg"
    require_yolo: bool = True
    bubble_first: bool = True
    ocr_region_mode: str = "bubble"
    bubble_model_repo: str = "huyvux3005/manga109-segmentation-bubble"
    bubble_model_file: str = "best.pt"
    bubble_model_path: str = ""
    bubble_device: str = ""
    bubble_confidence: float = 0.35
    bubble_img_size: int = 1024
    bubble_retina_masks: bool = True
    bubble_model_classes: str = ""
    bubble_include_labels: str = ""
    bubble_exclude_labels: str = "ignore_art,panel,page,background"
    bubble_max_area_ratio: float = 0.55
    inpaint_mode: str = "auto"
    split_merged_bubbles: bool = True
    bubble_split_min_ocr_groups: int = 2
    bubble_split_min_gap_px: int = 18
    bubble_split_gap_ratio: float = 0.70
    bubble_split_cluster_min_gap_px: int = 12
    bubble_split_cluster_gap_ratio: float = 0.35
    bubble_split_pad_x: float = 0.85
    bubble_split_pad_y: float = 1.05
    bubble_split_min_pad: int = 18
    ocr_group_merge_x_overlap: float = 0.52
    ocr_group_merge_y_gap_ratio: float = 0.55
    ocr_group_merge_cjk_y_overlap: float = 0.80
    ocr_group_merge_cjk_x_gap_ratio: float = 0.35
    ocr_group_merge_cjk_columns: bool = False
    ocr_group_merge_line_y_overlap: float = 0.72
    ocr_group_merge_line_x_gap_ratio: float = 0.20
    ocr_group_merge_line_horizontal_only: bool = True
    free_text_max_area_ratio: float = 0.12
    free_text_hard_max_area_ratio: float = 0.22
    free_text_max_width_ratio: float = 0.96
    free_text_max_height_ratio: float = 0.60
    free_text_min_confidence: float = 0.08
    free_text_large_min_confidence: float = 0.16
    free_text_gap_recovery: bool = True
    free_text_gap_max_px: int = 220
    free_text_gap_min_y_overlap: float = 0.45
    free_text_gap_min_ink_density: float = 0.025
    free_text_gap_max_ink_density: float = 0.90

    # Precisión fina: OCR polygons -> máscara de tinta.
    fine_text_detection: bool = True
    fine_text_mask_dilate: int = 2
    ink_mask_refinement: bool = True
    ink_mask_min_component_area: int = 3
    ink_mask_component_anchor_overlap: float = 0.03
    ink_mask_component_anchor_max_gap_ratio: float = 0.45
    text_halo_growth_px: int = 10
    # Orden de lectura sensible a paneles detectados con OpenCV.
    panel_aware_reading_order: bool = True
    panel_detection_min_area_ratio: float = 0.015
    panel_detection_max_area_ratio: float = 0.96
    panel_detection_gutter_px: int = 10
    # Render tipográfico: wrapping balanceado, cortes suaves y espaciado configurable.
    typography_smart_wrap: bool = True
    typography_hyphenation: bool = True
    typography_balance_lines: bool = True
    typography_line_spacing_factor: float = 1.0
    bubble_merge_debug: bool = False
    bubble_merge_debug_dir: str = ""
    bubble_merge_debug_pair_limit: int = 160


@dataclass(frozen=True)
class OnomatopoeiaConfig:
    mode: str = "translate"
    translate: bool = True
    clean: bool = True


@dataclass(frozen=True)
class CharacterMemoryConfig:
    enabled: bool = True
    path: str = ""
    max_context_pages: int = 8


@dataclass(frozen=True)
class ExportConfig:
    pdf: bool = True
    cbz: bool = False
    skip_pdf: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = "debug.log"


@dataclass(frozen=True)
class EvaluationSettings:
    """Dataset de regresión validado a mano, no la carpeta que procesa el usuario.

    `processing.ruta_carpeta_entrada` es el manga que se está traduciendo; esto es el
    banco de pruebas contra el que se mide si un cambio mejora o empeora.
    """

    dataset_dir: str = "dataset_eval"
    iou_threshold: float = 0.50
    tolerance: float = 0.005


@dataclass(frozen=True)
class ApplicationConfig:
    translation: TranslationConfig
    processing: ProcessingConfig
    ocr: OcrConfig = OcrConfig()
    quality: QualityConfig = QualityConfig()
    onomatopoeia: OnomatopoeiaConfig = OnomatopoeiaConfig()
    character_memory: CharacterMemoryConfig = CharacterMemoryConfig()
    export: ExportConfig = ExportConfig()
    logging: LoggingConfig = LoggingConfig()
    evaluation: EvaluationSettings = EvaluationSettings()
