from pathlib import Path


def construir_ruta(base, *paths) -> str:
    return str(Path(base).joinpath(*paths))


BASE_DIR = Path(__file__).resolve().parents[2]

# CONSTANTES GENERALES
PESO_MODELOS = 7.8  # GB reservados aprox. por proceso con modelos pesados de OCR/inpainting
RUTA_ACTUAL = str(BASE_DIR)
RUTA_REMOTA = "ParallelMangaTranslator"

IDIOMAS_ENTRADA_DISPONIBLES = ["Chino", "Coreano", "Inglés", "Japonés"]
IDIOMAS_SALIDA_DISPONIBLES = ["Español", "Inglés", "Portugués", "Francés", "Italiano"]
MODELOS_INPAINT = ["opencv-tela", "lama_mpe", "lama_large_512px", "aot", "B/N"]
MODELOS_INPAINT_UI = ["auto", *MODELOS_INPAINT]


def normalizar_modelo_inpaint(value, default: str = "auto") -> str:
    """Normaliza aliases de UI/config y conserva el identificador especial ``B/N``."""
    raw = str(value or default).strip()
    normalized = raw.lower()
    aliases = {
        "": default,
        "default": default,
        "automatico": "auto",
        "automático": "auto",
        "opencv": "opencv-tela",
        "telea": "opencv-tela",
        "opencv_telea": "opencv-tela",
        "lama": "lama_mpe",
        "lama-mpe": "lama_mpe",
        "lama_large": "lama_large_512px",
        "lama-large": "lama_large_512px",
        "bn": "B/N",
        "b/n": "B/N",
        "blanco_y_negro": "B/N",
    }
    candidate = aliases.get(normalized, normalized)
    if candidate == "b/n":
        candidate = "B/N"
    if candidate not in MODELOS_INPAINT_UI:
        fallback = aliases.get(str(default).strip().lower(), str(default).strip())
        return fallback if fallback in MODELOS_INPAINT_UI else "auto"
    return candidate

# RUTAS LOCALES
RUTA_LOCAL_MODELO_INPAINTING = construir_ruta(BASE_DIR, "models", "inpainting")
RUTA_MODELO_LAMA = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "lama_mpe.ckpt")
RUTA_MODELO_LAMA_LARGE = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "lama_large_512px.ckpt")
RUTA_MODELO_AOT = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "aot.ckpt")
RUTA_LOCAL_FUENTES = construir_ruta(BASE_DIR, "fonts")
RUTA_FUENTE = construir_ruta(RUTA_LOCAL_FUENTES, "NewWildWordsRoman.ttf")
RUTA_LOCAL_PDFS = construir_ruta(BASE_DIR, "pdfs")
RUTA_LOCAL_ZIPS = construir_ruta(BASE_DIR, "zips")
RUTA_LOCAL_TEMPORAL = construir_ruta(BASE_DIR, "temp")

# RENDERIZADO
TAMANIO_MINIMO_FUENTE = 12
FACTOR_ESPACIO = 0.42

# RECURSOS REMOTOS
URL_FUENTE = "https://drive.google.com/file/d/1uIAh-nGGi04f-7moWsKvRhTbAj-Oq84O/view?usp=sharing"
URL_MODELO_LAMA = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt"
URL_MODELO_LAMA_LARGE = "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt"
URL_MODELO_AOT = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt"
# Detector de texto de comic/manga (comic-text-detector). Sale del mismo release
# que lama y aot. Se usa la exportacion ONNX porque corre en `cv2.dnn`, que ya es
# dependencia: no anade paquetes ni compite por la GPU con YOLO/OCR/inpainting.
URL_MODELO_COMIC_TEXT_DETECTOR = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
# Detector de bloques de texto y globos (RT-DETR-v2, Apache-2.0). A diferencia del
# anterior NO corre en `cv2.dnn` —el importador ONNX de OpenCV falla en el nodo CumSum
# del encoder—, asi que necesita el extra opcional `onnxruntime`.
URL_MODELO_RTDETR_COMIC_TEXT = "https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/detector.onnx"

# COLORES
COLOR_BLANCO = (255, 255, 255)
COLOR_NEGRO = (0, 0, 0)