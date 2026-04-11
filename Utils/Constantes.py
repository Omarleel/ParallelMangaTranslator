from pathlib import Path


def construir_ruta(base, *paths) -> str:
    return str(Path(base).joinpath(*paths))


BASE_DIR = Path(__file__).resolve().parents[1]

# CONSTANTES GENERALES
PESO_MODELOS = 7.8  # GB reservados aprox. por proceso con modelos pesados de OCR/inpainting
RUTA_ACTUAL = str(BASE_DIR)
RUTA_REMOTA = "ParallelMangaTranslator"

IDIOMAS_ENTRADA_DISPONIBLES = ["Chino", "Coreano", "Inglés", "Japonés"]
IDIOMAS_SALIDA_DISPONIBLES = ["Español", "Inglés", "Portugués", "Francés", "Italiano"]
MODELOS_INPAINT = ["opencv-tela", "lama_mpe", "lama_large_512px", "aot", "B/N"]

# RUTAS LOCALES
RUTA_LOCAL_MODELO_INPAINTING = construir_ruta(BASE_DIR, "Models", "inpainting")
RUTA_MODELO_LAMA = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "lama_mpe.ckpt")
RUTA_MODELO_LAMA_LARGE = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "lama_large_512px.ckpt")
RUTA_MODELO_AOT = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "aot.ckpt")
RUTA_LOCAL_FUENTES = construir_ruta(BASE_DIR, "Fonts")
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

# COLORES
COLOR_BLANCO = (255, 255, 255)
COLOR_NEGRO = (0, 0, 0)