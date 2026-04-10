import os
from loguru import logger
import warnings
import torch
from dotenv import load_dotenv  # <-- Importamos dotenv

from Applications.ParallelProcessor import ParallelProcessor
from Applications.ImageProcessor import ImageProcessor
from Applications.Utilities import Utilities
from Utils.Constantes import MODELOS_INPAINT

logger.remove()
warnings.filterwarnings("ignore", message="The class ViTFeatureExtractor is deprecated", category=FutureWarning)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == '__main__':
    load_dotenv() 

    torch.cuda.empty_cache()
    utilities = Utilities()
    utilities.download_fonts()
    utilities.download_models()

    modelo_inpaint = MODELOS_INPAINT[1]
    print(f"Modelo inpaint seleccionado: {modelo_inpaint}")
    batch_size = 8 # Tamaño de lote -> Si hay 24 imágenes entonces se crearán 3 procesos de 8 imágenes cada uno
    idioma_entrada = "Japonés"
    idioma_salida = "Español"
    ruta_carpeta_entrada = "Dataset"
    
    metodo_traduccion = "LLM" 
    
    api_key_groq = os.getenv("GROQ_API_KEY", "")
    
    if metodo_traduccion == "LLM" and not api_key_groq:
        logger.warning("No se encontró GROQ_API_KEY en el archivo .env. La traducción LLM podría fallar.")
    
    usar_paralelismo = False if metodo_traduccion == "LLM" else True

    ruta_carpeta_salida = os.path.join(ruta_carpeta_entrada, "Outputs")
    ruta_carpeta_limpieza = os.path.join(ruta_carpeta_salida, "Limpieza")
    ruta_carpeta_traduccion = os.path.join(ruta_carpeta_salida, "Traduccion")
    os.makedirs(ruta_carpeta_salida, exist_ok=True)
    os.makedirs(ruta_carpeta_limpieza, exist_ok=True)
    os.makedirs(ruta_carpeta_traduccion, exist_ok=True)

    image_procesor = ImageProcessor(
        idioma_entrada=idioma_entrada,
        idioma_salida=idioma_salida,
        modelo_inpaint=modelo_inpaint,
        metodo_traduccion=metodo_traduccion,
        groq_api_key=api_key_groq
    )

    print(f"Iniciando procesamiento... Método: {metodo_traduccion} | Paralelismo: {usar_paralelismo}")

    parallel_processor = ParallelProcessor()
    parallel_processor.procesar(
        ruta_carpeta_entrada=ruta_carpeta_entrada,
        ruta_carpeta_salida=ruta_carpeta_salida,
        process_func=image_procesor.procesar,
        batch_size=batch_size,
        parallel=usar_paralelismo
    )