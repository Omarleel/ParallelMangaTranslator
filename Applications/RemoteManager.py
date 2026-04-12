import os
import re
import requests
from .LoggingConfig import get_logger

logger = get_logger(__name__)

class RemoteManager:
    def download_file_from_link(self, file_link: str, output_folder: str) -> str | None:
        try:
            logger.info("Iniciando descarga directa desde enlace externo...")
            
            respuesta = requests.get(file_link, stream=True)
            respuesta.raise_for_status()

            nombre_archivo = "Manga_Descargado.zip"
            header_cd = respuesta.headers.get('content-disposition')
            if header_cd:
                coincidencias = re.findall('filename="?([^"]+)"?', header_cd)
                if coincidencias:
                    nombre_archivo = coincidencias[0]

            if not nombre_archivo.lower().endswith(".zip"):
                nombre_archivo += ".zip"

            ruta_salida = os.path.join(output_folder, nombre_archivo)

            with open(ruta_salida, 'wb') as archivo:
                for chunk in respuesta.iter_content(chunk_size=8192):
                    if chunk:
                        archivo.write(chunk)

            logger.info("Descarga externa completada: %s", ruta_salida)
            return ruta_salida

        except Exception as exc:
            logger.error("Error al descargar desde el enlace directo: %s", exc)
            return None