import os
import re
from urllib.parse import unquote
import requests
from parallel_manga_translator.config.constants import RUTA_ACTUAL

class RemoteFileDownloader:
    def __init__(self):
        pass
    
    def prepare_url(self, download_url):
        """
        Prepara la URL de descarga para garantizar que sea un enlace directo utilizable.

        Args:
            download_url (str): La URL original de descarga.

        Returns:
            str: La URL de descarga preparada.
        """
        try:
            # Patrón de expresión regular para enlaces
            patron_google_drive = r'https?://drive\.google\.com/file/d/.+/view(\?usp=sharing)?'
            patron_dropbox = r'https?://(www\.)?dropbox\.com/scl/.+'
            download_link = download_url

            # Verificar si la URL corresponde a un enlace de Google Drive
            if re.match(patron_google_drive, download_url):
                file_id_search = re.search(r'd/([a-zA-Z0-9_-]+)', download_url)
                if file_id_search:
                    file_id = file_id_search.group(1)
                    download_link = f"https://drive.google.com/uc?export=download&id={file_id}"
            # Verificar si la URL corresponde a un enlace de Dropbox
            elif re.match(patron_dropbox, download_url):
                download_link = download_url.replace("?dl=0", "?dl=1")

            return download_link
        except Exception as e:
            print(f"Error al preparar la URL de descarga: {e}")
            return None
    
    def download(self, download_url, output_path = RUTA_ACTUAL, output_filename = None):
        """
        Descarga un archivo desde una URL remota y lo guarda en el sistema de archivos local.

        Args:
            download_url (str): La URL del archivo a descargar.
            output_path (str): El directorio donde se guardará el archivo descargado.
            output_filename (str, opcional): El nombre del archivo descargado. Si es None, se utilizará el nombre de archivo original.

        Returns:
            str: La ruta al archivo descargado en el sistema de archivos local.
        """
        try:
            os.makedirs(output_path, exist_ok=True)
            # Envía una solicitud HTTP GET a la URL de descarga
            remote_download_url = self.prepare_url(download_url)
            response = requests.get(remote_download_url, stream=True, headers={'user-agent': 'Wget/1.16 (linux-gnu)'})
            response.raise_for_status()  # Lanza una excepción para códigos de estado 4xx o 5xx
            content_disposition = response.headers.get('content-disposition')
            if output_filename is None:
                if content_disposition:
                    # Extrae el nombre del archivo del encabezado si está presente
                    filename = re.findall('filename="(.+)"', content_disposition)
                    if filename:
                       output_filename = unquote(filename[0])
                else:
                    output_filename = 'file'
            # Construye la ruta completa para el archivo de salida
            output_file_path = os.path.join(output_path, output_filename)

            with open(output_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # Filtra los keep-alive nuevos chunks
                        f.write(chunk)
                        
            return output_file_path  # Devuelve la ruta al archivo descargado

        except Exception as e:
            print(f"Error al descargar el archivo desde {download_url}: {e}")
            return None
