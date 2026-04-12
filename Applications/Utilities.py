from __future__ import annotations

import platform
import subprocess
import threading
import time
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from Utils.Constantes import (
    RUTA_ACTUAL,
    RUTA_FUENTE,
    RUTA_LOCAL_FUENTES,
    RUTA_LOCAL_MODELO_INPAINTING,
    RUTA_LOCAL_TEMPORAL,
    RUTA_MODELO_AOT,
    RUTA_MODELO_LAMA,
    RUTA_MODELO_LAMA_LARGE,
    URL_FUENTE,
    URL_MODELO_AOT,
    URL_MODELO_LAMA,
    URL_MODELO_LAMA_LARGE,
)
from .LoggingConfig import get_logger
from .RemoteFileDownloader import RemoteFileDownloader

logger = get_logger(__name__)


class Utilities:
    def __init__(self) -> None:
        self.canvas_pdf = None
        self.remote_file_downloader = RemoteFileDownloader()
        Path(RUTA_LOCAL_TEMPORAL).mkdir(parents=True, exist_ok=True)

    def descargar_pdf(self, ruta_pdf_resultante: str | None) -> None:
        if ruta_pdf_resultante:
            hilo_descarga = threading.Thread(target=self.realizar_descarga_pdf, args=(ruta_pdf_resultante,))
            hilo_descarga.daemon = True
            hilo_descarga.start()

    def abrir_pdf(self, ruta_pdf_resultante: str) -> None:
        try:
            if platform.system() == "Windows":
                subprocess.Popen([ruta_pdf_resultante], shell=True)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", ruta_pdf_resultante])
            else:
                subprocess.Popen(["xdg-open", ruta_pdf_resultante])
        except Exception as exc:
            logger.error("Error al abrir el archivo PDF: %s", exc)

    def realizar_descarga_pdf(self, ruta_pdf_resultante: str) -> None:
        try:
            from google.colab import files
            files.download(ruta_pdf_resultante)
        except Exception as exc:
            logger.error("Error al descargar el archivo: %s", exc)

    def capitalizar_oraciones(self, texto: str) -> str:
        oraciones = [s.strip().capitalize() for s in texto.replace("\n", " ").split(". ") if s]
        return " ".join(oraciones)

    def generar_pdf(self, imagen_actual: int, canvas_pdf, ruta_imagen_resultante: str) -> None:
        self.canvas_pdf = canvas_pdf
        img = Image.open(ruta_imagen_resultante)

        img_aspect_ratio = img.width / img.height
        page_width, page_height = self.canvas_pdf._pagesize

        if img_aspect_ratio > 1:
            new_img_width = page_width
            new_img_height = int(page_width / img_aspect_ratio)
        else:
            new_img_height = page_height
            new_img_width = int(page_height * img_aspect_ratio)

        img = img.resize((int(new_img_width), int(new_img_height)), Image.LANCZOS)

        timestamp = int(time.time())
        temp_dir = Path(RUTA_LOCAL_TEMPORAL)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_img_path = temp_dir / f"temp_image_{imagen_actual}_{timestamp}.jpg"
        img.save(temp_img_path, format="JPEG", quality=80)

        self.canvas_pdf.drawImage(str(temp_img_path), 0, 0, width=int(new_img_width), height=int(new_img_height))
        self.canvas_pdf.showPage()
        temp_img_path.unlink(missing_ok=True)

    def guardar_pdf(self) -> None:
        if self.canvas_pdf is not None:
            self.canvas_pdf.save()

    def _download_if_missing(self, target_file: str, download_url: str, output_path: str, output_filename: str | None = None) -> None:
        if not Path(target_file).is_file():
            logger.info("Descargando recurso faltante: %s", Path(target_file).name)
            self.remote_file_downloader.download(
                download_url=download_url,
                output_path=output_path,
                output_filename=output_filename,
            )

    def download_fonts(self) -> None:
        self._download_if_missing(
            target_file=RUTA_FUENTE,
            download_url=URL_FUENTE,
            output_path=RUTA_LOCAL_FUENTES,
        )

    def download_models(self) -> None:
        self._download_if_missing(
            target_file=RUTA_MODELO_LAMA,
            download_url=URL_MODELO_LAMA,
            output_path=RUTA_LOCAL_MODELO_INPAINTING,
            output_filename="lama_mpe.ckpt",
        )
        self._download_if_missing(
            target_file=RUTA_MODELO_LAMA_LARGE,
            download_url=URL_MODELO_LAMA_LARGE,
            output_path=RUTA_LOCAL_MODELO_INPAINTING,
            output_filename="lama_large_512px.ckpt",
        )
        self._download_if_missing(
            target_file=RUTA_MODELO_AOT,
            download_url=URL_MODELO_AOT,
            output_path=RUTA_LOCAL_MODELO_INPAINTING,
            output_filename="aot.ckpt",
        )

    def descargar_y_extraer_zip(self, manager, url_archivo: str) -> str | None:
        from zipfile import ZipFile
        from pathlib import Path
        try:
            ruta_archivo_descargado = manager.download_file_from_link(
                file_link=url_archivo,
                output_folder=RUTA_ACTUAL,
            )
            if not ruta_archivo_descargado:
                return None

            ruta_archivo = Path(ruta_archivo_descargado)
            ruta_extraccion = Path(RUTA_ACTUAL) / ruta_archivo.stem
            ruta_extraccion.mkdir(parents=True, exist_ok=True)

            logger.info("Extrayendo archivo ZIP en: %s", ruta_extraccion)
            with ZipFile(ruta_archivo, "r") as zip_ref:
                zip_ref.extractall(ruta_extraccion)

            ruta_archivo.unlink(missing_ok=True)
            return str(ruta_extraccion)
        except Exception as exc:
            logger.error("Error al descargar y extraer zip: %s", exc)
            return None

    def convertir_a_diccionarios(self, lista_de_listas):
        lotes_diccionarios = []
        indice_acumulado = 0
        for sublist in lista_de_listas:
            lote_diccionario = {}
            for archivo in sublist:
                lote_diccionario[indice_acumulado] = archivo
                indice_acumulado += 1
            lotes_diccionarios.append(lote_diccionario)
        return lotes_diccionarios

    @staticmethod
    def is_colab() -> bool:
        try:
            import google.colab  # noqa: F401
            return True
        except ImportError:
            return False