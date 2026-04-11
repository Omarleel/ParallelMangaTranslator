from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranslationConfig:
    idioma_entrada: str
    idioma_salida: str
    metodo_traduccion: str = "Tradicional"
    modelo_inpaint: str = "lama_mpe"
    lore_manga: str = ""
    groq_api_key: str = ""


@dataclass(frozen=True)
class ProcessingConfig:
    ruta_carpeta_entrada: str = "Dataset"
    batch_size: int = 8
    usar_paralelismo: bool = True

    @property
    def ruta_carpeta_salida(self) -> str:
        return str(Path(self.ruta_carpeta_entrada) / "Outputs")

    @property
    def ruta_carpeta_limpieza(self) -> str:
        return str(Path(self.ruta_carpeta_salida) / "Limpieza")

    @property
    def ruta_carpeta_traduccion(self) -> str:
        return str(Path(self.ruta_carpeta_salida) / "Traduccion")


@dataclass(frozen=True)
class ApplicationConfig:
    translation: TranslationConfig
    processing: ProcessingConfig