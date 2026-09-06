"""Escritura de imágenes que no falla en silencio.

`cv2.imwrite` **devuelve un bool** y no lanza nada. Ignorarlo convierte un fallo de
escritura en un error mucho más tarde y muy lejos de su causa: la UI escribía una revisión
de fondo que nunca llegaba al disco y el error aparecía después como "No se pudo leer la
capa de fondo", apuntando a un archivo que jamás existió.

El disparador medido en Windows es el límite `MAX_PATH` de 260 caracteres: con rutas más
largas `imwrite` devuelve `False` sin escribir. Le pasa a cualquiera con una carpeta de
trabajos anidada, no sólo a los tests.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

#: Longitud a partir de la cual Windows rechaza la ruta salvo que se active el soporte
#: de rutas largas. Se usa sólo para dar un diagnóstico útil, no para decidir.
MAX_PATH_WINDOWS = 260


def write_image(path: str | Path, image: np.ndarray) -> Path:
    """Escribe `image` en `path` y falla ruidosamente si no se pudo.

    Devuelve la ruta escrita para poder encadenar.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)

    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError(f"No hay imagen que escribir en {destino}")

    if cv2.imwrite(str(destino), image):
        return destino

    raise OSError(f"cv2.imwrite no pudo escribir {destino}.{_pista(destino)}")


def try_write_image(path: str | Path, image: np.ndarray, *, logger) -> bool:
    """Igual que `write_image` pero avisa en vez de fallar.

    Para artefactos opcionales —debug, volcados— donde perder el archivo es molesto pero
    abortar la página sería peor. Lo que no vale es la tercera opción: ignorarlo.
    """
    try:
        write_image(path, image)
        return True
    except (OSError, ValueError) as exc:
        logger.warning("No se pudo escribir el artefacto opcional: %s", exc)
        return False


def _pista(destino: Path) -> str:
    if len(str(destino)) > MAX_PATH_WINDOWS:
        return (
            f" La ruta tiene {len(str(destino))} caracteres y Windows corta en "
            f"{MAX_PATH_WINDOWS}: mueve el trabajo a una carpeta menos anidada o activa "
            "el soporte de rutas largas."
        )
    return ""
