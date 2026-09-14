"""Lo que una página acumula al pasar por las etapas del pipeline.

Vive en `models/` porque lo comparten dos capas que no deben conocerse: `processing/`, que
ejecuta las etapas, y quien las compone —`ImageProcessor`, `eval_runner`, la retraducción
de la UI—. Antes estaba dentro de `processing/pipeline.py` y eso obligaba a que cualquiera
que quisiera nombrar el estado de una página importase el módulo del pipeline entero.

Por qué existe
--------------
El estado de la página en curso vivía como atributos de `TranslateManga` (`ultimas_regiones`,
`ultimo_estilos_texto`, `ultimos_source_language_flags`, …). Cinco mixins se hablaban por
ahí sin declararlo, `ImageProcessor` lo leía con `getattr` para sacar métricas, la
retraducción de la UI **escribía** en esos atributos para poder reutilizar un paso, y hasta
el puerto `PageTranslatorPort` tuvo que declarar `ultimas_regiones` como parte del contrato.
Un objeto de vida larga haciendo de cuaderno de notas de la página en curso.

El limpiador tenía lo mismo en su mitad: `last_regions` (que ya no leía nadie) y el
contexto de depuración de inpaint, cuatro atributos que `PageCleanerPort` empujaba con dos
métodos antes de cada página y trece `getattr` leían después.

Ahora el cuaderno es este dato, se crea por página y se pasa explícito a las dos mitades.
`TranslateManga` y `CleanManga` vuelven a ser lo que debían: colaboradores y configuración,
sin memoria de la última página.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from parallel_manga_translator.models.processing_models import TextRegion


@dataclass
class PageContext:
    """El estado de una página mientras la recorren las etapas.

    Los campos son `Optional`/vacíos a propósito: una composición parcial deja sin rellenar
    los de las etapas que no incluye, y quien la use tiene que mirarlos. `imagen_final` a
    `None` significa que nadie rotuló, no que el rotulado fallara.
    """

    imagen: np.ndarray
    #: Posición de la página en el trabajo (0-based). La usan las colas de JSON y la
    #: memoria de personajes; es de la página, no del traductor.
    indice_pagina: int = 0
    #: Nombre con el que sale esta página, y el del archivo de entrada. Identifican la
    #: página en los artefactos de depuración y en los logs.
    nombre_archivo: str = ""
    archivo_origen: str = ""
    #: Dónde escribir los artefactos de depuración de esta página. Vacío = no escribir.
    #: Es una petición del que orquesta, no configuración del limpiador.
    debug_root: str = ""
    imagen_limpia: Optional[np.ndarray] = None
    mascara_capa: Optional[np.ndarray] = None
    regiones: List[TextRegion] = field(default_factory=list)
    #: Las que sobreviven a la extraccion, en orden de lectura y alineadas con `cuadros`.
    #: No es lo mismo que `regiones`: la extraccion puede descartar, asi que medir
    #: limpieza sobre estas en vez de sobre las crudas cambiaria lo que se mide.
    regiones_ordenadas: List[TextRegion] = field(default_factory=list)
    cuadros: List[Any] = field(default_factory=list)
    recortes: List[Any] = field(default_factory=list)
    #: Transcripción cruda del OCR, una por recorte.
    textos: List[str] = field(default_factory=list)
    #: La misma transcripción ya normalizada. Es la entrada real del traductor y lo que se
    #: escribe en `Transcripción.json`.
    textos_originales: List[str] = field(default_factory=list)
    textos_traducidos: List[str] = field(default_factory=list)
    #: Lo que se dibuja: vacío donde el filtro de idioma descartó o donde la onomatopeya
    #: conserva el arte original.
    textos_para_render: List[str] = field(default_factory=list)
    #: Estilo tipográfico por región (`dialogo`, `onomatopeya`, `narracion`, …).
    estilos: List[str] = field(default_factory=list)
    #: Qué regiones pasaron el filtro de idioma de origen.
    flags_idioma_origen: List[bool] = field(default_factory=list)
    #: Hablante asignado por la memoria de personajes. Vacío salvo en modo LLM.
    asignaciones_hablante: List[Dict[str, Any]] = field(default_factory=list)
    imagen_final: Optional[np.ndarray] = None

    def region_en(self, indice: int) -> Optional[TextRegion]:
        """La región alineada con la posición `indice`, o `None` si no hay.

        Los pasos recorren textos y cajas por posición, y no siempre hay tantas regiones
        como textos: sin regiones previas la extracción cae a la máscara de capa y no
        produce ninguna. Este acceso evita repetir la misma comprobación en ocho sitios.
        """
        if 0 <= indice < len(self.regiones_ordenadas):
            return self.regiones_ordenadas[indice]
        return None

    @property
    def regiones_alineadas(self) -> bool:
        """¿Hay exactamente una región por caja? Si no, no se puede indexar en paralelo."""
        return bool(self.regiones_ordenadas) and len(self.regiones_ordenadas) == len(self.cuadros)


__all__ = ["PageContext"]
