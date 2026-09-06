"""El pipeline de una página como lista de etapas, no como método.

Por qué
-------
`limpiar_manga` y `traducir_manga` encadenaban las etapas por dentro. Quien quisiera
ejecutar sólo una parte —medir sin traducir, transcribir sin rotular— no podía
componerla: tenía que reescribir la secuencia. `eval_runner` hacía exactamente eso, y al
reescribirla metía la mano en `ultimas_regiones`, que es estado interno de
`TranslateManga`. Dos secuencias paralelas para el mismo trabajo es una que se
desincroniza en silencio, y aquí la que se desincroniza es la que mide.

Aquí las etapas son objetos y la secuencia es un dato:

    Pipeline([LimpiarPagina(cleaner), ExtraerRegiones(t), TranscribirTextos(t)])

Qué NO hace
-----------
No decide reintentos, ni memoria, ni hilos, ni escribe ficheros. Eso sigue en
`ImageProcessor`, que es quien sabe de páginas, de OOM y de la tubería productor/
consumidor. Una etapa aquí es un paso puro sobre `PageContext`; el orquestador sigue
siendo el mismo.

Tampoco introduce configuración: las composiciones con nombre de abajo son las que ya
ejecutaba el código, escritas como lo que son.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Protocol, runtime_checkable

import numpy as np

from parallel_manga_translator.architecture.ports import PageCleanerPort, PageTranslatorPort
from parallel_manga_translator.models.processing_models import TextRegion


@dataclass
class PageContext:
    """Lo que una página acumula al pasar por las etapas.

    Empieza con la imagen original y cada etapa rellena lo suyo. Los campos son
    `Optional` a propósito: una composición parcial deja vacíos los de las etapas que no
    incluye, y quien la use tiene que mirarlos. `imagen_final` a `None` significa que
    nadie rotuló, no que el rotulado fallara.
    """

    imagen: np.ndarray
    imagen_limpia: Optional[np.ndarray] = None
    mascara_capa: Optional[np.ndarray] = None
    regiones: List[TextRegion] = field(default_factory=list)
    #: Las que sobreviven a la extraccion, en orden de lectura y alineadas con `cuadros`.
    #: No es lo mismo que `regiones`: la extraccion puede descartar, asi que medir
    #: limpieza sobre estas en vez de sobre las crudas cambiaria lo que se mide.
    regiones_ordenadas: List[TextRegion] = field(default_factory=list)
    cuadros: List[Any] = field(default_factory=list)
    recortes: List[Any] = field(default_factory=list)
    textos: List[str] = field(default_factory=list)
    textos_para_render: List[str] = field(default_factory=list)
    imagen_final: Optional[np.ndarray] = None


@runtime_checkable
class PageStage(Protocol):
    """Un paso sobre `PageContext`. `nombre` es lo que sale en logs y errores."""

    nombre: str

    def run(self, ctx: PageContext) -> None:
        ...


class LimpiarPagina:
    """Detección de regiones + borrado de tinta. La única etapa que usa el limpiador."""

    nombre = "limpieza"

    def __init__(self, cleaner: PageCleanerPort) -> None:
        self.cleaner = cleaner

    def run(self, ctx: PageContext) -> None:
        ctx.mascara_capa, ctx.imagen_limpia, regiones = self.cleaner.limpiar_manga(ctx.imagen)
        ctx.regiones = list(regiones)


class ExtraerRegiones:
    """Qué zonas llevan texto y sus recortes.

    Sin regiones previas cae a la máscara de capa, que es la rama que ya tenía
    `traducir_manga` cuando `text_regions` venía vacío.
    """

    nombre = "extraer_regiones"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        ctx.cuadros, ctx.recortes = self.translator.extraer_regiones(
            ctx.imagen, ctx.mascara_capa, ctx.regiones or None
        )
        # `extraer_regiones` deja las ordenadas en el traductor y los pasos siguientes
        # las leen de ahi. Copiarlas al contexto hace explicito ese acoplamiento en vez
        # de obligar a cada consumidor a hurgar en el estado del traductor.
        ctx.regiones_ordenadas = list(self.translator.ultimas_regiones)


class TranscribirTextos:
    """OCR de transcripción. Aquí termina un pipeline solo-OCR."""

    nombre = "transcribir"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        ctx.textos = self.translator.obtener_textos(ctx.recortes)


class TraducirTextos:
    """Normaliza, traduce y deja el rastro en las colas de JSON."""

    nombre = "traducir"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        ctx.textos_para_render = self.translator.traducir_textos_de_regiones(ctx.cuadros, ctx.textos)


class RotularPagina:
    """Dibuja los textos ya resueltos sobre la imagen limpia."""

    nombre = "rotular"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        ctx.imagen_final = self.translator.rotular(ctx.imagen_limpia, ctx.cuadros, ctx.textos_para_render)


class Pipeline:
    """Una secuencia de etapas sobre una misma página."""

    def __init__(self, stages: Iterable[PageStage]) -> None:
        self.stages: List[PageStage] = list(stages)
        for stage in self.stages:
            if not isinstance(stage, PageStage):
                raise TypeError(
                    f"{type(stage).__name__} no cumple PageStage: hace falta `nombre` y `run(ctx)`."
                )

    @property
    def nombres(self) -> List[str]:
        return [stage.nombre for stage in self.stages]

    def run(self, ctx: PageContext) -> PageContext:
        for stage in self.stages:
            stage.run(ctx)
        return ctx

    def __repr__(self) -> str:
        return f"Pipeline({' -> '.join(self.nombres)})"


# --------------------------------------------------------------------------------------
# Composiciones con nombre
#
# No son configuración ni modos nuevos: son las secuencias que el código ya ejecutaba.
# `limpieza` y `traduccion` estan separadas porque `ImageProcessor.procesar_pipeline` las
# corre en hilos distintos —el productor prepara la limpieza de la pagina N+1 mientras el
# principal termina la N—, de modo que esa division es real, no decorativa.
# --------------------------------------------------------------------------------------


def pipeline_limpieza(cleaner: PageCleanerPort) -> Pipeline:
    return Pipeline([LimpiarPagina(cleaner)])


def pipeline_traduccion(translator: PageTranslatorPort) -> Pipeline:
    return Pipeline([
        ExtraerRegiones(translator),
        TranscribirTextos(translator),
        TraducirTextos(translator),
        RotularPagina(translator),
    ])


def pipeline_solo_ocr(translator: PageTranslatorPort) -> Pipeline:
    """Transcribe y para. Sin traductor de por medio: nada aquí lo llama."""
    return Pipeline([ExtraerRegiones(translator), TranscribirTextos(translator)])


# No hay `pipeline_completo`, y es deliberado: ninguna ruta real encadena las cinco
# etapas de un tiron. Las dos guardan la pagina limpia y encolan el JSON entre la
# limpieza y la traduccion, y `procesar_pipeline` ademas las corre en hilos distintos.
# Una composicion "completa" solo serviria para que alguien la usara y se saltara eso.


def pipeline_limpieza_y_ocr(cleaner: PageCleanerPort, translator: PageTranslatorPort) -> Pipeline:
    """Lo que mide `eval_runner`: limpia, localiza y transcribe. No traduce ni rotula."""
    return Pipeline(pipeline_limpieza(cleaner).stages + pipeline_solo_ocr(translator).stages)


__all__ = [
    "PageContext",
    "PageStage",
    "Pipeline",
    "LimpiarPagina",
    "ExtraerRegiones",
    "TranscribirTextos",
    "TraducirTextos",
    "RotularPagina",
    "pipeline_limpieza",
    "pipeline_traduccion",
    "pipeline_solo_ocr",
    "pipeline_limpieza_y_ocr",
]
