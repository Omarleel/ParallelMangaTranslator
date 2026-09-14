"""El pipeline de una página como lista de etapas, no como método.

Por qué
-------
`limpiar_manga` y `traducir_manga` encadenaban las etapas por dentro. Quien quisiera
ejecutar sólo una parte —medir sin traducir, transcribir sin rotular— no podía
componerla: tenía que reescribir la secuencia. `eval_runner` hacía exactamente eso, y al
reescribirla metía la mano en el estado interno de `TranslateManga`. Dos secuencias paralelas para el mismo trabajo es una que se
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

from typing import Iterable, List, Protocol, runtime_checkable

from parallel_manga_translator.architecture.ports import PageCleanerPort, PageTranslatorPort
from parallel_manga_translator.models.page_context import PageContext


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
        self.cleaner.limpiar_manga(ctx)


class ExtraerRegiones:
    """Qué zonas llevan texto y sus recortes.

    Sin regiones previas cae a la máscara de capa, que es la rama que ya tenía
    `traducir_manga` cuando `text_regions` venía vacío.
    """

    nombre = "extraer_regiones"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        self.translator.extraer_regiones(ctx)


class TranscribirTextos:
    """OCR de transcripción. Aquí termina un pipeline solo-OCR."""

    nombre = "transcribir"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        self.translator.obtener_textos(ctx)


class TraducirTextos:
    """Normaliza, traduce y deja el rastro en las colas de JSON."""

    nombre = "traducir"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        self.translator.traducir_textos_de_regiones(ctx)


class PublicarTranscripcion:
    """Deja la transcripción en su cola sin traducir.

    Es lo que separa «limpiar y transcribir» de un solo-OCR que no deja rastro: el OCR ya
    puso los textos en el contexto, pero quien los escribe en `Transcripción.json` es el
    paso 3, y ese paso traduce.
    """

    nombre = "publicar_transcripcion"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        self.translator.publicar_transcripcion(ctx)


class RotularPagina:
    """Dibuja los textos ya resueltos sobre la imagen limpia."""

    nombre = "rotular"

    def __init__(self, translator: PageTranslatorPort) -> None:
        self.translator = translator

    def run(self, ctx: PageContext) -> None:
        self.translator.rotular(ctx)


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


def pipeline_transcripcion(translator: PageTranslatorPort) -> Pipeline:
    """Localiza, transcribe y publica. Ni traduce ni rotula."""
    return Pipeline([ExtraerRegiones(translator), TranscribirTextos(translator), PublicarTranscripcion(translator)])


#: Lo que se le puede pedir al pipeline para una página. `traducir` es el de siempre.
MODOS_PIPELINE = ("traducir", "limpiar", "limpiar_transcribir")


def normalizar_modo_pipeline(valor: object, por_defecto: str = "traducir") -> str:
    modo = str(valor or "").strip().lower()
    return modo if modo in MODOS_PIPELINE else por_defecto


def pipelines_por_modo(
    modo: str, cleaner: PageCleanerPort, translator: PageTranslatorPort
) -> tuple[Pipeline, Pipeline]:
    """Las dos composiciones que recibe `ImageProcessor`, según lo que se quiera obtener.

    Siguen siendo dos porque `procesar_pipeline` las corre en hilos distintos. Parar antes
    no es un modo degradado: limpiar sin traducir es un trabajo legítimo, y transcribir sin
    traducir deja el JSON que el editor manual sabe abrir.
    """
    modo = normalizar_modo_pipeline(modo)
    limpieza = pipeline_limpieza(cleaner)
    if modo == "limpiar":
        # Sin etapas: `imagen_final` queda a None y el orquestador no escribe traducción.
        return limpieza, Pipeline([])
    if modo == "limpiar_transcribir":
        return limpieza, pipeline_transcripcion(translator)
    return limpieza, pipeline_traduccion(translator)


def pipeline_limpieza_y_ocr(cleaner: PageCleanerPort, translator: PageTranslatorPort) -> Pipeline:
    """Lo que mide `eval_runner`: limpia, localiza y transcribe. No traduce ni rotula."""
    return Pipeline(pipeline_limpieza(cleaner).stages + pipeline_solo_ocr(translator).stages)


__all__ = [
    "MODOS_PIPELINE",
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
    "pipeline_transcripcion",
    "pipelines_por_modo",
    "normalizar_modo_pipeline",
    "PublicarTranscripcion",
]
