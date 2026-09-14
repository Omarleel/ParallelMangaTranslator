"""Elegir al crear el trabajo qué se le pide al pipeline.

Limpiar sin traducir, o limpiar y transcribir sin traducir, no son ejecuciones a medias:
son trabajos completos. Quien maqueta necesita las páginas limpias; quien traduce fuera
necesita `Transcripción.json` y luego reimporta con «Importar textos».

La capacidad ya existía —el pipeline es una lista de etapas componible— y esto solo la
expone. Por eso el mapa de modos vive en `processing/pipeline.py`, junto a las
composiciones, y no en la UI: la UI elige, no define.
"""

from __future__ import annotations

from pathlib import Path

from parallel_manga_translator.processing.pipeline import MODOS_PIPELINE, pipelines_por_modo
from parallel_manga_translator.ui.job_manager import (
    JobManager,
    JobOptions,
    JobState,
    normalize_pipeline_mode,
)

STATIC = Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui" / "static"


def test_el_defecto_es_el_trabajo_de_siempre() -> None:
    assert JobOptions().modo == "traducir"


def test_un_modo_desconocido_cae_en_hacer_el_trabajo_completo() -> None:
    """La caída es hacia MÁS trabajo, nunca hacia menos.

    Un valor raro que se convirtiera en «solo limpiar» dejaría al usuario sin traducción
    sin decírselo. Al revés solo se gasta tiempo de más, que se ve enseguida.
    """
    for modo in MODOS_PIPELINE:
        assert normalize_pipeline_mode(modo) == modo
    for basura in ("", None, "limpar", "borrar_todo"):
        assert normalize_pipeline_mode(basura) == "traducir", basura
    assert normalize_pipeline_mode("  TRADUCIR ") == "traducir", "el formulario puede mandar espacios y mayúsculas"


def test_cada_modo_compone_las_etapas_que_promete() -> None:
    class _Cleaner:
        def limpiar_manga(self, ctx):
            pass

    class _Translator:
        def extraer_regiones(self, ctx):
            pass

        def obtener_textos(self, ctx):
            pass

        def publicar_transcripcion(self, ctx):
            pass

        def traducir_textos_de_regiones(self, ctx):
            pass

        def rotular(self, ctx):
            pass

    esperado = {
        "traducir": ["extraer_regiones", "transcribir", "traducir", "rotular"],
        "limpiar_transcribir": ["extraer_regiones", "transcribir", "publicar_transcripcion"],
        "limpiar": [],
    }
    for modo, etapas in esperado.items():
        limpieza, traduccion = pipelines_por_modo(modo, _Cleaner(), _Translator())
        assert limpieza.nombres == ["limpieza"], f"{modo} siempre limpia"
        assert traduccion.nombres == etapas, modo


def test_la_eleccion_llega_a_la_configuracion_del_pipeline(tmp_path: Path) -> None:
    """Sin esto la opción se queda en el manifiesto y el pipeline hace lo de siempre."""
    manager = JobManager(jobs_root=tmp_path / "jobs", start_worker=False)
    job = JobState(
        job_id="job-modo",
        title="x",
        root_dir=str(tmp_path / "jobs" / "job-modo"),
        input_dir=str(tmp_path / "entrada"),
        output_dir=str(tmp_path / "salida"),
        options=JobOptions(modo="limpiar_transcribir"),
    )

    config = manager._build_config_for_job(job)

    assert config.processing.modo_pipeline == "limpiar_transcribir"


def test_la_ui_ofrece_los_tres_modos_y_los_envia() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="pipelineMode"' in html
    for modo in MODOS_PIPELINE:
        assert f'value="{modo}"' in html, f"falta la opción {modo} en el formulario"
    assert "data.append('modo', pipelineMode?.value || 'traducir')" in javascript
    # Un trabajo que no traduce no puede ofrecer ajustes de traducción como si contaran.
    assert "campo.disabled = !traduce" in javascript
