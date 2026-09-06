from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from parallel_manga_translator.ui.job_manager import JobManager, JobOptions, JobState, PageState
from parallel_manga_translator.ui.retranslator import JobRetranslator, RetranslatedRegion

MODULE = "parallel_manga_translator.ui.job_manager"


def _offline_assets():
    return (
        mock.patch(f"{MODULE}.prepare_runtime"),
        mock.patch(f"{MODULE}.prepare_assets"),
    )


def _write_image(path: Path, value: int, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((size, size, 3), value, np.uint8))


PIPELINE_LAYOUT = {
    "version": 2,
    "bbox": [4, 4, 12, 10],
    "uses_clip_mask": True,
    "block_count": 1,
    "blocks": [{"slot": [2, 2, 8, 6], "area": [3, 3, 6, 4], "font_size": 11, "lines": [], "line_spacing": 1.0, "stroke_width": 1}],
}


def _region(index: int = 0, original: str = "こんにちは", translated: str = "hola") -> dict:
    return {
        "index": index,
        "bbox": [4, 4, 12, 10],
        "source_bbox": [4, 4, 12, 10],
        "original_text": original,
        "translated_text": translated,
        "style": "dialogo",
        "type": "dialogue",
        "confidence": 0.9,
        "visible": True,
        "modified": False,
        "deleted": False,
        "rotation_angle": 0.0,
        "ui_layout": dict(PIPELINE_LAYOUT),
    }


class _FakeRetranslator:
    """Sustituye al retraductor real: no traduce, pero sí escribe la salida."""

    instances: list["_FakeRetranslator"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[int] = []
        _FakeRetranslator.instances.append(self)

    def retranslate_page(self, *, page_index, clean_path, output_path, regions):
        self.calls.append(page_index)
        _write_image(Path(output_path), 200)
        return [
            RetranslatedRegion(
                index=int(region["index"]),
                bbox=list(region["bbox"]),
                original_text=str(region["original_text"]),
                translated_text=f"retraducido:{region['original_text']}",
                rendered_text=f"retraducido:{region['original_text']}",
                style="dialogo",
            )
            for region in regions
        ]


def _manager_with_ready_job(tmp: str, *, translator: str = "llm") -> tuple[JobManager, JobState]:
    jobs_root = Path(tmp)
    root_dir = jobs_root / "job-retrad"
    input_dir = root_dir / "entrada"
    output_dir = root_dir / "outputs"
    _write_image(input_dir / "pagina.png", 30)
    _write_image(output_dir / "limpieza" / "pagina.png", 240)
    _write_image(output_dir / "traduccion" / "pagina.png", 120)
    (output_dir / "traduccion" / "Traducción.json").write_text(
        json.dumps(
            {
                "Título": "Prueba",
                "Traducción": [
                    {
                        "Página": 1,
                        "Formato": "color",
                        "Globos de texto": [
                            {
                                "Índice": 0,
                                "Coordenadas": [[4, 4], [16, 14]],
                                "Texto": "hola",
                                "Estilo": "dialogo",
                                "Tipo": "dialogue",
                                "Layout UI": dict(PIPELINE_LAYOUT),
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    page = PageState(
        index=0,
        source_filename="pagina.png",
        output_filename="pagina.png",
        status="ready",
        original_path=str(input_dir / "pagina.png"),
        clean_path=str(output_dir / "limpieza" / "pagina.png"),
        translated_path=str(output_dir / "traduccion" / "pagina.png"),
        corrected_path=str(output_dir / "corregida" / "pagina.png"),
        corrections_path=str(output_dir / "correcciones" / "pagina.json"),
        regions=[_region()],
    )
    job = JobState(
        job_id=root_dir.name,
        title="Prueba",
        root_dir=str(root_dir),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        status="ready",
        pages=[page],
        options=JobOptions(translator=translator),
    )
    manager = JobManager(jobs_root=jobs_root, start_worker=False)
    manager._jobs[job.job_id] = job
    manager.manifests.save(job)
    return manager, job


class RetranslationJobTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRetranslator.instances = []

    def test_retranslation_uses_selected_translator_and_updates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, job = _manager_with_ready_job(tmp, translator="llm")
            manager.retranslate_job(job.job_id, translator="google")

            self.assertEqual(job.status, "queued")
            self.assertEqual(job.pending_operation, "retranslate")
            self.assertEqual(job.retranslate_pages, [0])
            self.assertEqual(job.options.translator, "google")

            runtime, assets = _offline_assets()
            with runtime, assets, mock.patch(f"{MODULE}.JobRetranslator", _FakeRetranslator):
                manager._run_job(job.job_id)

            completed = manager.get_job(job.job_id)
            self.assertEqual(completed.status, "ready")
            self.assertEqual(completed.pending_operation, "process")
            self.assertEqual(completed.retranslate_pages, [])
            self.assertEqual(completed.pages[0].regions[0]["translated_text"], "retraducido:こんにちは")
            self.assertEqual(completed.pages[0].regions[0]["original_text"], "こんにちは")

            # El traductor recibió la configuración del trabajo ya actualizada.
            self.assertEqual(_FakeRetranslator.instances[0].config.translation.metodo_traduccion, "Tradicional")

            traduccion = json.loads(
                (Path(job.output_dir) / "traduccion" / "Traducción.json").read_text(encoding="utf-8")
            )
            pagina = traduccion["Traducción"][0]
            self.assertEqual(pagina["Formato"], "color")
            self.assertEqual(pagina["Globos de texto"][0]["Texto"], "retraducido:こんにちは")

            # La geometría del globo no cambia al retraducir: recalcularla sin la máscara
            # de limpieza agrandaba la fuente y descuadraba los globos partidos.
            self.assertEqual(completed.pages[0].regions[0]["ui_layout"], PIPELINE_LAYOUT)
            self.assertEqual(pagina["Globos de texto"][0]["Layout UI"], PIPELINE_LAYOUT)

    def test_pages_with_manual_corrections_are_preserved_unless_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, job = _manager_with_ready_job(tmp)
            _write_image(Path(job.pages[0].corrected_path), 90)
            Path(job.pages[0].corrections_path).parent.mkdir(parents=True, exist_ok=True)
            Path(job.pages[0].corrections_path).write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError):
                manager.retranslate_job(job.job_id, translator="google")
            self.assertTrue(Path(job.pages[0].corrected_path).exists())

            manager.retranslate_job(job.job_id, translator="google", overwrite_manual=True)
            self.assertEqual(job.retranslate_pages, [0])
            self.assertFalse(Path(job.pages[0].corrected_path).exists())
            self.assertFalse(Path(job.pages[0].corrections_path).exists())

    def test_running_job_cannot_be_retranslated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, job = _manager_with_ready_job(tmp)
            job.status = "processing"
            with self.assertRaises(ValueError):
                manager.retranslate_job(job.job_id, translator="google")

    def test_interrupted_retranslation_is_requeued_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, job = _manager_with_ready_job(tmp)
            manager.retranslate_job(job.job_id, translator="llm")
            job.status = "processing"
            manager.manifests.save(job)

            restarted = JobManager(jobs_root=Path(tmp), start_worker=False)
            recovered = restarted.get_job(job.job_id)
            self.assertEqual(recovered.status, "queued")
            self.assertEqual(recovered.pending_operation, "retranslate")
            self.assertEqual(recovered.retranslate_pages, [0])
            self.assertEqual(restarted._queue.position(job.job_id), 1)

    def test_failed_retranslation_returns_job_to_terminal_state(self) -> None:
        class _BrokenRetranslator(_FakeRetranslator):
            def retranslate_page(self, **kwargs):
                raise RuntimeError("proveedor caído")

        with tempfile.TemporaryDirectory() as tmp:
            manager, job = _manager_with_ready_job(tmp)
            manager.retranslate_job(job.job_id, translator="google")
            runtime, assets = _offline_assets()
            with runtime, assets, mock.patch(f"{MODULE}.JobRetranslator", _BrokenRetranslator):
                manager._run_job(job.job_id)

            completed = manager.get_job(job.job_id)
            self.assertEqual(completed.status, "ready")
            self.assertEqual(completed.pending_operation, "process")
            self.assertIn("proveedor caído", completed.message)
            self.assertEqual(completed.pages[0].regions[0]["translated_text"], "hola")


class _DegradedRetranslator(_FakeRetranslator):
    """Simula que el LLM se quedó sin tokens y cayó al traductor tradicional."""

    llm_fallback_reason = "El LLM agotó su límite de tokens (429) y se cambió al traductor tradicional."


class RetranslationFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRetranslator.instances = []

    def test_silent_llm_fallback_stops_the_retranslation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, job = _manager_with_ready_job(tmp)
            page_two = PageState(
                index=1,
                source_filename="pagina2.png",
                output_filename="pagina2.png",
                status="ready",
                original_path=str(Path(job.input_dir) / "pagina2.png"),
                clean_path=str(Path(job.output_dir) / "limpieza" / "pagina2.png"),
                translated_path=str(Path(job.output_dir) / "traduccion" / "pagina2.png"),
                corrected_path=str(Path(job.output_dir) / "corregida" / "pagina2.png"),
                corrections_path=str(Path(job.output_dir) / "correcciones" / "pagina2.json"),
                regions=[_region()],
            )
            _write_image(Path(page_two.original_path), 30)
            _write_image(Path(page_two.clean_path), 240)
            _write_image(Path(page_two.translated_path), 120)
            job.pages.append(page_two)

            manager.retranslate_job(job.job_id, translator="llm")
            self.assertEqual(job.retranslate_pages, [0, 1])

            runtime, assets = _offline_assets()
            with runtime, assets, mock.patch(f"{MODULE}.JobRetranslator", _DegradedRetranslator):
                manager._run_job(job.job_id)

            completed = manager.get_job(job.job_id)
            # Se detiene tras la primera página y deja intacta la segunda.
            self.assertEqual(_FakeRetranslator.instances[0].calls, [0])
            self.assertIn("agotó su límite de tokens", completed.message)
            self.assertIn("página 1 de 2", completed.message)
            self.assertEqual(completed.status, "ready")
            self.assertEqual(completed.pending_operation, "process")
            self.assertEqual(completed.pages[1].regions[0]["translated_text"], "hola")

    def test_google_retranslation_ignores_the_llm_fallback_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, job = _manager_with_ready_job(tmp)
            manager.retranslate_job(job.job_id, translator="google")

            runtime, assets = _offline_assets()
            with runtime, assets, mock.patch(f"{MODULE}.JobRetranslator", _DegradedRetranslator):
                manager._run_job(job.job_id)

            completed = manager.get_job(job.job_id)
            self.assertIn("finalizada", completed.message)
            self.assertEqual(completed.pages[0].regions[0]["translated_text"], "retraducido:こんにちは")


class _StubTranslateManga:
    """Doble de `TranslateManga` con las mismas piezas que usa el retraductor."""

    def __init__(self, *args, **kwargs) -> None:
        self.indice_imagen = 0
        self.ultimas_regiones = []
        self.ultimas_asignaciones_hablante = []
        self.ultimo_estilos_texto = []
        self.ultimos_source_language_flags = []
        self.textos_recibidos = []
        self.reading_order_resolver = mock.Mock(page_reads_right_to_left=True)
        self.translator_manager = mock.Mock(provider=mock.Mock(llm_fallback_reason=""))
        self.text_renderer = mock.Mock()
        self.text_renderer.render_with_layouts.side_effect = lambda imagen, *a, **k: np.full_like(imagen, 77)

    def normalizar_texto_ocr(self, texto: str) -> str:
        return texto.strip()

    def traducir_textos(self, textos):
        self.textos_recibidos = list(textos)
        self.ultimo_estilos_texto = ["dialogo", "onomatopeya"][: len(textos)]
        self.ultimos_source_language_flags = [True] * len(textos)
        return [f"es:{texto}" for texto in textos]

    def resolver_textos_para_render(self, limpios, traducidos):
        # Regla real del pipeline: la onomatopeya conservada no se dibuja.
        return [
            "" if estilo == "onomatopeya" else traducido
            for estilo, traducido in zip(self.ultimo_estilos_texto, traducidos)
        ]


class RetranslatorPageTests(unittest.TestCase):
    def _config(self):
        return mock.Mock(
            translation=mock.Mock(
                idioma_entrada="Japonés",
                idioma_salida="Español",
                metodo_traduccion="LLM",
                groq_api_key="",
                lore_manga="",
            ),
            ocr=mock.Mock(),
            quality=mock.Mock(),
            onomatopoeia=mock.Mock(),
            character_memory=mock.Mock(),
        )

    def test_page_is_rendered_from_clean_image_and_returns_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "limpieza" / "pagina.png"
            translated = Path(tmp) / "traduccion" / "pagina.png"
            _write_image(clean, 250)
            _write_image(translated, 10)

            with mock.patch("parallel_manga_translator.ui.retranslator.TranslateManga", _StubTranslateManga):
                retranslator = JobRetranslator(self._config())
                resultados = retranslator.retranslate_page(
                    page_index=3,
                    clean_path=clean,
                    output_path=translated,
                    regions=[
                        _region(index=0, original="こんにちは"),
                        _region(index=1, original="ドン"),
                        {**_region(index=2, original="borrada"), "deleted": True},
                        {**_region(index=3), "original_text": "   "},
                    ],
                )

            stub = retranslator.translator
            self.assertEqual(stub.indice_imagen, 3)
            self.assertEqual(stub.textos_recibidos, ["こんにちは", "ドン"])
            self.assertEqual([resultado.index for resultado in resultados], [0, 1])
            self.assertEqual(resultados[0].translated_text, "es:こんにちは")
            self.assertEqual(resultados[1].rendered_text, "")

            # La imagen se rerenderiza a partir de la limpia, no de la traducción previa,
            # y reutiliza la geometría que el pipeline calculó con la máscara del globo.
            imagen, cajas, textos = stub.text_renderer.render_with_layouts.call_args[0]
            kwargs = stub.text_renderer.render_with_layouts.call_args.kwargs
            self.assertEqual(int(imagen[0, 0, 0]), 250)
            self.assertEqual(list(cajas[0]), [4, 4, 12, 10])
            # La onomatopeya conservada no se dibuja, así que no entra en el render.
            self.assertEqual(textos, ["es:こんにちは"])
            self.assertEqual(kwargs["ui_layouts"], [PIPELINE_LAYOUT])
            self.assertEqual(kwargs["text_styles"], ["dialogo"])
            self.assertEqual(int(cv2.imread(str(translated))[0, 0, 0]), 77)
            self.assertFalse(list(translated.parent.glob("*.tmp*")))

    def test_page_without_transcription_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "limpieza" / "pagina.png"
            _write_image(clean, 250)
            with mock.patch("parallel_manga_translator.ui.retranslator.TranslateManga", _StubTranslateManga):
                retranslator = JobRetranslator(self._config())
                with self.assertRaises(ValueError):
                    retranslator.retranslate_page(
                        page_index=0,
                        clean_path=clean,
                        output_path=Path(tmp) / "salida.png",
                        regions=[{**_region(), "original_text": ""}],
                    )


if __name__ == "__main__":
    unittest.main()
