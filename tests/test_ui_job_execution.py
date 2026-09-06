"""Las fronteras entre el coordinador y los dos runners.

`JobManager` tenía la ejecución dentro: `_run_job` (226 líneas) y `_run_retranslation`
(136) convivían con la cola, el ciclo de vida y la edición manual. Medido por AST, ambos
necesitaban **exactamente las mismas trece cosas** del manager, que es la señal de que
compartían un andamiaje y no de que fueran inseparables de él.

Estos tests fijan el reparto resultante, no su implementación:

- el manager compone dos runners sobre un único `JobExecutionSupport`;
- `_run_job` sólo despacha;
- los runners no vuelven a mirar al manager (si lo hicieran, el ciclo regresa);
- el andamiaje resuelve `config_for_job` en cada llamada, que es lo que permite que
  sustituirlo sobre el manager siga surtiendo efecto.
"""

import ast
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from parallel_manga_translator.ui.job_execution import JobExecutionSupport
from parallel_manga_translator.ui.job_manager import JobManager
from parallel_manga_translator.ui.job_runner import JobRunner
from parallel_manga_translator.ui.job_state import JobState
from parallel_manga_translator.ui.retranslation_runner import RetranslationRunner

UI = Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui"


def _job(root: Path, **kwargs) -> JobState:
    base = {
        "job_id": "j1",
        "title": "t",
        "root_dir": str(root),
        "input_dir": str(root / "in"),
        "output_dir": str(root / "out"),
    }
    base.update(kwargs)
    return JobState(**base)


class _ColaFalsa:
    def __init__(self):
        self.encolados = []

    def enqueue(self, job_id, preserve_time=False):
        self.encolados.append((job_id, preserve_time))

    def remove(self, job_id):
        pass


class _ManifiestosFalsos:
    def __init__(self):
        self.guardados = 0

    def save(self, job):
        self.guardados += 1


class _ControlFalso:
    def __init__(self, cancel_requested=False):
        self.cancel_requested = cancel_requested


class ComposicionDeRunnersTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="pmt-exec-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _manager(self) -> JobManager:
        return JobManager(jobs_root=self.root / "jobs", start_worker=False)

    def test_el_manager_compone_los_dos_runners(self):
        manager = self._manager()

        self.assertIsInstance(manager.job_runner, JobRunner)
        self.assertIsInstance(manager.retranslation_runner, RetranslationRunner)

    def test_los_dos_runners_comparten_un_unico_andamiaje(self):
        """Si no lo compartieran habría dos locks de procesamiento y dos registros de
        controles: dos trabajos podrían correr a la vez y pausar dejaría de encontrarlos."""
        manager = self._manager()

        self.assertIs(manager.job_runner._support, manager.retranslation_runner._support)
        self.assertIs(manager.job_runner._support, manager._execution)

    def test_el_manager_ya_no_guarda_el_estado_de_ejecucion(self):
        """Ese estado es del andamiaje; duplicarlo es cómo se desincronizan las cosas."""
        manager = self._manager()

        self.assertFalse(hasattr(manager, "_processing_lock"))
        self.assertFalse(hasattr(manager, "_assets_prepared"))
        self.assertIs(manager._execution.controls, manager._execution.controls)

    def test_run_job_solo_despacha(self):
        """El punto de entrada del worker elige runner y nada más."""
        manager = self._manager()
        llamadas = []
        manager.job_runner.run = lambda job_id: llamadas.append(("normal", job_id))
        manager.retranslation_runner.run = lambda job_id: llamadas.append(("retraduccion", job_id))

        normal = _job(self.root, job_id="a")
        retra = _job(self.root, job_id="b", pending_operation="retranslate")
        manager._jobs["a"], manager._jobs["b"] = normal, retra
        manager.get_job = lambda job_id: manager._jobs[job_id]

        manager._run_job("a")
        manager._run_job("b")

        self.assertEqual(llamadas, [("normal", "a"), ("retraduccion", "b")])

    def test_la_configuracion_del_trabajo_se_resuelve_en_cada_llamada(self):
        """Se inyecta como invocable, no como método enlazado.

        Los tests sustituyen `manager._build_config_for_job` para no leer `config.yaml`.
        Si el andamiaje hubiera capturado el método al construirse, esa sustitución
        dejaría de surtir efecto y los tests medirían la configuración real sin avisar.
        """
        manager = self._manager()
        centinela = object()
        manager._build_config_for_job = lambda job: centinela

        self.assertIs(manager._execution.config_for_job(_job(self.root)), centinela)


class AndamiajeDeEjecucionTests(unittest.TestCase):
    """`JobExecutionSupport` por separado: no hace falta un manager para probarlo."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="pmt-scope-"))
        self.job = _job(self.root)
        self.cola = _ColaFalsa()
        self.manifiestos = _ManifiestosFalsos()
        self.despertados = []
        self.support = JobExecutionSupport(
            get_job=lambda job_id: self.job,
            manifests=self.manifiestos,
            state_lock=threading.RLock(),
            queue=self.cola,
            wake_worker=lambda: self.despertados.append(True),
            config_for_job=lambda job: None,
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_executing_registra_el_control_y_lo_retira_al_salir(self):
        with self.support.executing(self.job, message="en marcha") as control:
            self.assertIs(self.support.controls[self.job.job_id], control)
            self.assertEqual(self.job.status, "processing")
            self.assertEqual(self.job.message, "en marcha")

        self.assertNotIn(self.job.job_id, self.support.controls)

    def test_el_control_se_retira_aunque_la_ejecucion_reviente(self):
        """Sin esto, un fallo dejaría el trabajo imposible de pausar o cancelar."""
        with self.assertRaises(RuntimeError), self.support.executing(self.job, message="x"):
            raise RuntimeError("boom")

        self.assertNotIn(self.job.job_id, self.support.controls)

    def test_mark_started_solo_marca_el_inicio_cuando_se_pide(self):
        """Procesar marca `started_at`; retraducir no, porque el trabajo ya empezó."""
        with self.support.executing(self.job, message="x"):
            pass
        self.assertEqual(self.job.started_at, 0.0)

        with self.support.executing(self.job, message="x", mark_started=True):
            pass
        self.assertGreater(self.job.started_at, 0.0)

    def test_una_pausa_con_cancelacion_pendiente_delega_el_cierre(self):
        self.job.cancel_requested = True
        cerrados = []

        self.support.handle_pause(
            self.job,
            _ControlFalso(),
            on_cancelled=cerrados.append,
            resume_message="r",
            paused_message="p",
        )

        self.assertEqual(cerrados, [self.job])
        self.assertEqual(self.cola.encolados, [])

    def test_una_pausa_con_reanudacion_pendiente_reencola_y_despierta_al_worker(self):
        self.job.resume_requested = True

        self.support.handle_pause(
            self.job,
            _ControlFalso(),
            on_cancelled=lambda job: self.fail("no debía cerrarse"),
            resume_message="reanudado",
            paused_message="p",
        )

        self.assertEqual(self.job.status, "queued")
        self.assertEqual(self.job.message, "reanudado")
        self.assertEqual(self.cola.encolados, [(self.job.job_id, False)])
        self.assertEqual(self.despertados, [True])

    def test_una_pausa_normal_deja_el_trabajo_pausado_y_no_lo_reencola(self):
        self.support.handle_pause(
            self.job,
            _ControlFalso(),
            on_cancelled=lambda job: self.fail("no debía cerrarse"),
            resume_message="r",
            paused_message="pausado",
        )

        self.assertEqual(self.job.status, "paused")
        self.assertEqual(self.job.message, "pausado")
        self.assertTrue(self.job.pause_requested)
        self.assertEqual(self.cola.encolados, [])
        self.assertEqual(self.despertados, [])


class DireccionDeDependenciasTests(unittest.TestCase):
    """Las fronteras sólo se sostienen si nadie las cruza de vuelta."""

    @staticmethod
    def _importa(archivo: Path) -> set:
        modulos = set()
        for nodo in ast.walk(ast.parse(archivo.read_text(encoding="utf-8"))):
            if isinstance(nodo, ast.ImportFrom) and nodo.module:
                modulos.add(nodo.module)
            elif isinstance(nodo, ast.Import):
                modulos.update(alias.name for alias in nodo.names)
        return modulos

    def test_los_runners_no_dependen_del_manager(self):
        """Un import de vuelta reintroduce el ciclo y anula la extracción."""
        for nombre in ("job_runner.py", "retranslation_runner.py", "job_execution.py"):
            with self.subTest(modulo=nombre):
                self.assertNotIn(
                    "parallel_manga_translator.ui.job_manager", self._importa(UI / nombre)
                )

    def test_el_nucleo_no_depende_de_la_ui(self):
        """`ui` es un adaptador: el pipeline y la configuración no deben conocerlo."""
        raiz = UI.parent
        culpables = []
        for archivo in raiz.rglob("*.py"):
            if "ui" in archivo.relative_to(raiz).parts:
                continue
            for modulo in self._importa(archivo):
                if ".ui." in f".{modulo}." :
                    culpables.append(f"{archivo.relative_to(raiz)} -> {modulo}")

        self.assertEqual(culpables, [], f"El núcleo importa de la UI: {culpables}")

    def test_el_manager_ya_no_contiene_la_ejecucion(self):
        """Si vuelve a aparecer aquí, el reparto se deshizo."""
        fuente = (UI / "job_manager.py").read_text(encoding="utf-8")

        for senal in ("execution_control_scope", "build_image_processor", "JobRetranslator"):
            with self.subTest(senal=senal):
                self.assertNotIn(senal, fuente)


if __name__ == "__main__":
    unittest.main()
