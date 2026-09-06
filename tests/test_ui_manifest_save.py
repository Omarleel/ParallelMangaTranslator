"""`JobManifestStore`: el manifiesto de un trabajo frente a antivirus e hilos.

En Windows, el escáner on-access del antivirus abre los archivos recién escritos.
Mientras sostiene el handle, `Path.replace` sobre el destino devuelve
`PermissionError: [WinError 5]`. Es transitorio, pero perder el guardado abortaba el
trabajo entero.

Esto vivía en `JobManager` como cuatro métodos privados con 34 sitios de llamada, y la
atomicidad dependía de que cada uno de esos sitios tomara el lock del manager. Ahora es un
objeto con su propia garantía, y se puede probar sin construir el manager.
"""

import ast
import io
import json
import shutil
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

from parallel_manga_translator.ui.job_manifest_store import (
    MANIFEST_REPLACE_RETRIES,
    JobManifestStore,
)
from parallel_manga_translator.ui.job_state import JobState, PageState

JOB_MANAGER_SOURCE = Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui" / "job_manager.py"


def _job(root: Path, job_id: str = "test") -> JobState:
    return JobState(
        job_id=job_id,
        title="Trabajo de prueba",
        root_dir=str(root),
        input_dir=str(root / "entrada"),
        output_dir=str(root / "outputs"),
    )


def _replace_que_falla(fallos):
    """Sustituto de `Path.replace` que falla las primeras `fallos` veces.

    Tiene que ser una función, no un objeto invocable: al asignarla a la clase, es el
    protocolo de descriptor el que enlaza el `self` de la ruta de origen.
    """
    real = Path.replace
    marcador = {"intentos": 0}

    def replace(self, destino):
        marcador["intentos"] += 1
        if marcador["intentos"] <= fallos:
            raise PermissionError(5, "Acceso denegado")
        return real(self, destino)

    return replace, marcador


class GuardadoDeManifiestoTests(unittest.TestCase):
    def setUp(self):
        self.raiz_trabajos = Path(tempfile.mkdtemp(prefix="pmt-manifest-"))
        self.root = self.raiz_trabajos / "test"
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = JobManifestStore(self.raiz_trabajos)

    def tearDown(self):
        # El mismo antivirus puede sostener handles al limpiar; no es lo que se prueba.
        shutil.rmtree(self.raiz_trabajos, ignore_errors=True)

    def test_un_bloqueo_transitorio_no_pierde_el_guardado(self):
        job = _job(self.root)
        falso, marcador = _replace_que_falla(fallos=3)

        with unittest.mock.patch.object(Path, "replace", falso):
            self.store.save(job)

        self.assertEqual(marcador["intentos"], 4)
        guardado = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(guardado["job_id"], "test")
        # El temporal no debe quedar huérfano tras un guardado correcto.
        self.assertFalse((self.root / "manifest.json.tmp").exists())

    def test_un_bloqueo_persistente_sigue_propagando_el_error(self):
        """Callar el fallo sería peor: el manifiesto es la fuente de verdad del trabajo."""
        job = _job(self.root)
        falso, marcador = _replace_que_falla(fallos=MANIFEST_REPLACE_RETRIES)

        with unittest.mock.patch.object(Path, "replace", falso), self.assertRaises(PermissionError):
            self.store.save(job)

        self.assertEqual(marcador["intentos"], MANIFEST_REPLACE_RETRIES)

    def test_dos_hilos_guardando_el_mismo_trabajo_no_se_solapan(self):
        """El temporal tiene nombre fijo por trabajo.

        Dos hilos guardando el mismo job escribirían sobre el mismo archivo y uno podría
        tenerlo abierto mientras el otro lo renombra: el mismo WinError 5, esta vez sin
        antivirus de por medio. Antes eso dependía de que los 34 sitios de llamada
        tomaran el lock del manager; ahora lo garantiza el store.
        """
        job = _job(self.root)
        en_vuelo = {"actual": 0, "maximo": 0}
        contador = threading.Lock()
        real_write = Path.write_text

        def write_text(self, *args, **kwargs):
            with contador:
                en_vuelo["actual"] += 1
                en_vuelo["maximo"] = max(en_vuelo["maximo"], en_vuelo["actual"])
            try:
                time.sleep(0.01)  # ensancha la ventana de solape
                return real_write(self, *args, **kwargs)
            finally:
                with contador:
                    en_vuelo["actual"] -= 1

        with unittest.mock.patch.object(Path, "write_text", write_text):
            hilos = [threading.Thread(target=self.store.save, args=(job,)) for _ in range(4)]
            for hilo in hilos:
                hilo.start()
            for hilo in hilos:
                hilo.join()

        self.assertEqual(en_vuelo["maximo"], 1, "Dos guardados entraron a la vez al temporal")

    def test_todo_guardado_del_manifiesto_ocurre_bajo_el_lock_del_manager(self):
        """El lock del store cubre el archivo; el del manager, el objeto que se serializa.

        `asdict(job)` recorre el `JobState` entero: si otro hilo muta `job.pages` a la vez,
        el manifiesto sale a medias. Por eso el manager sigue teniendo que guardar bajo su
        propio lock, y esto lo comprueba.
        """
        arbol = ast.parse(io.open(JOB_MANAGER_SOURCE, encoding="utf-8").read())
        sin_lock = []

        def es_guardado(nodo):
            return (
                isinstance(nodo, ast.Call)
                and isinstance(nodo.func, ast.Attribute)
                and nodo.func.attr == "save"
                and isinstance(nodo.func.value, ast.Attribute)
                and nodo.func.value.attr == "manifests"
            )

        def recorrer(nodo, bajo_lock, funcion):
            for hijo in ast.iter_child_nodes(nodo):
                dentro, nombre = bajo_lock, funcion
                if isinstance(hijo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dentro, nombre = False, hijo.name
                if isinstance(hijo, ast.With) and any(
                    "_lock" in ast.unparse(item.context_expr) for item in hijo.items
                ):
                    dentro = True
                if es_guardado(hijo) and not bajo_lock:
                    sin_lock.append(f"{funcion} (línea {hijo.lineno})")
                recorrer(hijo, dentro, nombre)

        recorrer(arbol, False, "<módulo>")
        self.assertEqual(sin_lock, [], f"manifests.save fuera de self._lock en: {sin_lock}")


class CargaDeManifiestoTests(unittest.TestCase):
    """La carga y el reanclado de rutas, que antes no se podían probar sin el manager."""

    def setUp(self):
        self.raiz_trabajos = Path(tempfile.mkdtemp(prefix="pmt-manifest-load-"))
        self.store = JobManifestStore(self.raiz_trabajos)

    def tearDown(self):
        shutil.rmtree(self.raiz_trabajos, ignore_errors=True)

    def test_ida_y_vuelta_conserva_el_trabajo(self):
        root = self.raiz_trabajos / "job1"
        root.mkdir(parents=True)
        job = _job(root, job_id="job1")
        job.pages = [PageState(index=0, source_filename="a.png", output_filename="0001.png")]

        self.store.save(job)
        recuperado = self.store.load("job1")

        self.assertEqual(recuperado.job_id, "job1")
        self.assertEqual(len(recuperado.pages), 1)
        self.assertEqual(recuperado.pages[0].output_filename, "0001.png")

    def test_un_trabajo_movido_de_carpeta_se_reancla(self):
        """Copiar `.pmt_ui_jobs` a otra máquina o ruta no debe romper el trabajo."""
        root = self.raiz_trabajos / "job2"
        root.mkdir(parents=True)
        job = _job(root, job_id="job2")
        job.root_dir = "D:/ruta/vieja/job2"
        job.input_dir = "D:/ruta/vieja/job2/entrada"
        job.pages = [
            PageState(
                index=0,
                source_filename="a.png",
                output_filename="0001.png",
                original_path="D:/ruta/vieja/job2/entrada/a.png",
            )
        ]
        # Se escribe a mano en el sitio nuevo: `save` usa `job.root_dir`, que aún miente.
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "job_id": "job2",
                    "title": job.title,
                    "root_dir": job.root_dir,
                    "input_dir": job.input_dir,
                    "output_dir": job.output_dir,
                    "pages": [
                        {
                            "index": 0,
                            "source_filename": "a.png",
                            "output_filename": "0001.png",
                            "original_path": "D:/ruta/vieja/job2/entrada/a.png",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        recuperado = self.store.load("job2")

        self.assertEqual(Path(recuperado.root_dir), root)
        self.assertNotIn("vieja", recuperado.input_dir)
        self.assertNotIn("vieja", recuperado.pages[0].original_path)

    def test_un_manifiesto_de_una_version_anterior_se_puede_abrir(self):
        """Claves que ya no existen no deben impedir abrir un trabajo antiguo."""
        root = self.raiz_trabajos / "job3"
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "job_id": "job3",
                    "title": "Antiguo",
                    "root_dir": str(root),
                    "input_dir": str(root / "entrada"),
                    "output_dir": str(root / "outputs"),
                    "campo_que_ya_no_existe": 42,
                    "pages": [
                        {
                            "index": 0,
                            "source_filename": "a.png",
                            "output_filename": "0001.png",
                            "otro_campo_muerto": True,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        recuperado = self.store.load("job3")

        self.assertEqual(recuperado.job_id, "job3")
        self.assertEqual(len(recuperado.pages), 1)

    def test_un_trabajo_inexistente_lo_dice(self):
        with self.assertRaises(FileNotFoundError):
            self.store.load("no-existe")


if __name__ == "__main__":
    unittest.main()
