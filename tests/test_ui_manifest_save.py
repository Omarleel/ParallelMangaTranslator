"""Guardado del manifiesto de la UI frente a antivirus y a hilos concurrentes.

En Windows, el escáner on-access del antivirus abre los archivos recién escritos.
Mientras sostiene el handle, `Path.replace` sobre el destino devuelve
`PermissionError: [WinError 5]`. Es transitorio, pero perder el guardado abortaba el
trabajo entero.
"""

import ast
import io
import json
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from parallel_manga_translator.ui.job_manager import (
    _MANIFEST_REPLACE_RETRIES,
    JobManager,
    JobState,
)

JOB_MANAGER_SOURCE = Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui" / "job_manager.py"


def _job(root: Path) -> JobState:
    return JobState(
        job_id="test",
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
        self.root = Path(tempfile.mkdtemp(prefix="pmt-manifest-"))

    def tearDown(self):
        # El mismo antivirus puede sostener handles al limpiar; no es lo que se prueba.
        shutil.rmtree(self.root, ignore_errors=True)

    def test_un_bloqueo_transitorio_no_pierde_el_guardado(self):
        job = _job(self.root)
        falso, marcador = _replace_que_falla(fallos=3)

        with unittest.mock.patch.object(Path, "replace", falso):
            JobManager._save_manifest(None, job)

        self.assertEqual(marcador["intentos"], 4)
        guardado = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(guardado["job_id"], "test")
        # El temporal no debe quedar huérfano tras un guardado correcto.
        self.assertFalse((self.root / "manifest.json.tmp").exists())

    def test_un_bloqueo_persistente_sigue_propagando_el_error(self):
        """Callar el fallo sería peor: el manifiesto es la fuente de verdad del trabajo."""
        job = _job(self.root)
        falso, marcador = _replace_que_falla(fallos=_MANIFEST_REPLACE_RETRIES)

        with unittest.mock.patch.object(Path, "replace", falso):
            with self.assertRaises(PermissionError):
                JobManager._save_manifest(None, job)

        self.assertEqual(marcador["intentos"], _MANIFEST_REPLACE_RETRIES)

    def test_todo_guardado_del_manifiesto_ocurre_bajo_el_lock(self):
        """El temporal tiene nombre fijo por trabajo.

        Dos hilos guardando el mismo job escribirían sobre el mismo archivo y uno
        podría tenerlo abierto mientras el otro lo renombra: el mismo WinError 5, esta
        vez sin antivirus de por medio.
        """
        arbol = ast.parse(io.open(JOB_MANAGER_SOURCE, encoding="utf-8").read())
        sin_lock = []

        def recorrer(nodo, bajo_lock, funcion):
            for hijo in ast.iter_child_nodes(nodo):
                dentro, nombre = bajo_lock, funcion
                if isinstance(hijo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dentro, nombre = False, hijo.name
                if isinstance(hijo, ast.With):
                    if any("_lock" in ast.unparse(item.context_expr) for item in hijo.items):
                        dentro = True
                if (
                    isinstance(hijo, ast.Call)
                    and isinstance(hijo.func, ast.Attribute)
                    and hijo.func.attr == "_save_manifest"
                    and not bajo_lock
                ):
                    sin_lock.append(f"{funcion} (línea {hijo.lineno})")
                recorrer(hijo, dentro, nombre)

        recorrer(arbol, False, "<módulo>")
        self.assertEqual(sin_lock, [], f"_save_manifest fuera de self._lock en: {sin_lock}")


if __name__ == "__main__":
    unittest.main()
