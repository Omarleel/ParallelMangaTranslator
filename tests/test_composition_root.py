"""La dirección de las dependencias entre adaptadores de entrada.

CLI y UI son dos adaptadores del mismo dominio. Uno no debe colgar del otro: hasta ahora
`ui/job_manager.py` importaba `build_image_processor` y compañía de `cli.py`, de modo que
la web dependía del ejecutable de consola.

Eso tenía un efecto que no se veía leyendo el import: `cli.py` configuraba el proceso al
importarse —sink de loguru, `configure_logging()`, filtro de warnings y
`PYTORCH_CUDA_ALLOC_CONF`— y la UI heredaba todo eso de rebote, porque `ui/app.py`
construye un `JobManager` en tiempo de import. Reordenar los imports del CLI habría dejado
la web sin configurar sin que fallara nada.

Estos tests fijan la dirección, no el detalle: la composición vive en `bootstrap` y de ahí
dependen todos.
"""

import ast
import io
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE = RAIZ / "parallel_manga_translator"

#: Único sitio legítimo desde el que se importa el CLI: el script que lo lanza.
IMPORTADORES_PERMITIDOS = {"ParallelMangaTranslator.py"}

MODULO_CLI = "parallel_manga_translator.cli"
MODULO_BOOTSTRAP = "parallel_manga_translator.bootstrap"


def _modulos_importados(archivo: Path) -> set[str]:
    """Módulos que importa un fichero, incluidos los imports dentro de funciones."""
    importados: set[str] = set()
    arbol = ast.parse(io.open(archivo, encoding="utf-8").read())
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module:
            importados.add(nodo.module)
        elif isinstance(nodo, ast.Import):
            importados.update(alias.name for alias in nodo.names)
    return importados


def _ficheros_python():
    for archivo in RAIZ.rglob("*.py"):
        partes = set(archivo.parts)
        if partes & {".git", ".venv", "build", "dist", "__pycache__", ".codebase-memory"}:
            continue
        yield archivo


class RaizDeComposicionTests(unittest.TestCase):
    def test_solo_el_entrypoint_importa_el_cli(self):
        """Si esto falla, algo volvió a colgar del adaptador de consola."""
        culpables = []
        for archivo in _ficheros_python():
            if archivo.name in IMPORTADORES_PERMITIDOS:
                continue
            if MODULO_CLI in _modulos_importados(archivo):
                culpables.append(str(archivo.relative_to(RAIZ)))

        self.assertEqual(
            culpables,
            [],
            "Importan parallel_manga_translator.cli (usa bootstrap en su lugar): " f"{culpables}",
        )

    def test_la_ui_construye_desde_el_bootstrap(self):
        """No basta con no importar el CLI: tiene que depender de la composición."""
        importados = _modulos_importados(PAQUETE / "ui" / "job_manager.py")

        self.assertIn(MODULO_BOOTSTRAP, importados)

    def test_el_bootstrap_no_depende_de_ningun_adaptador(self):
        """Si el composition root importara un adaptador, el ciclo volvería por detrás."""
        importados = _modulos_importados(PAQUETE / "bootstrap.py")
        adaptadores = [m for m in importados if m.endswith(".cli") or ".ui" in m]

        self.assertEqual(adaptadores, [], f"bootstrap depende de adaptadores: {adaptadores}")

    def test_el_cli_no_reexporta_la_composicion(self):
        """Reexportarla dejaría viva la puerta de atrás que se acaba de cerrar."""
        cli = PAQUETE / "cli.py"
        arbol = ast.parse(io.open(cli, encoding="utf-8").read())
        publicos = [
            nodo.name
            for nodo in arbol.body
            if isinstance(nodo, ast.FunctionDef) and not nodo.name.startswith("_")
        ]

        self.assertEqual(sorted(publicos), ["main", "run_pipeline"])


if __name__ == "__main__":
    unittest.main()
