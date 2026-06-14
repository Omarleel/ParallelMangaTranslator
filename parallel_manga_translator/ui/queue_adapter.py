from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict

from parallel_manga_translator.io.json_generator import JsonGenerator


class CapturingJsonQueue:
    """Adaptador síncrono con la misma API mínima que usan los JsonWriter.

    El pipeline existente publica eventos en colas multiprocessing. Para la UI local
    necesitamos procesar página por página y consultar el JSON inmediatamente, así
    que este adaptador aplica los mensajes en memoria y escribe checkpoints atómicos.
    """

    def __init__(self, output_path: str | Path | None = None) -> None:
        self.json_generator = JsonGenerator()
        self.output_path = Path(output_path) if output_path else None
        self._lock = Lock()

    @property
    def data(self) -> Dict[str, Any]:
        with self._lock:
            return self.json_generator.datos

    def put(self, data: Dict[str, Any]) -> None:
        with self._lock:
            if not data:
                return
            metodo = next(iter(data))
            payload = data[metodo]

            if metodo == "agregar_entrada":
                for clave, valor in payload.items():
                    self.json_generator.agregar_entrada(clave=clave, valor=valor)
            elif metodo == "agregar_elemento_a_lista":
                for clave, elemento in payload.items():
                    self.json_generator.agregar_elemento_a_lista(clave=clave, elemento=elemento)
            elif metodo == "agregar_a_sublista":
                self.json_generator.agregar_a_sublista(
                    clave_lista=payload["clave_lista"],
                    pagina=payload["pagina"],
                    clave_sublista=payload["clave_sublista"],
                    elemento_sublista=payload["elemento_sublista"],
                )
            elif metodo == "ordenar_por_paginas":
                self.json_generator.ordenar_por_paginas(tipo=payload["tipo"])
            elif metodo == "guardar_en_archivo":
                if isinstance(payload, str):
                    self.output_path = Path(payload)
                elif isinstance(payload, dict) and payload.get("path"):
                    self.output_path = Path(payload["path"])
            else:
                return
            self.dump()

    def dump(self) -> None:
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        self.json_generator.guardar_en_archivo(str(tmp_path))
        tmp_path.replace(self.output_path)
