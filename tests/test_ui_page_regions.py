"""Reglas de precedencia al componer las regiones que ve el editor.

Tres fuentes se pisan por capas: la transcripción del pipeline, su traducción, y lo que el
usuario corrigió a mano. Cuál gana en cada campo es el contrato real de este módulo, y
hasta ahora vivía dentro de `JobManager` como métodos privados: para ejercitarlo había que
construir el manager entero, que arranca un worker y toca disco, así que en la práctica no
se probaba.
"""

import unittest
from dataclasses import dataclass
from typing import Tuple

from parallel_manga_translator.ui.job_state import JobState, PageState
from parallel_manga_translator.ui.page_regions import (
    apply_saved_corrections,
    coords_to_bbox,
    merge_page_regions,
    push_retranslated_page,
    read_json,
)
from parallel_manga_translator.ui.queue_adapter import CapturingJsonQueue


@dataclass
class _Resultado:
    """La forma que `JobRetranslator` devuelve por región."""

    index: int
    bbox: Tuple[int, int, int, int]
    translated_text: str
    style: str = "dialogo"
    source_language_ok: bool = True


def _job() -> JobState:
    return JobState(job_id="j", title="t", root_dir="/r", input_dir="/r/in", output_dir="/r/out")


def _page(index: int = 0) -> PageState:
    return PageState(index=index, source_filename="a.png", output_filename="0001.png")


def _transcripcion(*globos):
    return {"Transcripción": [{"Página": 1, "Globos de texto": list(globos)}]}


def _traduccion(*globos):
    return {"Traducción": [{"Página": 1, "Globos de texto": list(globos)}]}


class FusionDeTranscripcionYTraduccionTests(unittest.TestCase):
    def test_la_traduccion_pisa_texto_y_estilo_de_la_transcripcion(self):
        regiones = merge_page_regions(
            _job(),
            _page(),
            trans_data=_transcripcion(
                {"Índice": 0, "Coordenadas": [[10, 20], [50, 60]], "Texto": "こんにちは", "Estilo": "dialogo"}
            ),
            trad_data=_traduccion({"Índice": 0, "Coordenadas": [[10, 20], [50, 60]], "Texto": "Hola", "Estilo": "grito"}),
        )

        self.assertEqual(len(regiones), 1)
        self.assertEqual(regiones[0]["original_text"], "こんにちは")
        self.assertEqual(regiones[0]["translated_text"], "Hola")
        self.assertEqual(regiones[0]["style"], "grito")

    def test_se_funden_por_indice_no_por_posicion(self):
        """Si el pipeline descarta una región, los índices dejan de ser correlativos."""
        regiones = merge_page_regions(
            _job(),
            _page(),
            trans_data=_transcripcion(
                {"Índice": 0, "Coordenadas": [[0, 0], [10, 10]], "Texto": "uno"},
                {"Índice": 7, "Coordenadas": [[0, 0], [10, 10]], "Texto": "siete"},
            ),
            trad_data=_traduccion({"Índice": 7, "Coordenadas": [[0, 0], [10, 10]], "Texto": "seven"}),
        )

        por_indice = {r["index"]: r for r in regiones}
        self.assertEqual(por_indice[7]["translated_text"], "seven")
        self.assertNotIn("translated_text", por_indice[0])

    def test_las_regiones_salen_ordenadas_por_indice(self):
        regiones = merge_page_regions(
            _job(),
            _page(),
            trans_data=_transcripcion(
                {"Índice": 5, "Coordenadas": [[0, 0], [1, 1]], "Texto": "b"},
                {"Índice": 2, "Coordenadas": [[0, 0], [1, 1]], "Texto": "a"},
            ),
            trad_data=_traduccion(),
        )

        self.assertEqual([r["index"] for r in regiones], [2, 5])

    def test_una_pagina_sin_datos_no_revienta(self):
        self.assertEqual(merge_page_regions(_job(), _page(), trans_data={}, trad_data={}), [])


class CorreccionesManualesTests(unittest.TestCase):
    def test_la_correccion_manual_gana_sobre_el_pipeline(self):
        base = [{"index": 0, "bbox": [1, 2, 3, 4], "translated_text": "auto", "style": "dialogo"}]

        fusionadas = apply_saved_corrections(base, [{"index": 0, "text": "a mano", "style": "grito"}])

        self.assertEqual(fusionadas[0]["translated_text"], "a mano")
        self.assertEqual(fusionadas[0]["style"], "grito")

    def test_una_region_creada_por_el_usuario_se_anade(self):
        """No existe en el pipeline: si no se añadiera, el usuario perdería su trabajo."""
        base = [{"index": 0, "bbox": [1, 2, 3, 4], "translated_text": "auto"}]

        fusionadas = apply_saved_corrections(base, [{"index": 9, "text": "nueva", "bbox": [5, 5, 5, 5]}])

        self.assertEqual(len(fusionadas), 2)
        anadida = fusionadas[-1]
        self.assertEqual(anadida["index"], 9)
        self.assertTrue(anadida["manual"])
        self.assertEqual(anadida["translated_text"], "nueva")

    def test_una_region_sin_correccion_conserva_su_source_bbox(self):
        """`source_bbox` es de dónde venía: sin él, mover dos veces deja un fantasma."""
        base = [{"index": 0, "bbox": [1, 2, 3, 4], "translated_text": "auto"}]

        fusionadas = apply_saved_corrections(base, [])

        self.assertEqual(fusionadas[0]["source_bbox"], [1, 2, 3, 4])
        self.assertFalse(fusionadas[0]["modified"])


class ReescrituraDeUnaPaginaRetraducidaTests(unittest.TestCase):
    def test_solo_cambian_texto_y_estilo(self):
        """El resto del globo se arrastra: retraducir no debe perder ajustes de la UI."""
        cola = CapturingJsonQueue()
        cola.put({
            "establecer_elemento_en_lista": {
                "Traducción": {
                    "Página": 1,
                    "Globos de texto": [
                        {
                            "Índice": 0,
                            "Coordenadas": [[10, 20], [50, 60]],
                            "Texto": "viejo",
                            "Estilo": "dialogo",
                            "Layout UI": {"font_size": 22},
                        }
                    ],
                }
            }
        })

        push_retranslated_page(cola, _page(), [_Resultado(0, (10, 20, 40, 40), "nuevo", "grito")])

        globo = cola.data["Traducción"][0]["Globos de texto"][0]
        self.assertEqual(globo["Texto"], "nuevo")
        self.assertEqual(globo["Estilo"], "grito")
        self.assertEqual(globo["Layout UI"], {"font_size": 22})
        self.assertEqual(globo["Coordenadas"], [[10, 20], [50, 60]])

    def test_lo_que_descarto_el_filtro_de_idioma_no_se_publica(self):
        """El pipeline tampoco lo publica; retraducir no debe reintroducirlo."""
        cola = CapturingJsonQueue()

        push_retranslated_page(
            cola,
            _page(),
            [_Resultado(0, (0, 0, 1, 1), "sí"), _Resultado(1, (0, 0, 1, 1), "no", source_language_ok=False)],
        )

        globos = cola.data["Traducción"][0]["Globos de texto"]
        self.assertEqual([g["Índice"] for g in globos], [0])

    def test_una_region_nueva_se_crea_con_sus_coordenadas(self):
        cola = CapturingJsonQueue()

        push_retranslated_page(cola, _page(), [_Resultado(3, (5, 6, 7, 8), "texto")])

        globo = cola.data["Traducción"][0]["Globos de texto"][0]
        self.assertEqual(globo["Coordenadas"], [[5, 6], [12, 14]])


class HelpersTests(unittest.TestCase):
    def test_coordenadas_malformadas_dan_una_caja_minima(self):
        """Un JSON a medias no debe tumbar la página entera del editor."""
        self.assertEqual(coords_to_bbox(None), [0, 0, 1, 1])
        self.assertEqual(coords_to_bbox([[1, 2]]), [0, 0, 1, 1])

    def test_una_caja_degenerada_conserva_ancho_y_alto_minimos(self):
        self.assertEqual(coords_to_bbox([[10, 10], [10, 10]]), [10, 10, 1, 1])

    def test_un_json_ausente_o_corrupto_se_lee_como_vacio(self, ):
        from pathlib import Path
        import tempfile

        self.assertEqual(read_json(Path("no-existe.json")), {})
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            fh.write("{esto no es json")
            roto = Path(fh.name)
        try:
            self.assertEqual(read_json(roto), {})
        finally:
            roto.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
