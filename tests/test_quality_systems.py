import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

sys.modules.setdefault("easyocr", types.SimpleNamespace(Reader=object))

from Applications.CharacterMemoryManager import CharacterMemoryManager, validate_character_memory_response
from Applications.EvaluationManager import EvaluationManager, EvaluationConfig, box_iou, char_error_rate
from Applications.TranslatorManager import validate_translation_response
from Applications.BubbleDetector import BubbleDetector
from Applications.CleanManga import CleanManga
from Applications.OnomatopoeiaManager import OnomatopoeiaManager
from Applications.ProcessingModels import TextRegion
from Applications.SourceLanguageFilter import SourceLanguageFilter
from Applications.TranslateManga import TranslateManga


class StrictLLMJsonTests(unittest.TestCase):
    def test_translation_response_rejects_extra_fields_and_missing_ids(self):
        valid = {"traducciones": [{"id": 0, "traduccion": "Hola"}, {"id": 1, "traduccion": "Vamos"}]}
        self.assertEqual(len(validate_translation_response(valid, [0, 1])), 2)

        with self.assertRaises(ValueError):
            validate_translation_response({"traducciones": [{"id": 0, "traduccion": "Hola", "nota": "extra"}]}, [0])

        with self.assertRaises(ValueError):
            validate_translation_response({"traducciones": [{"id": 0, "traduccion": "Hola"}]}, [0, 1])


class EvaluationManagerTests(unittest.TestCase):
    def test_iou_and_cer_are_measured(self):
        self.assertGreater(box_iou((10, 10, 100, 50), (15, 12, 98, 48)), 0.80)
        self.assertAlmostEqual(char_error_rate("行くぞ", "行くそ"), 1 / 3, places=3)

    def test_evaluate_dataset_from_pipeline_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / "ground_truth"
            gt_dir.mkdir()
            (gt_dir / "0001.json").write_text(json.dumps({
                "page": "0001.png",
                "regions": [
                    {"bbox": [10, 10, 100, 50], "text_ja": "行くぞ", "translation_es": "¡Vamos!"},
                    {"bbox": [200, 20, 80, 40], "text_ja": "待て", "translation_es": "Espera"},
                ],
            }, ensure_ascii=False), encoding="utf-8")

            transcription = root / "Transcripción.json"
            transcription.write_text(json.dumps({
                "Transcripción": [{
                    "Página": 1,
                    "Globos de texto": [
                        {"Coordenadas": [[11, 11], [110, 60]], "Texto": "行くぞ"},
                        {"Coordenadas": [[200, 20], [280, 60]], "Texto": "待て"},
                    ],
                }]
            }, ensure_ascii=False), encoding="utf-8")
            translation = root / "Traducción.json"
            translation.write_text(json.dumps({
                "Traducción": [{
                    "Página": 1,
                    "Globos de texto": [
                        {"Coordenadas": [[11, 11], [110, 60]], "Texto": "¡Vamos!"},
                        {"Coordenadas": [[200, 20], [280, 60]], "Texto": "Espera"},
                    ],
                }]
            }, ensure_ascii=False), encoding="utf-8")

            report = EvaluationManager(EvaluationConfig(iou_threshold=0.5)).evaluate_dataset(
                gt_dir,
                transcription_json=transcription,
                translation_json=translation,
            )

        self.assertEqual(report["summary"]["matched_regions"], 2)
        self.assertEqual(report["summary"]["detection_f1"], 1.0)
        self.assertEqual(report["summary"]["mean_ocr_cer"], 0.0)
        self.assertEqual(report["summary"]["mean_translation_cer"], 0.0)


class CharacterMemoryManagerTests(unittest.TestCase):
    def test_character_memory_response_validation(self):
        payload = {
            "characters": [{
                "character_id": "new_1",
                "display_name": "Personaje 1",
                "aliases": [],
                "role": "protagonista probable",
                "speech_style": "directo e informal",
                "personality_notes": "impulsivo",
                "confidence": 0.72,
            }],
            "assignments": [{
                "text_id": 0,
                "speaker_id": "new_1",
                "confidence": 0.7,
                "is_narration": False,
                "evidence": "tono enérgico",
            }],
            "page_summary": "Un personaje quiere avanzar.",
        }
        validated = validate_character_memory_response(payload, [0])
        self.assertEqual(validated["assignments"][0]["speaker_id"], "new_1")

    def test_character_memory_merges_new_llm_character(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CharacterMemoryManager(project_dir=tmp)
            inference = {
                "characters": [{
                    "character_id": "new_1",
                    "display_name": "Personaje 1",
                    "aliases": ["俺"],
                    "role": "hablante principal",
                    "speech_style": "informal",
                    "personality_notes": "decidido",
                    "confidence": 0.8,
                }],
                "assignments": [{
                    "text_id": 0,
                    "speaker_id": "new_1",
                    "confidence": 0.8,
                    "is_narration": False,
                    "evidence": "primera línea de diálogo",
                }],
                "page_summary": "Presenta a un hablante decidido.",
            }
            assignments = manager._merge_inference(inference, page_index=0, texts=["俺は行く"])
            saved = json.loads((Path(tmp) / "character_memory.json").read_text(encoding="utf-8"))

        self.assertEqual(assignments[0]["speaker_id"], "char_001")
        self.assertIn("char_001", saved["characters"])
        self.assertEqual(saved["characters"]["char_001"]["utterance_count"], 1)


class SourceLanguageFilterTests(unittest.TestCase):
    def test_japanese_filter_accepts_japanese_and_rejects_latin_even_fullwidth(self):
        filtro = SourceLanguageFilter("Japonés")

        self.assertTrue(filtro.should_process_text("おわり"))
        self.assertTrue(filtro.should_process_text("終"))
        self.assertFalse(filtro.should_process_text("Word"))
        self.assertFalse(filtro.should_process_text("Ｗｏｒｄ"))


    def test_dialogue_region_with_weak_latin_noise_hint_is_not_blocked(self):
        filtro = SourceLanguageFilter("Japonés")
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        region.source_text_hint = "a ;"

        self.assertTrue(filtro.should_process_region(region, allow_unknown=True))
        self.assertEqual(filtro.explain_region(region), "pista_global_debil_ignorada")

    def test_translator_skips_ocr_when_region_hint_is_not_source_language(self):
        translator = object.__new__(TranslateManga)
        translator.idioma_entrada = "Japonés"
        translator.source_language_filter = SourceLanguageFilter("Japonés")
        translator.normalizar_texto_ocr = lambda texto: texto
        translator.ocr_manager = types.SimpleNamespace(
            extract_texts=lambda _imagenes: (_ for _ in ()).throw(AssertionError("no debe llamar MangaOCR/EasyOCR"))
        )
        region = OnomatopoeiaKeepModeTests._region("free_text")
        region.source_text_hint = "EAST"
        translator.ultimas_regiones = [region]

        textos = translator.obtener_textos([np.zeros((20, 20, 3), dtype=np.uint8)])

        self.assertEqual(textos, [""])
        self.assertFalse(region.metadata["source_language_allowed"])
        self.assertEqual(region.metadata["source_language_filter"], "pista_global_en_idioma_distinto")

    def test_cleaner_filters_non_source_text_region_before_cleaning(self):
        cleaner = object.__new__(CleanManga)
        cleaner.idioma_entrada = "Japonés"
        cleaner.source_language_filter = SourceLanguageFilter("Japonés")
        region = OnomatopoeiaKeepModeTests._region("free_text")
        region.source_text_hint = "EAST"

        filtradas = cleaner._filter_regions_by_source_language([region])

        self.assertEqual(filtradas, [])
        self.assertFalse(region.metadata["source_language_allowed"])


class OnomatopoeiaKeepModeTests(unittest.TestCase):
    @staticmethod
    def _region(kind: str) -> TextRegion:
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        return TextRegion(bbox=(10, 10, 30, 30), text_bbox=(10, 10, 30, 30), mask=mask, kind=kind, confidence=0.9)

    def test_cleaner_does_not_clean_sfx_when_keep_mode_is_requested(self):
        cleaner = object.__new__(CleanManga)
        cleaner.onomatopoeia_mode = "keep"
        cleaner.translate_onomatopoeia = False
        cleaner.clean_onomatopoeia = False
        cleaner.idioma_entrada = "Japonés"
        cleaner.onomatopoeia_manager = OnomatopoeiaManager()

        self.assertFalse(cleaner._should_clean_non_bubble_region(self._region("sfx")))
        self.assertFalse(cleaner._should_clean_non_bubble_region(self._region("onomatopoeia")))

    def test_translator_keeps_any_sfx_region_out_of_llm_payload(self):
        translator = object.__new__(TranslateManga)
        translator.onomatopoeia_mode = "keep"
        translator.ultimas_regiones = [self._region("sfx")]
        translator.idioma_entrada = "Japonés"
        translator.idioma_salida = "Español"
        translator.onomatopoeia_manager = OnomatopoeiaManager()

        self.assertEqual(translator._traducir_onomatopeyas_con_diccionario(["texto raro"]), ["texto raro"])

    def test_cleaner_uses_free_text_onomatopoeia_metadata(self):
        cleaner = object.__new__(CleanManga)
        cleaner.onomatopoeia_mode = "translate"
        cleaner.translate_onomatopoeia = True
        cleaner.clean_onomatopoeia = False
        cleaner.idioma_entrada = "Japonés"
        cleaner.onomatopoeia_manager = OnomatopoeiaManager()
        region = self._region("free_text")
        region.metadata["free_text_onomatopoeia"] = True

        self.assertFalse(cleaner._should_clean_non_bubble_region(region))


    def test_cleaner_rechecks_free_text_with_region_ocr_before_erasing(self):
        cleaner = object.__new__(CleanManga)
        cleaner.onomatopoeia_mode = "keep"
        cleaner.translate_onomatopoeia = False
        cleaner.clean_onomatopoeia = False
        cleaner.idioma_entrada = "Japonés"
        cleaner.onomatopoeia_manager = OnomatopoeiaManager()
        region = self._region("free_text")
        region.source_text_hint = "A 、 附A"
        cleaner._ocr_text_for_clean_guard = lambda _imagen, _region: "ハッハッ"

        imagen = np.full((80, 80, 3), 255, dtype=np.uint8)

        self.assertFalse(cleaner._should_clean_non_bubble_region(region, imagen))
        self.assertTrue(region.metadata["free_text_onomatopoeia"])
        self.assertEqual(region.metadata["clean_guard_source"], "pre_clean_region_ocr")

    def test_translator_respects_translate_false_for_free_text_onomatopoeia_metadata(self):
        translator = object.__new__(TranslateManga)
        translator.onomatopoeia_mode = "translate"
        translator.translate_onomatopoeia = False
        translator.ultimas_regiones = [self._region("free_text")]
        translator.ultimas_regiones[0].metadata["free_text_onomatopoeia"] = True
        translator.idioma_entrada = "Japonés"
        translator.idioma_salida = "Español"
        translator.onomatopoeia_manager = OnomatopoeiaManager()

        self.assertEqual(translator._traducir_onomatopeyas_con_diccionario(["ドン"]), ["ドン"])

    def test_translator_reuses_clean_guard_ocr_for_kept_free_text_onomatopoeia(self):
        translator = object.__new__(TranslateManga)
        region = self._region("free_text")
        region.metadata["free_text_onomatopoeia_keep"] = True
        region.metadata["free_text_onomatopoeia"] = True
        region.metadata["clean_guard_ocr_text"] = "ハッハッ"
        translator.ultimas_regiones = [region]
        translator.normalizar_texto_ocr = lambda texto: texto
        translator.ocr_manager = types.SimpleNamespace(
            extract_texts=lambda _imagenes: (_ for _ in ()).throw(AssertionError("no debe repetir OCR"))
        )

        textos = translator.obtener_textos([np.zeros((20, 20, 3), dtype=np.uint8)])

        self.assertEqual(textos, ["ハッハッ"])

    def test_bubble_detector_marks_free_text_onomatopoeia_metadata(self):
        detector = object.__new__(BubbleDetector)
        detector.idioma_entrada = "Japonés"
        detector.onomatopoeia_manager = OnomatopoeiaManager()

        metadata = detector._free_text_onomatopoeia_metadata("ドン")

        self.assertTrue(metadata["free_text_onomatopoeia"])
        self.assertTrue(metadata["onomatopoeia"])
        self.assertIn("onomatopoeia_key", metadata)

    def test_translator_never_marks_dialogue_as_onomatopoeia_by_heuristic(self):
        translator = object.__new__(TranslateManga)
        translator.onomatopoeia_mode = "translate"
        translator.ultimas_regiones = [self._region("dialogue"), self._region("sfx")]
        translator.idioma_entrada = "Japonés"
        translator.onomatopoeia_manager = OnomatopoeiaManager()

        estilos = translator._clasificar_estilos_texto(["ドバ", "ドバ"])

        self.assertEqual(estilos, ["dialogo", "onomatopeya"])

    def test_json_payloads_include_estilo_in_transcription_and_translation(self):
        class DummyQueue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        translator = object.__new__(TranslateManga)
        translator.indice_imagen = 0
        translator.transcripcion_queue = DummyQueue()
        translator.traduccion_queue = DummyQueue()
        translator.ultimas_regiones = [self._region("dialogue"), self._region("sfx")]
        translator.ultimo_estilos_texto = ["dialogo", "onomatopeya"]
        translator.ultimas_asignaciones_hablante = []

        boxes = [(1, 2, 30, 40), (50, 60, 70, 80)]
        translator._push_original_texts_to_queue(boxes, ["やあ", "ドン"])
        translator._push_translated_texts_to_queue(boxes, ["Hola", "BOOM"])

        transcripcion = [
            item["agregar_a_sublista"]["elemento_sublista"]
            for item in translator.transcripcion_queue.items
        ]
        traduccion = [
            item["agregar_a_sublista"]["elemento_sublista"]
            for item in translator.traduccion_queue.items
        ]

        self.assertEqual([row["Estilo"] for row in transcripcion], ["dialogo", "onomatopeya"])
        self.assertEqual([row["Estilo"] for row in traduccion], ["dialogo", "onomatopeya"])
        self.assertNotIn("Subtitpo", transcripcion[0])
        self.assertNotIn("Subtitpo", traduccion[0])


if __name__ == "__main__":
    unittest.main()
