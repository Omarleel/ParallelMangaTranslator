import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.modules.setdefault("easyocr", types.SimpleNamespace(Reader=object))

from parallel_manga_translator.translation.character_memory_manager import CharacterMemoryManager, validate_character_memory_response
from parallel_manga_translator.quality.evaluation_manager import EvaluationManager, EvaluationConfig, box_iou, char_error_rate
from parallel_manga_translator.quality.visual_inpaint_verifier import VisualInpaintVerifier
from parallel_manga_translator.translation.translator_manager import validate_translation_response
from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.processing.clean_manga import CleanManga
from parallel_manga_translator.processing.clean_inpainting_pipeline_mixin import CleanInpaintingPipelineMixin
from parallel_manga_translator.language.onomatopoeia_manager import OnomatopoeiaManager
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.processing.translate_manga import TranslateManga


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


class VisualInpaintVerifierTests(unittest.TestCase):
    @staticmethod
    def _bubble_fixture():
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        context = np.zeros((120, 160), dtype=np.uint8)
        cv2.ellipse(context, (80, 60), (55, 35), 0, 0, 360, 255, -1)
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[45:76, 60:101] = 255
        return image, mask, context

    def test_accepts_clean_uniform_fill(self):
        before, mask, context = self._bubble_fixture()
        report = VisualInpaintVerifier().evaluate(before, before.copy(), mask, context_mask=context)

        self.assertTrue(report.passed)
        self.assertEqual(report.failed_checks, [])

    def test_detects_halo_and_ink_residue(self):
        before, mask, context = self._bubble_fixture()
        after = before.copy()
        halo = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        halo = cv2.bitwise_and(halo, cv2.bitwise_not(mask))
        after[halo > 0] = 220
        cv2.putText(after, "x", (68, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)

        report = VisualInpaintVerifier().evaluate(before, after, mask, context_mask=context)

        self.assertFalse(report.passed)
        self.assertIn("halo", report.failed_checks)
        self.assertTrue(
            "ink_residue" in report.failed_checks or "texture_mismatch" in report.failed_checks
        )

    def test_detects_flat_patch_on_screentone(self):
        before, mask, context = self._bubble_fixture()
        before[:] = 235
        for x in range(0, before.shape[1], 6):
            cv2.line(before, (x, 0), (x, before.shape[0] - 1), (210, 210, 210), 1)
        after = before.copy()
        after[mask > 0] = 235

        report = VisualInpaintVerifier().evaluate(before, after, mask, context_mask=context)

        self.assertFalse(report.passed)
        self.assertIn("flat_patch", report.failed_checks)

    def test_debug_artifacts_are_written_under_debug_inpaint(self):
        class DummyDebugWriter(CleanInpaintingPipelineMixin):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = DummyDebugWriter()
            writer.visual_inpaint_debug = True
            writer.set_visual_inpaint_debug_context(str(root), 0, "0001.jpg")

            before, mask, context = self._bubble_fixture()
            after = before.copy()
            region = TextRegion(
                bbox=(20, 20, 120, 80),
                text_bbox=(60, 45, 41, 31),
                mask=context,
                kind="dialogue",
                confidence=0.91,
                clean_mask=mask,
            )
            report = VisualInpaintVerifier().evaluate(before, after, mask, context_mask=context)

            metadata = writer._write_visual_inpaint_region_debug_summary(
                region_index=0,
                region=region,
                before_image=before,
                after_image=after,
                clean_mask=mask,
                safe_mask=context,
                fill_color=(255, 255, 255),
                fill_strategy="inpaint",
                method="solid_color",
                chosen_candidate="solid",
                report=report,
                attempts=[{"candidate": "solid", "accepted": True, "score": 0.0}],
            )

            page_dir = root / "debug_inpaint" / "0001"
            self.assertTrue(page_dir.exists())
            self.assertTrue((page_dir / "manifest.json").exists())
            self.assertTrue((page_dir / "r000_report.json").exists())
            self.assertIn("debug_inpaint/0001/manifest.json", metadata["visual_inpaint_debug_manifest"])
            for relative in metadata["visual_inpaint_debug_files"].values():
                self.assertTrue((root / relative).exists(), relative)


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

    def test_dialogue_onomatopoeia_inside_bubble_is_translatable(self):
        filtro = SourceLanguageFilter("Japonés")
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        region.source_text_hint = "は"
        region.metadata.update({
            "free_text_onomatopoeia": True,
            "onomatopoeia": True,
            "onomatopoeia_key": "surprise",
            # Metadata que podía quedar de v6: ya no debe bloquear globos.
            "free_text_onomatopoeia_keep": True,
            "visual_expression": True,
            "skip_cleanup_translation": True,
            "visual_expression_source": "short_bubble_ocr_hint",
        })

        self.assertFalse(filtro.should_preserve_region_without_processing(region))
        self.assertTrue(filtro.should_process_region(region, allow_unknown=True))

    def test_cleaner_keeps_dialogue_onomatopoeia_in_processing_pipeline(self):
        cleaner = object.__new__(CleanManga)
        cleaner.idioma_entrada = "Japonés"
        cleaner.source_language_filter = SourceLanguageFilter("Japonés")
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        region.source_text_hint = "は"
        region.metadata.update({"free_text_onomatopoeia": True, "onomatopoeia": True})

        filtradas = cleaner._filter_regions_by_source_language([region])

        self.assertEqual(filtradas, [region])
        self.assertTrue(region.metadata["source_language_allowed"])
        self.assertNotIn("processing_skipped", region.metadata)

    def test_cleaner_skips_region_when_specialized_ocr_finds_no_text(self):
        cleaner = object.__new__(CleanManga)
        cleaner.idioma_entrada = "Japonés"
        cleaner._ocr_text_for_processing_guard = lambda _imagen, _region: ""
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        imagen = np.full((80, 80, 3), 255, dtype=np.uint8)

        filtradas = cleaner._filter_regions_by_specialized_ocr_guard(imagen, [region])

        self.assertEqual(filtradas, [])
        self.assertTrue(region.metadata["specialized_ocr_guard_empty"])
        self.assertTrue(region.metadata["processing_skipped"])
        self.assertEqual(region.metadata["processing_skip_reason"], "ocr_especializado_sin_texto")

    def test_translator_reuses_specialized_ocr_guard_text_for_dialogue(self):
        translator = object.__new__(TranslateManga)
        translator.idioma_entrada = "Japonés"
        translator.source_language_filter = SourceLanguageFilter("Japonés")
        translator.normalizar_texto_ocr = lambda texto: texto
        translator.ocr_manager = types.SimpleNamespace(
            extract_texts=lambda _imagenes: (_ for _ in ()).throw(AssertionError("no debe repetir OCR"))
        )
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        region.metadata.update({
            "specialized_ocr_guard_passed": True,
            "region_ocr_cache_reusable": True,
            "region_ocr_text": "行くぞ",
        })
        translator.ultimas_regiones = [region]

        textos = translator.obtener_textos([np.zeros((20, 20, 3), dtype=np.uint8)])

        self.assertEqual(textos, ["行くぞ"])

    def test_translator_skips_region_empty_by_specialized_ocr_guard(self):
        translator = object.__new__(TranslateManga)
        translator.idioma_entrada = "Japonés"
        translator.source_language_filter = SourceLanguageFilter("Japonés")
        translator.normalizar_texto_ocr = lambda texto: texto
        translator.ocr_manager = types.SimpleNamespace(
            extract_texts=lambda _imagenes: (_ for _ in ()).throw(AssertionError("no debe llamar OCR"))
        )
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        region.metadata.update({
            "specialized_ocr_guard_empty": True,
            "processing_skip_reason": "ocr_especializado_sin_texto",
        })
        translator.ultimas_regiones = [region]

        textos = translator.obtener_textos([np.zeros((20, 20, 3), dtype=np.uint8)])

        self.assertEqual(textos, [""])

    def test_detector_materializes_bubble_onomatopoeia_as_translatable(self):
        detector = object.__new__(BubbleDetector)
        detector.idioma_entrada = "Japonés"
        detector.onomatopoeia_manager = OnomatopoeiaManager()
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        region.source_text_hint = "は"

        detector._annotate_bubble_visual_expressions([region])

        self.assertTrue(region.metadata["free_text_onomatopoeia"])
        self.assertTrue(region.metadata["bubble_onomatopoeia"])
        self.assertTrue(region.metadata["translate_inside_bubble"])
        self.assertNotIn("free_text_onomatopoeia_keep", region.metadata)
        self.assertNotIn("skip_cleanup_translation", region.metadata)
        self.assertEqual(region.metadata["visual_expression_source"], "short_bubble_ocr_hint")
        self.assertFalse(SourceLanguageFilter("Japonés").should_preserve_region_without_processing(region))

    def test_detector_does_not_materialize_long_dialogue_hint_as_visual_expression(self):
        detector = object.__new__(BubbleDetector)
        detector.idioma_entrada = "Japonés"
        detector.onomatopoeia_manager = OnomatopoeiaManager()
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        region.source_text_hint = "パーティーに行く"

        detector._annotate_bubble_visual_expressions([region])

        self.assertFalse(region.metadata.get("skip_cleanup_translation", False))
        self.assertFalse(region.metadata.get("bubble_onomatopoeia", False))
        self.assertFalse(SourceLanguageFilter("Japonés").should_preserve_region_without_processing(region))

    def test_translator_runs_ocr_for_dialogue_onomatopoeia_inside_bubble(self):
        translator = object.__new__(TranslateManga)
        translator.idioma_entrada = "Japonés"
        translator.source_language_filter = SourceLanguageFilter("Japonés")
        translator.normalizar_texto_ocr = lambda texto: texto
        translator.ocr_manager = types.SimpleNamespace(extract_texts=lambda _imagenes: ["ハッ"])
        region = OnomatopoeiaKeepModeTests._region("dialogue")
        region.source_text_hint = "は"
        region.metadata.update({"free_text_onomatopoeia": True, "onomatopoeia": True})
        translator.ultimas_regiones = [region]

        textos = translator.obtener_textos([np.zeros((20, 20, 3), dtype=np.uint8)])

        self.assertEqual(textos, ["ハッ"])
        self.assertTrue(region.metadata["source_language_allowed"])
        self.assertNotIn("processing_skipped", region.metadata)


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


class CleanMaskSeparationTests(unittest.TestCase):
    def _cleaner(self):
        cleaner = object.__new__(CleanManga)
        cleaner.bubble_fill_edge_margin = 3
        cleaner.bubble_fill_text_dilate = 1
        cleaner.bubble_fill_feather = 1.0
        cleaner.bubble_fill_whole_interior = False
        cleaner.bubble_fill_flat_max_rectangularity = 0.86
        cleaner.bubble_fill_strategy = "solid"
        cleaner.bubble_fill_background_std_threshold = 18.0
        cleaner.bubble_fill_inpaint_padding = 8
        cleaner.inpaint_model = "opencv-tela"
        return cleaner

    def test_bubble_region_mask_and_ink_clean_mask_are_separate(self):
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        image[48:62, 44:70] = 0
        bubble_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_mask[18:100, 18:100] = 255
        region = TextRegion(
            bbox=(18, 18, 82, 82),
            text_bbox=(44, 48, 26, 14),
            mask=bubble_mask,
            kind="dialogue",
            detections_count=1,
            metadata={},
        )

        cleaner = self._cleaner()
        [prepared] = cleaner._attach_clean_masks(image, [region])

        bubble_pixels = int(np.count_nonzero(prepared.mask))
        clean_pixels = int(np.count_nonzero(prepared.clean_mask))
        self.assertGreater(bubble_pixels, 5000)
        self.assertGreater(clean_pixels, 0)
        self.assertLess(clean_pixels, bubble_pixels * 0.30)
        self.assertEqual(
            int(np.count_nonzero(BubbleDetector.compose_clean_mask([prepared], image.shape))),
            clean_pixels,
        )
        self.assertGreater(
            int(np.count_nonzero(BubbleDetector.compose_mask([prepared], image.shape))),
            clean_pixels,
        )
        self.assertTrue(prepared.metadata["bubble_clean_mask_separated"])
        self.assertEqual(prepared.metadata["clean_mask_source"], "text_ink_inside_bubble")

    def test_bubble_without_text_zone_does_not_clean_whole_bubble(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        bubble_mask = np.zeros((100, 100), dtype=np.uint8)
        bubble_mask[10:90, 10:90] = 255
        region = TextRegion(
            bbox=(10, 10, 80, 80),
            text_bbox=(10, 10, 80, 80),
            mask=bubble_mask,
            kind="dialogue",
            detections_count=0,
            metadata={},
        )

        cleaner = self._cleaner()
        [prepared] = cleaner._attach_clean_masks(image, [region])

        self.assertEqual(int(np.count_nonzero(prepared.clean_mask)), 0)
        self.assertEqual(int(np.count_nonzero(BubbleDetector.compose_clean_mask([prepared], image.shape))), 0)
        self.assertGreater(int(np.count_nonzero(BubbleDetector.compose_mask([prepared], image.shape))), 0)
        self.assertEqual(prepared.metadata["clean_mask_source"], "empty_text_ink_inside_bubble")

    def test_bubble_without_ocr_boxes_still_erases_its_ink(self):
        """El localizador puede fallar dentro de un globo que YOLO sí encontró.

        Sin este rescate el texto original nunca se borra y la traducción se dibuja
        encima.
        """
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        image[50:64, 40:80] = 0
        bubble_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_mask[20:100, 20:100] = 255
        region = TextRegion(
            bbox=(20, 20, 80, 80),
            text_bbox=(20, 20, 80, 80),
            mask=bubble_mask,
            kind="dialogue",
            detections_count=0,
            metadata={},
        )

        cleaner = self._cleaner()
        [prepared] = cleaner._attach_clean_masks(image, [region])

        clean_pixels = int(np.count_nonzero(prepared.clean_mask))
        self.assertGreater(clean_pixels, 300)
        self.assertLess(clean_pixels, int(np.count_nonzero(prepared.mask)) * 0.30)
        self.assertEqual(prepared.metadata["clean_mask_source"], "bubble_interior_ink_without_ocr")

    def test_bubble_without_ocr_boxes_keeps_dense_interior_untouched(self):
        """Una máscara con demasiada tinta no es un globo con texto: no se toca.

        Cubre el caso peligroso: una detección que cayó sobre arte, no sobre un globo.
        """
        image = np.full((120, 120, 3), 235, dtype=np.uint8)
        for fila in range(20, 100, 4):
            image[fila:fila + 2, 20:100] = 20
        bubble_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_mask[20:100, 20:100] = 255
        region = TextRegion(
            bbox=(20, 20, 80, 80),
            text_bbox=(20, 20, 80, 80),
            mask=bubble_mask,
            kind="dialogue",
            detections_count=0,
            metadata={},
        )

        cleaner = self._cleaner()
        [prepared] = cleaner._attach_clean_masks(image, [region])

        self.assertEqual(int(np.count_nonzero(prepared.clean_mask)), 0)
        self.assertEqual(prepared.metadata["clean_mask_source"], "empty_text_ink_inside_bubble")

    def test_dark_narration_box_uses_bright_ink_clean_mask(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[15:85, 15:85] = 20
        image[42:55, 38:64] = 245
        box_mask = np.zeros((100, 100), dtype=np.uint8)
        box_mask[15:85, 15:85] = 255
        region = TextRegion(
            bbox=(15, 15, 70, 70),
            text_bbox=(38, 42, 26, 13),
            mask=box_mask,
            kind="narration",
            detections_count=1,
            metadata={},
        )

        cleaner = self._cleaner()
        [prepared] = cleaner._attach_clean_masks(image, [region])

        clean_pixels = int(np.count_nonzero(prepared.clean_mask))
        self.assertGreater(clean_pixels, 0)
        self.assertLess(clean_pixels, int(np.count_nonzero(prepared.mask)) * 0.30)
        self.assertEqual(prepared.metadata["clean_mask_source"], "text_ink_inside_bubble")

    def test_dark_bubble_fill_uses_dark_background_not_white_ink(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[15:85, 15:85] = 18
        image[42:55, 38:64] = 245
        box_mask = np.zeros((100, 100), dtype=np.uint8)
        box_mask[15:85, 15:85] = 255
        region = TextRegion(
            bbox=(15, 15, 70, 70),
            text_bbox=(38, 42, 26, 13),
            mask=box_mask,
            kind="narration",
            detections_count=1,
            metadata={},
        )

        cleaner = self._cleaner()
        [prepared] = cleaner._attach_clean_masks(image, [region])
        filled = cleaner._fill_bubble_interiors(image, [prepared])

        self.assertLess(float(np.mean(filled[45:52, 42:60])), 55.0)
        self.assertLess(float(np.mean(prepared.metadata["fill_color_bgr"])), 55.0)
        self.assertEqual(prepared.metadata["fill_color_source"], "safe_region_minus_clean_mask")

    def test_light_bubble_fill_still_uses_light_background(self):
        image = np.full((100, 100, 3), 30, dtype=np.uint8)
        image[15:85, 15:85] = 242
        image[42:55, 38:64] = 5
        box_mask = np.zeros((100, 100), dtype=np.uint8)
        box_mask[15:85, 15:85] = 255
        region = TextRegion(
            bbox=(15, 15, 70, 70),
            text_bbox=(38, 42, 26, 13),
            mask=box_mask,
            kind="dialogue",
            detections_count=1,
            metadata={},
        )

        cleaner = self._cleaner()
        [prepared] = cleaner._attach_clean_masks(image, [region])
        filled = cleaner._fill_bubble_interiors(image, [prepared])

        self.assertGreater(float(np.mean(filled[45:52, 42:60])), 215.0)
        self.assertGreater(float(np.mean(prepared.metadata["fill_color_bgr"])), 215.0)

    def test_inpaint_strategy_uses_configured_inpaint_even_on_uniform_background(self):
        class DummyConfiguredInpainter:
            def __init__(self):
                self.calls = 0

            def inpaint(self, img, mask):
                self.calls += 1
                out = img.copy()
                out[mask > 0] = 77
                return out

        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        image[48:62, 44:70] = 0
        bubble_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_mask[18:102, 18:102] = 255
        region = TextRegion(
            bbox=(18, 18, 84, 84),
            text_bbox=(44, 48, 26, 14),
            mask=bubble_mask,
            kind="dialogue",
            detections_count=1,
            metadata={},
        )

        cleaner = self._cleaner()
        cleaner.bubble_fill_strategy = "inpaint"
        cleaner.bubble_fill_background_std_threshold = 999.0
        cleaner.inpainter = DummyConfiguredInpainter()
        [prepared] = cleaner._attach_clean_masks(image, [region])
        filled = cleaner._fill_bubble_interiors(image, [prepared])

        self.assertEqual(cleaner.inpainter.calls, 1)
        self.assertEqual(prepared.metadata["bubble_fill_method"], "configured_inpaint")
        self.assertEqual(prepared.metadata["bubble_fill_strategy"], "inpaint")
        self.assertAlmostEqual(float(np.mean(filled[50:58, 48:65])), 77.0, delta=6.0)

    def test_varied_bubble_background_uses_configured_inpaint_model(self):
        class DummyConfiguredInpainter:
            def __init__(self):
                self.calls = 0

            def inpaint(self, img, mask):
                self.calls += 1
                out = img.copy()
                out[mask > 0] = 123
                return out

        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        # Fondo variado dentro del globo: simula trama/transparencia sobre dibujo.
        for y in range(18, 102):
            for x in range(18, 102):
                v = 170 + ((x * 7 + y * 5) % 70)
                image[y, x] = (v, v, v)
        image[48:62, 44:70] = 0
        bubble_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_mask[18:102, 18:102] = 255
        region = TextRegion(
            bbox=(18, 18, 84, 84),
            text_bbox=(44, 48, 26, 14),
            mask=bubble_mask,
            kind="dialogue",
            detections_count=1,
            metadata={},
        )

        cleaner = self._cleaner()
        cleaner.bubble_fill_strategy = "auto"
        cleaner.bubble_fill_background_std_threshold = 4.0
        cleaner.inpainter = DummyConfiguredInpainter()
        [prepared] = cleaner._attach_clean_masks(image, [region])
        filled = cleaner._fill_bubble_interiors(image, [prepared])

        self.assertEqual(cleaner.inpainter.calls, 1)
        self.assertEqual(prepared.metadata["bubble_fill_method"], "configured_inpaint")
        self.assertEqual(prepared.metadata["bubble_fill_inpaint_model"], "opencv-tela")
        self.assertGreater(prepared.metadata["background_variation_score"], 4.0)
        self.assertAlmostEqual(float(np.mean(filled[50:58, 48:65])), 123.0, delta=6.0)

    def test_bubble_fill_strategy_solid_forces_solid_fill(self):
        class DummyConfiguredInpainter:
            def __init__(self):
                self.calls = 0

            def inpaint(self, img, mask):
                self.calls += 1
                return img.copy()

        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        for y in range(18, 102):
            for x in range(18, 102):
                v = 170 + ((x * 7 + y * 5) % 70)
                image[y, x] = (v, v, v)
        image[48:62, 44:70] = 0
        bubble_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_mask[18:102, 18:102] = 255
        region = TextRegion(
            bbox=(18, 18, 84, 84),
            text_bbox=(44, 48, 26, 14),
            mask=bubble_mask,
            kind="dialogue",
            detections_count=1,
            metadata={},
        )

        cleaner = self._cleaner()
        cleaner.bubble_fill_strategy = "solid"
        cleaner.bubble_fill_background_std_threshold = 4.0
        cleaner.inpainter = DummyConfiguredInpainter()
        [prepared] = cleaner._attach_clean_masks(image, [region])
        cleaner._fill_bubble_interiors(image, [prepared])

        self.assertEqual(cleaner.inpainter.calls, 0)
        self.assertEqual(prepared.metadata["bubble_fill_method"], "solid_color")
        self.assertGreater(prepared.metadata["background_variation_score"], 4.0)

class MaturePrecisionAdaptationsTests(unittest.TestCase):
    def test_fine_text_mask_uses_ocr_polygon_not_full_bbox(self):
        from parallel_manga_translator.quality.text_mask_refiner import TextInkMaskRefiner

        detection = ([[10, 10], [42, 14], [38, 28], [8, 24]], "字", 0.9)
        mask = TextInkMaskRefiner.mask_from_detections((60, 60, 3), [detection], dilate_px=0, min_pad=0)
        polygon_pixels = int(np.count_nonzero(mask))
        bbox_pixels = 34 * 19

        self.assertGreater(polygon_pixels, 0)
        self.assertLess(polygon_pixels, bbox_pixels)

    def test_ink_refinement_keeps_text_anchor_and_rejects_far_noise(self):
        from parallel_manga_translator.quality.text_mask_refiner import TextInkMaskRefiner, TextMaskRefinementOptions

        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[40:52, 40:60] = 0
        image[75:78, 75:78] = 0
        safe = np.zeros((100, 100), dtype=np.uint8)
        safe[10:90, 10:90] = 255
        raw = np.zeros((100, 100), dtype=np.uint8)
        raw[38:54, 38:62] = 255
        text_zone = np.zeros((100, 100), dtype=np.uint8)
        text_zone[30:85, 30:85] = 255
        initial = np.zeros((100, 100), dtype=np.uint8)
        initial[40:52, 40:60] = 255
        initial[75:78, 75:78] = 255

        refined = TextInkMaskRefiner.refine(
            image,
            safe,
            text_zone,
            raw_text_mask=raw,
            initial_ink_mask=initial,
            options=TextMaskRefinementOptions(fine_mask_dilate=1, min_component_area=2),
        )

        self.assertGreater(int(np.count_nonzero(refined[42:50, 44:56])), 0)
        self.assertEqual(int(np.count_nonzero(refined[75:78, 75:78])), 0)

    def test_panel_aware_order_reads_right_panel_first_for_japanese(self):
        import cv2
        from parallel_manga_translator.layout.panel_order_resolver import PanelAwareReadingOrderResolver, PanelOrderConfig
        from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver

        image = np.full((160, 260, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (115, 145), (0, 0, 0), 3)
        cv2.rectangle(image, (145, 10), (250, 145), (0, 0, 0), 3)
        resolver = PanelAwareReadingOrderResolver(
            ReadingOrderResolver("Japonés"),
            PanelOrderConfig(enabled=True, min_area_ratio=0.03, gutter_px=7),
        )
        left = TextRegion((35, 40, 35, 35), (35, 40, 35, 35), np.ones((160, 260), dtype=np.uint8) * 255, metadata={})
        right = TextRegion((180, 40, 35, 35), (180, 40, 35, 35), np.ones((160, 260), dtype=np.uint8) * 255, metadata={})

        ordered = resolver.sort_regions(image, [left, right])

        self.assertIs(ordered[0], right)
        self.assertEqual(right.metadata["panel_index"], 0)
        self.assertEqual(left.metadata["panel_index"], 1)
        self.assertEqual(right.metadata["panel_order_source"], "opencv_panel_edges")

    def test_typography_breaks_long_latin_token_with_soft_hyphen(self):
        from parallel_manga_translator.rendering.text_renderer import TextRenderer

        renderer = TextRenderer(smart_typography=True, hyphenation=True, balance_lines=True)
        font = renderer._get_font(18)
        token = "extraordinariamente"
        max_width = max(20, renderer._text_width(token, font) // 2)

        parts = renderer._break_long_token(token, font, max_width)

        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(any(part.endswith("-") for part in parts[:-1]))


class FreeTextRegionCleaningTests(unittest.TestCase):
    """Limpieza de texto libre/SFX: una pasada por región, no una por página."""

    @staticmethod
    def _cleaner():
        cleaner = object.__new__(CleanManga)
        cleaner.bubble_fill_feather = 1.0
        cleaner.bubble_fill_background_std_threshold = 18.0
        cleaner.bubble_fill_strategy = "solid"
        cleaner.bubble_fill_inpaint_padding = 8
        cleaner.inpaint_model = "opencv-tela"
        cleaner.visual_inpaint_verifier_enabled = False
        cleaner.idioma_entrada = "Japonés"
        cleaner.onomatopoeia_mode = "translate"
        cleaner.translate_onomatopoeia = True
        cleaner.clean_onomatopoeia = True
        cleaner.onomatopoeia_manager = OnomatopoeiaManager()
        return cleaner

    @staticmethod
    def _free_text_region(image_shape, rect):
        x, y, w, h = rect
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mask[y:y + h, x:x + w] = 255
        return TextRegion(
            bbox=rect,
            text_bbox=rect,
            mask=mask.copy(),
            clean_mask=mask,
            kind="free_text",
            confidence=0.5,
            metadata={},
        )

    def _page_with_two_texts(self):
        # Fondo con textura: es justo lo que una pasada de inpaint sobre toda la página
        # difumina cuando la máscara cubre zonas grandes y separadas.
        image = np.full((300, 300, 3), 235, dtype=np.uint8)
        image[:, ::7] = 180
        rects = [(30, 40, 60, 50), (200, 210, 55, 45)]
        for x, y, w, h in rects:
            image[y + 8:y + h - 8, x + 8:x + w - 8] = 15
        return image, rects

    def test_cada_region_se_limpia_por_separado(self):
        image, rects = self._page_with_two_texts()
        cleaner = self._cleaner()
        regions = [self._free_text_region(image.shape, rect) for rect in rects]

        llamadas = []
        original = CleanManga._clean_region_with_masks

        def espia(self, salida, region, region_index, clean_mask, safe_mask):
            llamadas.append((region_index, int(cv2.countNonZero(clean_mask))))
            return original(self, salida, region, region_index, clean_mask, safe_mask)

        cleaner._clean_region_with_masks = espia.__get__(cleaner, CleanManga)
        limpia = cleaner._clean_free_text_regions(image, regions, debug_index_offset=3)

        self.assertEqual([indice for indice, _ in llamadas], [3, 4])
        self.assertTrue(all(pixeles > 0 for _, pixeles in llamadas))
        for x, y, w, h in rects:
            self.assertLess(int(np.count_nonzero(limpia[y:y + h, x:x + w] < 60)), 10)

    def test_el_fondo_fuera_de_las_mascaras_no_se_toca(self):
        image, rects = self._page_with_two_texts()
        cleaner = self._cleaner()
        regions = [self._free_text_region(image.shape, rect) for rect in rects]

        limpia = cleaner._clean_free_text_regions(image, regions)

        union = np.zeros(image.shape[:2], dtype=np.uint8)
        for x, y, w, h in rects:
            union[y:y + h, x:x + w] = 255
        # Margen para el difuminado del borde de la máscara.
        fuera = cv2.bitwise_not(cv2.dilate(union, np.ones((9, 9), np.uint8), iterations=1))
        self.assertEqual(int(np.count_nonzero(cv2.absdiff(image, limpia)[fuera > 0])), 0)

    def test_regiones_a_limpiar_excluye_onomatopeyas_conservadas(self):
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        cleaner = self._cleaner()
        cleaner.onomatopoeia_mode = "keep"
        cleaner.translate_onomatopoeia = False
        cleaner.clean_onomatopoeia = False
        cleaner.source_language_filter = SourceLanguageFilter("Japonés")

        globo = self._free_text_region(image.shape, (10, 10, 40, 40))
        globo.kind = "dialogue"
        globo.source_text_hint = "こんにちは"
        sfx = self._free_text_region(image.shape, (60, 60, 40, 40))
        sfx.kind = "sfx"

        globos, libres = cleaner.regiones_a_limpiar(image, [globo, sfx])

        self.assertEqual(globos, [globo])
        self.assertEqual(libres, [])


class FreeTextInkMaskTests(unittest.TestCase):
    """La máscara de borrado del texto libre son los trazos, no la caja del OCR."""

    ALTO, ANCHO = 260, 320
    CAJA = (60, 60, 200, 140)

    def _pagina_con_rotulo_claro(self):
        """Rótulo blanco sobre gris medio: el caso que rompía la detección de polaridad."""
        image = np.full((self.ALTO, self.ANCHO, 3), 170, dtype=np.uint8)
        x, y, w, h = self.CAJA
        for fila in range(y + 12, y + h - 12, 24):
            image[fila:fila + 14, x + 14:x + w - 14] = 250
        return image

    def _region_de_caja_ocr(self):
        """Como la construye el detector: el polígono OCR relleno, es decir la caja entera."""
        mask = np.zeros((self.ALTO, self.ANCHO), dtype=np.uint8)
        x, y, w, h = self.CAJA
        mask[y:y + h, x:x + w] = 255
        return TextRegion(
            bbox=self.CAJA,
            text_bbox=self.CAJA,
            mask=mask.copy(),
            clean_mask=mask.copy(),
            text_mask=mask.copy(),
            kind="free_text",
            confidence=0.5,
            metadata={},
        )

    @staticmethod
    def _cleaner():
        cleaner = object.__new__(CleanManga)
        cleaner.bubble_fill_edge_margin = 3
        cleaner.bubble_fill_text_dilate = 2
        cleaner.bubble_fill_whole_interior = False
        cleaner.bubble_fill_flat_max_rectangularity = 0.86
        cleaner.fine_text_detection = True
        cleaner.fine_text_mask_dilate = 2
        cleaner.ink_mask_refinement = True
        cleaner.ink_mask_min_component_area = 3
        cleaner.ink_mask_component_anchor_overlap = 0.03
        cleaner.ink_mask_component_anchor_max_gap_ratio = 0.45
        return cleaner

    def test_la_mascara_de_borrado_no_es_la_caja_ocr(self):
        """Sembrar el refinamiento con la propia caja lo anulaba: devolvía el bloque entero.

        Con un bloque sólido, cualquier inpainter barre el fondo en vez de borrar el texto.
        """
        image = self._pagina_con_rotulo_claro()
        [prepared] = self._cleaner()._attach_clean_masks(image, [self._region_de_caja_ocr()])

        pixeles_caja = self.CAJA[2] * self.CAJA[3]
        pixeles_tinta = int(np.count_nonzero(prepared.clean_mask))
        self.assertGreater(pixeles_tinta, 0)
        self.assertLess(pixeles_tinta, pixeles_caja * 0.75)
        self.assertEqual(prepared.metadata["clean_mask_source"], "region_text_ink")

    def _pagina_con_trazos(self, fondo: int, tinta: int, grosor: int):
        """Rótulo de N trazos dentro de la caja OCR, para variar la densidad de tinta."""
        image = np.full((self.ALTO, self.ANCHO, 3), fondo, dtype=np.uint8)
        x, y, w, h = self.CAJA
        for i in range(5):
            fila = y + 10 + i * 26
            image[fila:fila + grosor, x + 10:x + w - 10] = tinta
        return image

    def _cobertura_de_tinta(self, image, tinta_es_clara: bool) -> float:
        [prepared] = self._cleaner()._attach_clean_masks(image, [self._region_de_caja_ocr()])
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        trazos = (gris >= 235) if tinta_es_clara else (gris <= 60)
        x, y, w, h = self.CAJA
        dentro = np.zeros(gris.shape, dtype=bool)
        dentro[y:y + h, x:x + w] = True
        trazos = trazos & dentro
        total = int(np.count_nonzero(trazos))
        self.assertGreater(total, 0)
        return int(np.count_nonzero(trazos & (prepared.clean_mask > 0))) / total

    def test_detecta_la_tinta_aunque_sea_escasa_en_la_caja(self):
        """Con trazos finos, los cuartiles caen los dos en el fondo y la polaridad se perdía."""
        self.assertGreater(self._cobertura_de_tinta(self._pagina_con_trazos(170, 250, 3), True), 0.9)

    def test_detecta_la_tinta_aunque_domine_la_caja(self):
        """Si la tinta ocupa casi toda la caja, la mediana 'del fondo' era la propia tinta."""
        self.assertGreater(self._cobertura_de_tinta(self._pagina_con_trazos(170, 250, 22), True), 0.9)

    def test_detecta_tinta_oscura_escasa_sobre_blanco(self):
        self.assertGreater(self._cobertura_de_tinta(self._pagina_con_trazos(250, 20, 3), False), 0.9)

    def test_detecta_texto_de_color_con_la_misma_luminancia_que_el_fondo(self):
        """Texto de color sobre gris: la luminancia no lo distingue, el color sí.

        Detectando la tinta por luma, un rótulo cuyo brillo coincide con el del fondo es
        invisible y sobrevive entero a la limpieza; es lo que pasaba con los rótulos de
        color sobre arte claro.
        """
        fondo = (150, 150, 150)
        tinta = (60, 170, 190)  # luma ~163 frente a 150: indistinguible en gris
        image = np.full((self.ALTO, self.ANCHO, 3), fondo, dtype=np.uint8)
        x, y, w, h = self.CAJA
        for i in range(5):
            fila = y + 10 + i * 26
            image[fila:fila + 8, x + 10:x + w - 10] = tinta

        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.assertLess(abs(int(np.median(gris)) - 150), 20, "el test debe ser ambiguo en luma")

        [prepared] = self._cleaner()._attach_clean_masks(image, [self._region_de_caja_ocr()])
        trazos = np.all(image == np.array(tinta, dtype=np.uint8), axis=-1)
        cubiertos = int(np.count_nonzero(trazos & (prepared.clean_mask > 0)))
        self.assertGreater(cubiertos / max(1, int(np.count_nonzero(trazos))), 0.9)

    def test_la_mascara_cubre_los_trazos_claros_sobre_gris(self):
        """La polaridad la decide el contraste: sobre gris medio la tinta puede ser clara."""
        image = self._pagina_con_rotulo_claro()
        [prepared] = self._cleaner()._attach_clean_masks(image, [self._region_de_caja_ocr()])

        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        trazos = gris >= 235
        cubiertos = int(np.count_nonzero(trazos & (prepared.clean_mask > 0)))
        self.assertGreater(cubiertos / max(1, int(np.count_nonzero(trazos))), 0.85)


class AutoInpaintCandidateTests(unittest.TestCase):
    """`auto` reserva LaMa para fondos complejos: color o trama."""

    UMBRAL = 18.0

    @staticmethod
    def _cleaner():
        return object.__new__(CleanManga)

    def test_pagina_a_color_usa_lama(self):
        color = np.zeros((80, 80, 3), dtype=np.uint8)
        color[:, :, 2] = 220  # canal rojo dominante: página a color
        elegido = self._cleaner()._auto_inpaint_candidate(color, 0.0, self.UMBRAL)
        self.assertEqual(elegido, "lama_mpe")

    def test_fondo_monocromo_con_trama_usa_lama(self):
        """Una página B/N con trama es fondo complejo aunque no tenga color."""
        gris = np.full((80, 80, 3), 200, dtype=np.uint8)
        elegido = self._cleaner()._auto_inpaint_candidate(gris, self.UMBRAL + 5, self.UMBRAL)
        self.assertEqual(elegido, "lama_mpe")

    def test_fondo_monocromo_plano_no_usa_lama(self):
        """Interior de globo blanco: LaMa no aporta y cuesta GPU."""
        gris = np.full((80, 80, 3), 245, dtype=np.uint8)
        elegido = self._cleaner()._auto_inpaint_candidate(gris, 2.0, self.UMBRAL)
        self.assertEqual(elegido, "solid")


class TextHaloGrowthTests(unittest.TestCase):
    """La máscara debe tragarse el halo del texto, o el inpainter lo propaga hacia dentro."""

    LADO = 260
    CAJA = (60, 60, 140, 140)

    def _pagina(self):
        """Glifos oscuros con halo claro sobre un fondo con textura, como una trama."""
        from parallel_manga_translator.quality.text_mask_refiner import TextInkMaskRefiner  # noqa: F401

        image = np.full((self.LADO, self.LADO, 3), 147, dtype=np.uint8)
        image[::3, :] = 129          # textura del fondo
        image[1::3, :] = 165
        x, y, w, h = self.CAJA
        halo = np.zeros((self.LADO, self.LADO), dtype=bool)
        glifo = np.zeros((self.LADO, self.LADO), dtype=bool)
        # El texto debe dominar la caja: es el balance de la situación real, y es el que
        # hace que Otsu corte entre los glifos y el resto, dejando el halo con el fondo.
        for i in range(4):
            fila = y + 2 + i * 34
            halo[fila:fila + 30, x + 4:x + w - 4] = True
            glifo[fila + 10:fila + 20, x + 4:x + w - 4] = True
        halo &= ~glifo
        image[halo] = 214            # halo claro alrededor del trazo
        image[glifo] = 12            # tinta
        return image, halo, glifo

    @staticmethod
    def _mascara(image, caja, growth_px):
        from parallel_manga_translator.quality.text_mask_refiner import (
            TextInkMaskRefiner,
            TextMaskRefinementOptions,
        )

        x, y, w, h = caja
        zona = np.zeros(image.shape[:2], dtype=np.uint8)
        zona[y:y + h, x:x + w] = 255
        crecimiento = None
        if growth_px:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * growth_px + 5,) * 2)
            crecimiento = cv2.dilate(zona, kernel)
        return TextInkMaskRefiner.refine(
            image,
            zona,
            zona,
            raw_text_mask=zona,
            initial_ink_mask=None,
            options=TextMaskRefinementOptions(halo_growth_px=growth_px),
            growth_mask=crecimiento,
        )

    def test_el_umbral_del_halo_queda_entre_el_fondo_y_la_tinta(self):
        """En una región hay tres poblaciones, no dos.

        Otsu corta en dos: con los glifos muy lejos del fondo, el halo queda agrupado con
        el fondo y el umbral de tinta no sirve para crecer hacia él. El umbral del halo
        tiene que caer en medio.
        """
        from parallel_manga_translator.quality.text_mask_refiner import TextInkMaskRefiner

        # Tres poblaciones como las medidas en material real: fondo ~49, halo ~137,
        # glifos ~240 de distancia al color del fondo.
        distancia = np.concatenate([
            np.full(9000, 49.0), np.full(46000, 137.0), np.full(19000, 240.0),
        ]).astype(np.float32)
        distancia = distancia.reshape(-1, 1)
        zona = np.full(distancia.shape, 255, dtype=np.uint8)

        umbral_tinta = 185.0
        umbral_halo = TextInkMaskRefiner._halo_threshold(distancia, zona, umbral_tinta)

        # El umbral es una cota exclusiva (convenio de OpenCV): lo que cuenta es a qué
        # población deja pasar, no su valor.
        self.assertLess(umbral_halo, 137.0, "el halo debe quedar por encima del umbral")
        self.assertGreaterEqual(umbral_halo, 49.0, "el fondo debe quedar por debajo")

    def test_el_crecimiento_cubre_el_halo(self):
        image, halo, glifo = self._pagina()
        mascara = self._mascara(image, self.CAJA, 10)
        self.assertGreater(
            np.count_nonzero(halo & (mascara > 0)) / max(1, np.count_nonzero(halo)), 0.85
        )
        self.assertGreater(
            np.count_nonzero(glifo & (mascara > 0)) / max(1, np.count_nonzero(glifo)), 0.85
        )

    def test_el_crecimiento_se_detiene_en_el_fondo(self):
        """Debe pararse al tocar la trama, no seguir comiéndose la página."""
        image, halo, glifo = self._pagina()
        mascara = self._mascara(image, self.CAJA, 10)
        texto = cv2.dilate(
            (halo | glifo).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        )
        lejos = (texto == 0) & (mascara > 0)
        self.assertLess(np.count_nonzero(lejos) / mascara.size, 0.01)


class BubbleDatasetExportTests(unittest.TestCase):
    """Exportación de trabajos corregidos como dataset YOLO-seg."""

    @staticmethod
    def _pagina_con_globo(destino: Path):
        """Un globo elíptico blanco con borde negro sobre fondo gris."""
        imagen = np.full((400, 300, 3), 120, dtype=np.uint8)
        cv2.ellipse(imagen, (150, 200), (90, 60), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(imagen, (150, 200), (90, 60), 0, 0, 360, (0, 0, 0), 4)
        cv2.imwrite(str(destino), imagen)
        return imagen

    def _trabajo(self, raiz: Path, job_id: str, bbox):
        job = raiz / job_id
        job.mkdir(parents=True, exist_ok=True)
        imagen_path = job / "0001.jpg"
        self._pagina_con_globo(imagen_path)
        manifest = {
            "job_id": job_id,
            "pages": [{
                "index": 0,
                "original_path": str(imagen_path),
                "regions": [
                    {"index": 0, "type": "dialogue", "bbox": list(bbox), "deleted": False},
                    {"index": 1, "type": "free_text", "bbox": [10, 10, 40, 20], "deleted": False},
                    {"index": 2, "type": "dialogue", "bbox": list(bbox), "deleted": True},
                ],
            }],
        }
        (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_exporta_poligonos_normalizados_y_solo_de_globos(self):
        from parallel_manga_translator.quality.export_bubble_dataset import export_jobs

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            self._trabajo(raiz / "jobs", "trabajo_a", (60, 140, 180, 120))
            stats = export_jobs(raiz / "jobs", raiz / "out", dataset_dir=raiz / "no_existe", val_ratio=0.0)

            # El texto libre no es un globo y la región borrada tampoco cuenta.
            self.assertEqual(stats.regions, 1)
            self.assertEqual(stats.labelled, 1)

            etiquetas = list((raiz / "out" / "labels").rglob("*.txt"))
            self.assertEqual(len(etiquetas), 1)
            partes = etiquetas[0].read_text(encoding="utf-8").split()
            self.assertEqual(partes[0], "0")
            coordenadas = [float(v) for v in partes[1:]]
            self.assertGreaterEqual(len(coordenadas), 6)
            self.assertEqual(len(coordenadas) % 2, 0)
            self.assertTrue(all(0.0 <= c <= 1.0 for c in coordenadas))
            self.assertTrue((raiz / "out" / "data.yaml").is_file())

    def test_excluye_los_trabajos_que_originan_el_banco(self):
        """Entrenar con las páginas del banco invalidaría la evaluación."""
        from parallel_manga_translator.quality.export_bubble_dataset import export_jobs

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            self._trabajo(raiz / "jobs", "trabajo_del_banco", (60, 140, 180, 120))
            caso = raiz / "banco" / "caso_x"
            caso.mkdir(parents=True)
            (caso / "case.json").write_text(json.dumps({"origin_job": "trabajo_del_banco"}), encoding="utf-8")

            stats = export_jobs(raiz / "jobs", raiz / "out", dataset_dir=raiz / "banco")
            self.assertEqual(stats.labelled, 0)

            forzado = export_jobs(raiz / "jobs", raiz / "out2", dataset_dir=raiz / "banco",
                                  include_eval_jobs=True)
            self.assertEqual(forzado.labelled, 1)

    def test_el_reparto_train_val_es_estable(self):
        """`hash()` de Python está aleatorizado por proceso y no sirve aquí."""
        from parallel_manga_translator.quality.export_bubble_dataset import _split_for

        primero = [_split_for("trabajo", i, 0.2) for i in range(40)]
        segundo = [_split_for("trabajo", i, 0.2) for i in range(40)]
        self.assertEqual(primero, segundo)
        self.assertIn("val", primero)
        self.assertIn("train", primero)


class BestOfInpaintCandidateTests(unittest.TestCase):
    """Sobre fondo con textura conviene mirar todos los rellenos, no cortar en el primero."""

    UMBRAL = 18.0
    #: Menor es mejor, así que `lama_mpe` debería ganar si se exploran todos.
    PUNTUACIONES = {"opencv-tela": 0.9, "solid": 0.5, "lama_mpe": 0.2, "aot": 0.7}

    def _cleaner(self, best_of: bool, aplicados: list):
        puntuaciones = self.PUNTUACIONES

        class VerificadorFalso:
            """Puntúa el último candidato aplicado; todos aprueban, para aislar la elección."""

            def evaluate(self, before, after, mask, context_mask=None):
                return types.SimpleNamespace(
                    passed=True,
                    score=puntuaciones[aplicados[-1]],
                    failed_checks=[],
                    to_dict=lambda: {},
                )

        cleaner = object.__new__(CleanManga)
        cleaner.inpaint_model = "opencv-tela"
        cleaner.visual_inpaint_retry = True
        cleaner.visual_inpaint_retry_models = "solid,opencv-tela,lama_mpe,aot"
        cleaner.visual_inpaint_max_retries = 4
        cleaner.visual_inpaint_debug = False
        cleaner.visual_inpaint_best_of_textured = best_of
        cleaner.visual_inpaint_verifier = VerificadorFalso()

        def aplicar(imagen, clean_mask, safe_mask, fill_color, candidate, *, sigma):
            aplicados.append(candidate)
            return np.full((8, 8, 3), 10, dtype=np.uint8), f"configured_inpaint:{candidate}"

        cleaner._apply_visual_inpaint_candidate = aplicar
        return cleaner

    def _ejecutar(self, best_of: bool, variacion: float):
        aplicados: list = []
        cleaner = self._cleaner(best_of, aplicados)
        imagen = np.full((8, 8, 3), 200, dtype=np.uint8)
        mascara = np.zeros((8, 8), dtype=np.uint8)
        mascara[2:6, 2:6] = 255
        resultado = cleaner._apply_bubble_cleaning_with_visual_verifier(
            imagen, mascara, mascara, (255, 255, 255),
            fill_strategy="inpaint",
            background_variation=variacion,
            variation_threshold=self.UMBRAL,
            sigma=0.6,
        )
        return resultado, aplicados

    def test_sobre_fondo_plano_corta_en_el_primero_que_aprueba(self):
        (_img, _metodo, _informe, _intentos, elegido), aplicados = self._ejecutar(True, variacion=2.0)

        self.assertEqual(elegido, "opencv-tela")
        self.assertEqual(aplicados, ["opencv-tela"])

    def test_sobre_fondo_con_textura_elige_el_de_mejor_score(self):
        (_img, metodo, _informe, _intentos, elegido), aplicados = self._ejecutar(True, variacion=40.0)

        self.assertEqual(elegido, "lama_mpe")
        self.assertGreater(len(aplicados), 1, "debe probar más de un candidato")
        self.assertIn("best_of_candidates", metodo)

    def test_apagado_mantiene_el_corte_aunque_haya_textura(self):
        (_img, _metodo, _informe, _intentos, elegido), aplicados = self._ejecutar(False, variacion=40.0)

        self.assertEqual(elegido, "opencv-tela")
        self.assertEqual(aplicados, ["opencv-tela"])
