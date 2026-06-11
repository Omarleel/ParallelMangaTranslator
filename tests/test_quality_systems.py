import json
import tempfile
import unittest
from pathlib import Path

from Applications.CharacterMemoryManager import CharacterMemoryManager, validate_character_memory_response
from Applications.EvaluationManager import EvaluationManager, EvaluationConfig, box_iou, char_error_rate
from Applications.TranslatorManager import validate_translation_response


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


if __name__ == "__main__":
    unittest.main()
