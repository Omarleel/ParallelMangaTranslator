import unittest
import unittest.mock

from parallel_manga_translator.config.app_config import OcrConfig
from parallel_manga_translator.ocr.engines import OcrFactory
from parallel_manga_translator.ocr.paddle_result import normalize_paddle_result
from parallel_manga_translator.ocr.settings import OcrSettings
from parallel_manga_translator.ocr.text_detection import TextDetectionFactory


class OcrRefactorTests(unittest.TestCase):
    def test_shared_settings_resolve_paddle_subprocess_once(self):
        self.assertTrue(OcrSettings("Japonés", gpu=True, paddle_subprocess="auto").use_paddle_subprocess())
        self.assertFalse(OcrSettings("Japonés", gpu=True, paddle_subprocess="never").use_paddle_subprocess())
        self.assertEqual(OcrSettings("Japonés").paddle_lang, "japan")
        self.assertEqual(OcrSettings("Español").easyocr_langs, ["es", "en"])

    def test_transcription_factory_keeps_legacy_defaults_and_aliases(self):
        japanese_auto = OcrFactory.create("Japonés", OcrConfig())
        english_auto = OcrFactory.create("Inglés", OcrConfig())
        paddle_gpu = OcrFactory.create("Inglés", OcrConfig(transcription_engine="paddle", gpu=True))

        self.assertEqual(japanese_auto.engine_id, "mangaocr+easyocr")
        self.assertEqual(english_auto.engine_id, "easyocr")
        self.assertEqual(paddle_gpu.engine_id, "paddle_subprocess")

    def test_detection_factory_redirects_mangaocr_to_easyocr_and_paddle_gpu_to_worker(self):
        manga_detector = TextDetectionFactory.create("Japonés", OcrConfig(detection_engine="mangaocr"))
        paddle_gpu_detector = TextDetectionFactory.create("Inglés", OcrConfig(detection_engine="paddle", gpu=True))

        self.assertEqual(manga_detector.engine_id, "easyocr")
        self.assertEqual(paddle_gpu_detector.engine_id, "paddle_subprocess")

    def test_paddle_result_normalizer_accepts_v2_and_v3_shapes(self):
        v2_result = [
            [
                [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Hola", 0.93)],
                [[[12, 0], [25, 0], [25, 10], [12, 10]], ("mundo", 0.91)],
            ]
        ]
        v3_result = {
            "rec_texts": ["行くぞ"],
            "rec_scores": [0.88],
            "dt_polys": [[[1, 2], [7, 2], [7, 9], [1, 9]]],
        }

        self.assertEqual([line["text"] for line in normalize_paddle_result(v2_result)], ["Hola", "mundo"])
        normalized_v3 = normalize_paddle_result(v3_result)
        self.assertEqual(normalized_v3[0]["text"], "行くぞ")
        self.assertAlmostEqual(normalized_v3[0]["confidence"], 0.88)
        self.assertEqual(normalized_v3[0]["box"][0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()


class AutoEngineForLanguageTests(unittest.TestCase):
    """Qué motor elige `auto`, que es lo que usa casi todo el mundo."""

    def test_japanese_keeps_mangaocr(self):
        self.assertEqual(OcrFactory._auto_engine_for("Japonés"), "mangaocr_auto")

    def test_latin_keeps_easyocr(self):
        for idioma in ("Inglés", "Español"):
            with self.subTest(idioma=idioma):
                self.assertEqual(OcrFactory._auto_engine_for(idioma), "easyocr")

    def test_chinese_and_korean_prefer_paddle(self):
        """EasyOCR no lee bien estos dos, y el guardián de OCR borra lo que no se lee.

        Medido sobre un tomo chino real: de 31 regiones EasyOCR leyó 8 y Paddle 20. La
        diferencia no es peor transcripción, son regiones que desaparecen de la página.
        """
        from parallel_manga_translator.ocr import settings as ajustes

        ajustes.paddle_disponible.cache_clear()
        try:
            with unittest.mock.patch.object(ajustes.importlib.util, "find_spec", return_value=object()):
                for idioma in ("Chino", "Coreano"):
                    with self.subTest(idioma=idioma):
                        self.assertEqual(OcrFactory._auto_engine_for(idioma), "paddleocr")
        finally:
            ajustes.paddle_disponible.cache_clear()

    def test_without_paddle_installed_it_falls_back_instead_of_breaking(self):
        """Paddle es un extra opcional: el valor por defecto no puede exigirlo."""
        from parallel_manga_translator.ocr import settings as ajustes

        ajustes.paddle_disponible.cache_clear()
        try:
            with unittest.mock.patch.object(ajustes.importlib.util, "find_spec", return_value=None):
                self.assertEqual(OcrFactory._auto_engine_for("Chino"), "easyocr")
        finally:
            ajustes.paddle_disponible.cache_clear()

    def test_on_gpu_chinese_goes_through_the_subprocess(self):
        """Paddle y YOLO en la misma GPU rompen CUDA; por eso existe el subproceso."""
        settings = OcrSettings("Chino", gpu=True, paddle_subprocess="auto")

        self.assertEqual(
            OcrFactory._resolve_engine_name("auto", "Chino", settings), "paddle_subprocess"
        )


class AutoLocalizerForLanguageTests(unittest.TestCase):
    """Qué localizador elige `auto`. Importa más de lo que parece.

    Las reglas de texto libre deciden a partir del texto que el localizador cree leer, así
    que un localizador que no sabe leer el idioma no transcribe peor: hace que se descarten
    regiones que sí tienen texto.
    """

    def test_chinese_and_korean_prefer_paddle(self):
        from parallel_manga_translator.ocr import settings as ajustes

        ajustes.paddle_disponible.cache_clear()
        try:
            with unittest.mock.patch.object(ajustes.importlib.util, "find_spec", return_value=object()):
                for idioma in ("Chino", "Coreano"):
                    with self.subTest(idioma=idioma):
                        self.assertEqual(TextDetectionFactory._auto_engine_for(idioma), "paddleocr")
        finally:
            ajustes.paddle_disponible.cache_clear()

    def test_japanese_and_latin_keep_easyocr(self):
        for idioma in ("Japonés", "Inglés", "Español"):
            with self.subTest(idioma=idioma):
                self.assertEqual(TextDetectionFactory._auto_engine_for(idioma), "easyocr")

    def test_without_paddle_installed_it_falls_back(self):
        from parallel_manga_translator.ocr import settings as ajustes

        ajustes.paddle_disponible.cache_clear()
        try:
            with unittest.mock.patch.object(ajustes.importlib.util, "find_spec", return_value=None):
                self.assertEqual(TextDetectionFactory._auto_engine_for("Chino"), "easyocr")
        finally:
            ajustes.paddle_disponible.cache_clear()

    def test_on_gpu_it_goes_through_the_subprocess(self):
        """Paddle y YOLO en la misma GPU rompen CUDA; por eso existe el subproceso."""
        settings = OcrSettings("Chino", gpu=True, paddle_subprocess="auto")

        self.assertEqual(
            TextDetectionFactory._resolve_engine_name("auto", settings), "paddle_subprocess"
        )
