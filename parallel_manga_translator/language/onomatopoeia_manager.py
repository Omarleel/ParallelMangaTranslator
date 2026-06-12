from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional, Tuple

from parallel_manga_translator.language.onomatopoeia_repository import OnomatopoeiaYamlRepository


class OnomatopoeiaManager:
    """
    Reconoce y adapta onomatopeyas frecuentes en manga/manhwa/manhua/cómic.

    Los diccionarios viven fuera del código, agrupados por idioma en:
    parallel_manga_translator/resources/onomatopoeias/<codigo_idioma>/onomatopoeias.yaml

    Cada entrada YAML usa una clave semántica compartida. `sources` son las
    formas que pueden aparecer en OCR/texto fuente y `target` es la forma natural
    para renderizar en ese idioma cuando el modo de onomatopeyas permite traducir.
    """

    DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "resources" / "onomatopoeias"

    TARGET_BY_KEY: Dict[str, Dict[str, str]] = {}
    RAW_SOURCE_MAP: Dict[str, Dict[str, str]] = {}
    SOURCE_TO_KEY: Dict[str, str] = {}
    LANGUAGE_ALIASES: Dict[str, str] = {}
    _DATA_LOADED = False

    _LATIN_REPEATED = re.compile(r"\b([a-z]{1,4})(?:[-\s]*\1){1,}\b", re.IGNORECASE)
    _MOSTLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
    _JAPANESE_KANA_RE = re.compile(r"^[ぁ-ゟ゠-ヿーｯっ゛゜\s!！?？…\.｡。・･、,~〜\-]+$")
    _HANGUL_RE = re.compile(r"^[\uac00-\ud7af\s!！?？…\.~〜\-]+$")
    _CJK_RE = re.compile(r"^[\u4e00-\u9fff\s!！?？…\.~〜\-]+$")

    def __init__(self) -> None:
        self._ensure_data_loaded()

    @classmethod
    def _ensure_data_loaded(cls) -> None:
        if cls._DATA_LOADED:
            return
        cls._load_yaml_dictionaries(cls.DEFAULT_DATA_DIR)
        cls._build_source_map()
        cls._DATA_LOADED = True

    @classmethod
    def _load_yaml_dictionaries(cls, data_dir: Path) -> None:
        repository = OnomatopoeiaYamlRepository(data_dir, normalize_alias=cls.normalize_key)
        dictionaries = repository.load()
        cls.TARGET_BY_KEY = dictionaries.target_by_key
        cls.RAW_SOURCE_MAP = dictionaries.raw_source_map
        cls.LANGUAGE_ALIASES = dictionaries.language_aliases

    @classmethod
    def _build_source_map(cls) -> None:
        cls.SOURCE_TO_KEY = {}
        for _lang, rows in cls.RAW_SOURCE_MAP.items():
            for source, key in rows.items():
                cls.SOURCE_TO_KEY[cls.normalize_key(source)] = key

    @staticmethod
    def _strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    @classmethod
    def normalize_key(cls, text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text or "")).strip()
        text = text.replace("…", "...").replace("〜", "~").replace("ー", "-")
        # Quitar acentos ayuda en idiomas latinos, pero no debe eliminar dakuten/handakuten
        # japoneses porque ドン y トン significan cosas distintas.
        if re.search(r"[A-Za-zÀ-ÿ]", text):
            text = cls._strip_accents(text)
        text = text.lower()
        text = re.sub(r"[¡!¿?.,:;\"'“”‘’`´_*=+|/\\()[\]{}<>]+", " ", text)
        text = re.sub(r"[-~]{2,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Normaliza letras alargadas en alfabetos latinos: boooom -> boom, aaah -> aaah se conserva parcialmente.
        text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
        return text

    @classmethod
    def _canonical_language(cls, idioma: Optional[str]) -> Optional[str]:
        if not idioma:
            return None
        cls._ensure_data_loaded()
        normalized = cls.normalize_key(idioma)
        return cls.LANGUAGE_ALIASES.get(normalized, idioma)

    @staticmethod
    def _mostly_short(text: str) -> bool:
        stripped = re.sub(r"\s+", "", str(text or ""))
        return 1 <= len(stripped) <= 14

    @classmethod
    def _looks_like_repeated_sfx(cls, text: str) -> bool:
        normalized = cls.normalize_key(text)
        if not normalized:
            return False
        if cls._LATIN_REPEATED.search(normalized):
            return True
        no_space = normalized.replace(" ", "")
        if len(no_space) >= 4:
            half = len(no_space) // 2
            if len(no_space) % 2 == 0 and no_space[:half] == no_space[half:]:
                return True
        return False

    @staticmethod
    def _hiragana_to_katakana(text: str) -> str:
        return "".join(
            chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch
            for ch in str(text or "")
        )

    @classmethod
    def _normalize_for_similarity(cls, text: str, idioma: Optional[str] = None) -> str:
        idioma = cls._canonical_language(idioma)
        text = unicodedata.normalize("NFKC", str(text or "")).strip()
        if idioma == "Japonés":
            text = cls._hiragana_to_katakana(text)
            text = re.sub(r'[\s　!！?？…｡。・･,.:;"\'“”‘’`´_*=+|/\\()[\]{}<>]+', "", text)
            text = re.sub(r"[ー〜~\-]+", "", text)
            text = text.replace("ッ", "").replace("っ", "")
            small_to_large = str.maketrans("ァィゥェォャュョヮ", "アイウエオヤユヨワ")
            text = text.translate(small_to_large)
            # Confusiones frecuentes de OCR japonés en texto estilizado. La meta no
            # es traducir mejor, sino evitar que estos SFX terminen en el traductor.
            ocr_confusions = str.maketrans({
                "ソ": "ン",
                "ツ": "シ",
                "ヅ": "ジ",
                "口": "ロ",
                "〇": "ロ",
                "○": "ロ",
            })
            text = text.translate(ocr_confusions)
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)
            return text
        return cls.normalize_key(text).replace(" ", "")

    @classmethod
    def _source_rows_for_language(cls, idioma: Optional[str]) -> Dict[str, str]:
        cls._ensure_data_loaded()
        canonical = cls._canonical_language(idioma)
        if canonical and canonical in cls.RAW_SOURCE_MAP:
            return cls.RAW_SOURCE_MAP[canonical]
        return {source: key for rows in cls.RAW_SOURCE_MAP.values() for source, key in rows.items()}

    @classmethod
    def _similarity_threshold(cls, normalized_text: str, idioma: Optional[str]) -> float:
        idioma = cls._canonical_language(idioma)
        length = len(normalized_text)
        if idioma == "Japonés":
            if length <= 2:
                return 1.0
            if length <= 4:
                return 0.80
            return 0.74
        return 0.84

    def similar_semantic_key(
        self,
        text: str,
        idioma: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> Optional[Tuple[str, float, str]]:
        """Devuelve (clave_semantica, similitud, fuente) si el texto se parece a una onomatopeya.

        Se usa sobre todo para texto libre: allí el OCR suele deformar SFX
        estilizados y no conviene mandarlos al traductor.
        """
        idioma = self._canonical_language(idioma)
        normalized = self._normalize_for_similarity(text, idioma)
        if not normalized:
            return None

        if idioma == "Japonés":
            raw = unicodedata.normalize("NFKC", str(text or "")).strip()
            if not self._mostly_short(raw):
                return None
            if not self._JAPANESE_KANA_RE.match(raw):
                return None

        best_key: Optional[str] = None
        best_source = ""
        best_score = 0.0
        rows = self._source_rows_for_language(idioma)
        for source, key in rows.items():
            normalized_source = self._normalize_for_similarity(source, idioma)
            if not normalized_source:
                continue
            if normalized == normalized_source:
                return key, 1.0, source
            score = SequenceMatcher(None, normalized, normalized_source).ratio()
            if score > best_score:
                best_key = key
                best_source = source
                best_score = score

        threshold = min_similarity if min_similarity is not None else self._similarity_threshold(normalized, idioma)
        if best_key is not None and best_score >= threshold:
            return best_key, best_score, best_source
        return None

    def dictionary_semantic_key(self, text: str, idioma: Optional[str] = None) -> Optional[str]:
        """Clasificación segura: sólo diccionario exacto/normalizado.

        Este método es el único que debe usarse para decidir el estilo final de
        un diálogo normal. No aplica similitud ni heurísticas de forma.
        """
        self._canonical_language(idioma)  # Carga alias/diccionarios; el mapa global es multilingüe.
        normalized = self.normalize_key(text)
        if not normalized:
            return None

        if normalized in self.SOURCE_TO_KEY:
            return self.SOURCE_TO_KEY[normalized]

        compact = normalized.replace(" ", "")
        if compact in self.SOURCE_TO_KEY:
            return self.SOURCE_TO_KEY[compact]

        # Variante todavía segura: coincide con una entrada del diccionario tras
        # eliminar alargamientos/espacios. No introduce palabras nuevas por regex.
        compact_soft = re.sub(r"[-~]+", "", compact)
        compact_soft = re.sub(r"([a-z])\1{2,}", r"\1\1", compact_soft)
        if compact_soft in self.SOURCE_TO_KEY:
            return self.SOURCE_TO_KEY[compact_soft]

        return None

    def heuristic_semantic_key(self, text: str, idioma: Optional[str] = None) -> Optional[str]:
        """Pista débil para candidatos SFX, nunca para diálogo final.

        Se usa sólo antes de tener una región final, por ejemplo para decidir si
        un texto libre o una región SFX candidata debe tratarse como posible
        onomatopeya.
        """
        idioma = self._canonical_language(idioma)
        raw_text = str(text or "").strip()
        if not raw_text or self._MOSTLY_PUNCT_RE.match(raw_text):
            return None

        if idioma == "Japonés" and self._mostly_short(raw_text) and self._JAPANESE_KANA_RE.match(raw_text):
            # No basta con ー/〜/~: son comunes en diálogo. Requiere sílabas
            # típicas de SFX o una repetición clara.
            if re.search(r"[ドバガゴズザギキシチュヒフハパピプポ]", raw_text):
                return "impact" if re.search(r"[ドゴズガ]", raw_text) else "whoosh"
            if re.search(r"[ンッ]", raw_text) and self._looks_like_repeated_sfx(raw_text):
                return "whoosh"

        if idioma == "Coreano" and self._mostly_short(raw_text) and self._HANGUL_RE.match(raw_text):
            if self._looks_like_repeated_sfx(raw_text) or re.search(r"[쾅쿵퍽짝휙헉꺄]", raw_text):
                return "impact"

        if idioma == "Chino" and self._mostly_short(raw_text) and self._CJK_RE.match(raw_text):
            if self._looks_like_repeated_sfx(raw_text) or re.search(r"[砰轰啪唰呼嗒怦啊哈呜叮铃哗]", raw_text):
                return "impact"

        if idioma in {"Inglés", "Español", "Portugués", "Francés", "Italiano"} and self._mostly_short(raw_text):
            if self._looks_like_repeated_sfx(raw_text):
                return "impact"

        return None

    def candidate_semantic_key(
        self,
        text: str,
        idioma: Optional[str] = None,
        *,
        allow_similarity: bool = True,
        allow_heuristic: bool = True,
    ) -> Optional[Tuple[str, float, str, str]]:
        """Devuelve una clave para candidatos no-dialogue.

        Orden del flujo:
        1) diccionario exacto/normalizado,
        2) similitud contra diccionario,
        3) heurística débil de forma/caracteres.
        """
        key = self.dictionary_semantic_key(text, idioma)
        if key is not None:
            return key, 1.0, str(text or ""), "dictionary"

        if allow_similarity:
            match = self.similar_semantic_key(text, idioma)
            if match is not None:
                key, score, source = match
                return key, float(score), source, "similarity"

        if allow_heuristic:
            key = self.heuristic_semantic_key(text, idioma)
            if key is not None:
                return key, 0.50, str(text or ""), "heuristic"

        return None

    def is_free_text_onomatopoeia(self, text: str, idioma: Optional[str] = None) -> bool:
        return self.candidate_semantic_key(text, idioma, allow_similarity=True, allow_heuristic=False) is not None

    def is_onomatopoeia_candidate(self, text: str, idioma: Optional[str] = None) -> bool:
        text = str(text or "").strip()
        if not text or self._MOSTLY_PUNCT_RE.match(text):
            return False
        return self.candidate_semantic_key(text, idioma) is not None

    def semantic_key(self, text: str, idioma: Optional[str] = None) -> Optional[str]:
        return self.dictionary_semantic_key(text, idioma)

    def is_onomatopoeia(self, text: str, idioma: Optional[str] = None) -> bool:
        text = str(text or "").strip()
        if not text or self._MOSTLY_PUNCT_RE.match(text):
            return False
        return self.dictionary_semantic_key(text, idioma) is not None

    @staticmethod
    def _copy_intensity(original: str, translated: str) -> str:
        original = str(original or "")
        translated = str(translated or "").strip()
        if not translated or translated == "...":
            return translated
        bangs = original.count("!") + original.count("！")
        questions = original.count("?") + original.count("？")
        if bangs and "!" not in translated and "！" not in translated:
            translated += "!"
        if questions and "?" not in translated and "？" not in translated:
            translated += "?"
        if re.search(r"([ー〜~\-])\1+", original) and not translated.endswith(("…", "...")):
            translated += "…"
        return translated

    def translate(self, text: str, idioma_entrada: str, idioma_salida: str) -> Optional[str]:
        key = self.semantic_key(text, idioma_entrada)
        if key is None:
            return None
        idioma_salida = self._canonical_language(idioma_salida) or idioma_salida
        translated = self.TARGET_BY_KEY.get(key, {}).get(idioma_salida)
        if not translated:
            return None
        return self._copy_intensity(text, translated)

    def render_style(self, text: str, idioma: Optional[str] = None) -> str:
        return "onomatopeya" if self.is_onomatopoeia(text, idioma) else "dialogo"
