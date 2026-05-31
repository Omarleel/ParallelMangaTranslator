from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional


class OnomatopoeiaManager:
    """
    Reconoce y adapta onomatopeyas frecuentes en manga/manhwa/manhua/cómic.

    La idea no es traducir literalmente cada ruido, sino entregar una equivalencia
    corta, natural y fácil de insertar en la imagen. Si no hay una equivalencia
    segura, el flujo normal de traducción se mantiene como fallback.
    """

    TARGET_BY_KEY: Dict[str, Dict[str, str]] = {
        "impact": {
            "Español": "¡BUM!",
            "Inglés": "BOOM!",
            "Portugués": "BUM!",
            "Francés": "BOUM !",
            "Italiano": "BUM!",
            "Japonés": "ドン!",
            "Coreano": "쾅!",
            "Chino": "砰!",
        },
        "hit": {
            "Español": "¡PAM!",
            "Inglés": "BAM!",
            "Portugués": "PÁ!",
            "Francés": "BAM !",
            "Italiano": "BAM!",
            "Japonés": "バン!",
            "Coreano": "퍽!",
            "Chino": "啪!",
        },
        "slash": {
            "Español": "¡ZAS!",
            "Inglés": "SLASH!",
            "Portugués": "ZÁS!",
            "Francés": "SCHLAK !",
            "Italiano": "ZAC!",
            "Japonés": "ザシュ!",
            "Coreano": "슥!",
            "Chino": "唰!",
        },
        "step": {
            "Español": "TAP",
            "Inglés": "TAP",
            "Portugués": "TOC",
            "Francés": "TAP",
            "Italiano": "TAP",
            "Japonés": "トン",
            "Coreano": "톡",
            "Chino": "嗒",
        },
        "footsteps": {
            "Español": "TAP TAP",
            "Inglés": "TAP TAP",
            "Portugués": "TOC TOC",
            "Francés": "TAP TAP",
            "Italiano": "TAP TAP",
            "Japonés": "トコトコ",
            "Coreano": "뚜벅뚜벅",
            "Chino": "嗒嗒",
        },
        "heartbeat": {
            "Español": "DOKI DOKI",
            "Inglés": "THUMP THUMP",
            "Portugués": "TUM TUM",
            "Francés": "BOUM BOUM",
            "Italiano": "TUM TUM",
            "Japonés": "ドキドキ",
            "Coreano": "두근두근",
            "Chino": "怦怦",
        },
        "surprise": {
            "Español": "¡EH!",
            "Inglés": "HUH!",
            "Portugués": "HÃ!",
            "Francés": "HEIN !",
            "Italiano": "EH!",
            "Japonés": "ハッ!",
            "Coreano": "헉!",
            "Chino": "啊!",
        },
        "gasp": {
            "Español": "¡AH!",
            "Inglés": "GASP!",
            "Portugués": "AH!",
            "Francés": "AH !",
            "Italiano": "AH!",
            "Japonés": "はっ!",
            "Coreano": "헉!",
            "Chino": "哈!",
        },
        "scream": {
            "Español": "¡AAAH!",
            "Inglés": "AAAH!",
            "Portugués": "AAAH!",
            "Francés": "AAAH !",
            "Italiano": "AAAH!",
            "Japonés": "キャー!",
            "Coreano": "꺄악!",
            "Chino": "啊啊!",
        },
        "laugh": {
            "Español": "JA JA",
            "Inglés": "HA HA",
            "Portugués": "HA HA",
            "Francés": "HA HA",
            "Italiano": "AH AH",
            "Japonés": "ハハ",
            "Coreano": "하하",
            "Chino": "哈哈",
        },
        "chuckle": {
            "Español": "JE JE",
            "Inglés": "HEH HEH",
            "Portugués": "HE HE",
            "Francés": "HÉ HÉ",
            "Italiano": "EHEH",
            "Japonés": "フフ",
            "Coreano": "흐흐",
            "Chino": "呵呵",
        },
        "cry": {
            "Español": "BUA",
            "Inglés": "WAAH",
            "Portugués": "BUÁ",
            "Francés": "OUIN",
            "Italiano": "BUA",
            "Japonés": "うう",
            "Coreano": "엉엉",
            "Chino": "呜呜",
        },
        "silence": {
            "Español": "...",
            "Inglés": "...",
            "Portugués": "...",
            "Francés": "...",
            "Italiano": "...",
            "Japonés": "シーン",
            "Coreano": "고요",
            "Chino": "静",
        },
        "stare": {
            "Español": "MIRA FIJO",
            "Inglés": "STARE",
            "Portugués": "ENCARA",
            "Francés": "FIXE",
            "Italiano": "FISSA",
            "Japonés": "じー",
            "Coreano": "빤히",
            "Chino": "盯",
        },
        "sparkle": {
            "Español": "BRILLO",
            "Inglés": "SPARKLE",
            "Portugués": "BRILHO",
            "Francés": "ÉCLAT",
            "Italiano": "LUCCICHIO",
            "Japonés": "キラキラ",
            "Coreano": "반짝반짝",
            "Chino": "闪闪",
        },
        "running": {
            "Español": "TAC TAC",
            "Inglés": "DASH",
            "Portugués": "TAC TAC",
            "Francés": "TAC TAC",
            "Italiano": "TAC TAC",
            "Japonés": "ダダダ",
            "Coreano": "다다다",
            "Chino": "哒哒哒",
        },
        "rumble": {
            "Español": "GRRR",
            "Inglés": "RUMBLE",
            "Portugués": "GRRR",
            "Francés": "GRRR",
            "Italiano": "GRRR",
            "Japonés": "ゴゴゴ",
            "Coreano": "우르릉",
            "Chino": "轰隆",
        },
        "whisper": {
            "Español": "SUSURRO",
            "Inglés": "WHISPER",
            "Portugués": "SUSSURRO",
            "Francés": "CHUCHOTE",
            "Italiano": "SUSSURRO",
            "Japonés": "ヒソヒソ",
            "Coreano": "소곤소곤",
            "Chino": "窃窃",
        },
        "kiss": {
            "Español": "MUAC",
            "Inglés": "SMOOCH",
            "Portugués": "SMACK",
            "Francés": "SMACK",
            "Italiano": "SMACK",
            "Japonés": "チュ",
            "Coreano": "쪽",
            "Chino": "啵",
        },
        "doorbell": {
            "Español": "DING DONG",
            "Inglés": "DING DONG",
            "Portugués": "DING DONG",
            "Francés": "DING DONG",
            "Italiano": "DIN DON",
            "Japonés": "ピンポーン",
            "Coreano": "딩동",
            "Chino": "叮咚",
        },
        "phone": {
            "Español": "RING RING",
            "Inglés": "RING RING",
            "Portugués": "TRIM TRIM",
            "Francés": "DRING",
            "Italiano": "DRIN DRIN",
            "Japonés": "プルル",
            "Coreano": "따르릉",
            "Chino": "铃铃",
        },
        "splash": {
            "Español": "¡CHOF!",
            "Inglés": "SPLASH!",
            "Portugués": "SPLASH!",
            "Francés": "PLOUF !",
            "Italiano": "SPLASH!",
            "Japonés": "バシャ",
            "Coreano": "첨벙",
            "Chino": "哗啦",
        },
        "whoosh": {
            "Español": "¡FIU!",
            "Inglés": "WHOOSH!",
            "Portugués": "VUSH!",
            "Francés": "VOUF !",
            "Italiano": "FUUU!",
            "Japonés": "ヒュッ",
            "Coreano": "휙",
            "Chino": "呼",
        },
    }

    SOURCE_TO_KEY: Dict[str, str] = {}

    RAW_SOURCE_MAP: Dict[str, Dict[str, str]] = {
        "Japonés": {
            "ドン": "impact", "ドーン": "impact", "ズドン": "impact", "ガン": "impact", "ゴン": "impact",
            "バン": "hit", "パン": "hit", "ボコ": "hit", "バキ": "hit", "ガツ": "hit",
            "ザシュ": "slash", "ザク": "slash", "スパ": "slash", "シュッ": "whoosh", "ヒュ": "whoosh",
            "トン": "step", "トコトコ": "footsteps", "タッ": "step", "ダダダ": "running",
            "ドキドキ": "heartbeat", "ハッ": "surprise", "はっ": "gasp", "キャー": "scream",
            "ハハ": "laugh", "フフ": "chuckle", "うう": "cry", "シーン": "silence", "じー": "stare",
            "キラキラ": "sparkle", "ゴゴゴ": "rumble", "ヒソヒソ": "whisper", "チュ": "kiss",
            "ピンポーン": "doorbell", "プルル": "phone", "バシャ": "splash",
        },
        "Inglés": {
            "boom": "impact", "booom": "impact", "kaboom": "impact", "thud": "impact", "slam": "impact",
            "bam": "hit", "pow": "hit", "wham": "hit", "smack": "hit", "bonk": "hit",
            "slash": "slash", "slice": "slash", "shing": "slash", "whoosh": "whoosh", "woosh": "whoosh",
            "tap": "step", "tap tap": "footsteps", "step": "step", "dash": "running",
            "thump thump": "heartbeat", "thump": "heartbeat", "gasp": "gasp", "huh": "surprise",
            "aaah": "scream", "aah": "scream", "haha": "laugh", "ha ha": "laugh", "hehe": "chuckle",
            "sob": "cry", "sniff": "cry", "silence": "silence", "stare": "stare", "sparkle": "sparkle",
            "rumble": "rumble", "grrr": "rumble", "whisper": "whisper", "smooch": "kiss", "kiss": "kiss",
            "ding dong": "doorbell", "ring ring": "phone", "splash": "splash",
        },
        "Español": {
            "bum": "impact", "boom": "impact", "pum": "impact", "zas": "slash", "pam": "hit", "plaf": "hit",
            "toc": "step", "toc toc": "footsteps", "tap tap": "footsteps", "tac tac": "running",
            "doki doki": "heartbeat", "ah": "gasp", "eh": "surprise", "aaah": "scream",
            "jaja": "laugh", "ja ja": "laugh", "jeje": "chuckle", "bua": "cry", "silencio": "silence",
            "brillo": "sparkle", "grrr": "rumble", "susurro": "whisper", "muac": "kiss",
            "ding dong": "doorbell", "ring ring": "phone", "chof": "splash", "fiu": "whoosh",
        },
        "Coreano": {
            "쾅": "impact", "쿵": "impact", "퍽": "hit", "짝": "hit", "슥": "slash", "휙": "whoosh",
            "톡": "step", "뚜벅뚜벅": "footsteps", "다다다": "running", "두근두근": "heartbeat",
            "헉": "gasp", "꺄악": "scream", "하하": "laugh", "흐흐": "chuckle", "엉엉": "cry",
            "고요": "silence", "빤히": "stare", "반짝반짝": "sparkle", "우르릉": "rumble",
            "소곤소곤": "whisper", "쪽": "kiss", "딩동": "doorbell", "따르릉": "phone", "첨벙": "splash",
        },
        "Chino": {
            "砰": "impact", "轰": "impact", "轰隆": "rumble", "啪": "hit", "啪啪": "hit",
            "唰": "slash", "呼": "whoosh", "嗒": "step", "嗒嗒": "footsteps", "哒哒哒": "running",
            "怦怦": "heartbeat", "啊": "gasp", "啊啊": "scream", "哈哈": "laugh", "呵呵": "chuckle",
            "呜呜": "cry", "静": "silence", "盯": "stare", "闪闪": "sparkle", "窃窃": "whisper",
            "啵": "kiss", "叮咚": "doorbell", "铃铃": "phone", "哗啦": "splash",
        },
    }

    _LATIN_REPEATED = re.compile(r"\b([a-z]{1,4})(?:[-\s]*\1){1,}\b", re.IGNORECASE)
    _MOSTLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
    _JAPANESE_KANA_RE = re.compile(r"^[ぁ-ゟ゠-ヿーｯっ゛゜\s!！?？…\.・~〜\-]+$")
    _HANGUL_RE = re.compile(r"^[\uac00-\ud7af\s!！?？…\.~〜\-]+$")
    _CJK_RE = re.compile(r"^[\u4e00-\u9fff\s!！?？…\.~〜\-]+$")

    def __init__(self) -> None:
        if not self.SOURCE_TO_KEY:
            self._build_source_map()

    @classmethod
    def _build_source_map(cls) -> None:
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

    def semantic_key(self, text: str, idioma: Optional[str] = None) -> Optional[str]:
        normalized = self.normalize_key(text)
        if not normalized:
            return None

        if normalized in self.SOURCE_TO_KEY:
            return self.SOURCE_TO_KEY[normalized]

        compact = normalized.replace(" ", "")
        if compact in self.SOURCE_TO_KEY:
            return self.SOURCE_TO_KEY[compact]

        # Permite detectar variantes alargadas: ドーーン, boooom, grrrrr.
        compact_soft = re.sub(r"[-~]+", "", compact)
        compact_soft = re.sub(r"([a-z])\1{2,}", r"\1\1", compact_soft)
        if compact_soft in self.SOURCE_TO_KEY:
            return self.SOURCE_TO_KEY[compact_soft]

        if idioma == "Japonés" and self._mostly_short(text) and self._JAPANESE_KANA_RE.match(str(text).strip()):
            if re.search(r"[ドバガゴズザギキシチュヒフハパピプポンッー〜~]", str(text)):
                return "impact" if re.search(r"[ドゴズガ]", str(text)) else "whoosh"

        if idioma == "Coreano" and self._mostly_short(text) and self._HANGUL_RE.match(str(text).strip()):
            if self._looks_like_repeated_sfx(text) or re.search(r"[쾅쿵퍽짝휙헉꺄]", str(text)):
                return "impact"

        if idioma == "Chino" and self._mostly_short(text) and self._CJK_RE.match(str(text).strip()):
            if self._looks_like_repeated_sfx(text) or re.search(r"[砰轰啪唰呼嗒怦啊哈呜叮铃哗]", str(text)):
                return "impact"

        if idioma in {"Inglés", "Español", "Portugués", "Francés", "Italiano"} and self._mostly_short(text):
            if self._looks_like_repeated_sfx(text):
                return "impact"

        return None

    def is_onomatopoeia(self, text: str, idioma: Optional[str] = None) -> bool:
        text = str(text or "").strip()
        if not text or self._MOSTLY_PUNCT_RE.match(text):
            return False
        return self.semantic_key(text, idioma) is not None

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
        translated = self.TARGET_BY_KEY.get(key, {}).get(idioma_salida)
        if not translated:
            return None
        return self._copy_intensity(text, translated)

    def render_style(self, text: str, idioma: Optional[str] = None) -> str:
        return "onomatopeya" if self.is_onomatopoeia(text, idioma) else "dialogo"
