from __future__ import annotations

import re


class OcrTextNormalizer:
    """Normaliza texto OCR/traducido sin conocer detalles de OCR, traducción o render."""

    SPECIAL_REPLACEMENTS = {
        "。": ".",
        "·": ".",
        "？": "?",
        "．": ".",
        "・": ".",
        "！": "!",
        "０": "",
        "“": '"',
        "”": '"',
        "’": "'",
    }

    def replace_special_characters(self, text: str) -> str:
        normalized = str(text or "")
        for special, replacement in self.SPECIAL_REPLACEMENTS.items():
            normalized = normalized.replace(special, replacement)
        return normalized

    @staticmethod
    def suppress_repeated_characters(text: str, min_reps: int = 3) -> str:
        pattern = r"(.)\1{{{},}}".format(min_reps)

        def replacement(match):
            return match.group(1) * 3

        return re.sub(pattern, replacement, str(text or ""))

    @staticmethod
    def suppress_symbols_and_spaces(text: str) -> str:
        normalized = str(text or "")
        for char in normalized:
            if char.isalnum():
                return normalized
        return ""

    def normalize_ocr_text(self, text: str) -> str:
        normalized = self.replace_special_characters(text)
        normalized = normalized.replace("\u3000", " ")
        normalized = re.sub(r"[|]{2,}", "I", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = self.suppress_repeated_characters(normalized, min_reps=4)
        return self.suppress_symbols_and_spaces(normalized)

    def normalize_translated_text(self, text: str, style: str) -> str:
        normalized = self.replace_special_characters(text).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        # En onomatopeyas conviene conservar alargamientos moderados: BOOOM, Aaaah, grrr.
        min_reps = 7 if style == "onomatopeya" else 3
        normalized = self.suppress_repeated_characters(normalized, min_reps=min_reps)
        return self.suppress_symbols_and_spaces(normalized)
