from __future__ import annotations

import re
from typing import List




class SlotTextSplitterMixin:
    """Segmentación de oraciones para múltiples ranuras de renderizado."""

    @staticmethod
    def _sentence_units(texto: str) -> List[str]:
        texto = re.sub(r"\s+", " ", str(texto or " ")).strip() or " "
        # Conserva el signo de puntuación en la unidad. Incluye puntuación española,
        # japonesa y puntos suspensivos para separar frases naturales de un mismo globo doble.
        units = re.findall(r".+?(?:[.!?。！？…]+|$)(?:\s+|$)", texto, flags=re.S)
        units = [re.sub(r"\s+", " ", unit).strip() for unit in units if unit.strip()]
        return units or [texto]

    def _split_text_for_slots(self, texto: str, slot_count: int) -> List[str]:
        slot_count = max(1, int(slot_count))
        texto = self._normalize_text(texto)
        if slot_count == 1:
            return [texto]

        units = self._sentence_units(texto)
        if len(units) >= slot_count:
            chunks: List[str] = []
            remaining_units = list(units)
            for slot_idx in range(slot_count):
                remaining_slots = slot_count - slot_idx
                if remaining_slots == 1:
                    chunks.append(" ".join(remaining_units).strip())
                    break
                remaining_chars = sum(len(unit) for unit in remaining_units)
                target = max(1, remaining_chars / remaining_slots)
                current: List[str] = []
                current_len = 0
                while remaining_units and len(remaining_units) > remaining_slots - 1:
                    next_unit = remaining_units[0]
                    if current and current_len + len(next_unit) > target * 1.18:
                        break
                    current.append(remaining_units.pop(0))
                    current_len += len(next_unit)
                    if current_len >= target * 0.82:
                        break
                chunks.append(" ".join(current).strip())
            return [(chunk or " ") for chunk in chunks[:slot_count]]

        # Si no hay suficientes frases, parte por palabras procurando equilibrio.
        words = texto.split()
        if len(words) < slot_count * 2:
            # Último recurso para idiomas sin espacios: corte por caracteres.
            total = len(texto)
            chunks = []
            start = 0
            for slot_idx in range(slot_count):
                end = total if slot_idx == slot_count - 1 else int(round(total * (slot_idx + 1) / slot_count))
                chunks.append(texto[start:end].strip() or " ")
                start = end
            return chunks

        chunks = []
        start = 0
        for slot_idx in range(slot_count):
            remaining_slots = slot_count - slot_idx
            if remaining_slots == 1:
                chunks.append(" ".join(words[start:]).strip() or " ")
                break
            remaining_words = len(words) - start
            take = max(1, int(round(remaining_words / remaining_slots)))
            chunks.append(" ".join(words[start:start + take]).strip() or " ")
            start += take
        return chunks
