"""Contrato del relleno de globos: qué pide la configuración y qué decide el verificador.

`quality.bubble_fill_strategy` elige el **primer candidato**, no el resultado. Con
`quality.visual_inpaint_verifier` activo —que es el valor por defecto— el verificador
puede rechazarlo y `visual_inpaint_retry` sustituirlo por otro, incluso por un modelo
distinto del configurado en `translation.inpaint_model`.

Medido sobre `dataset_eval`: con `solid` sobre una trama el resultado real es
`configured_inpaint:aot`; con `inpaint` sobre fondo uniforme acaba en `solid_color`; y con
`auto` más `opencv-tela` configurado, en `configured_inpaint:lama_mpe`.

Esto no estaba escrito en ninguna parte y los comentarios de `config.yaml` afirmaban lo
contrario ("usa siempre color sólido local"). La consecuencia fue un test que medía una
configuración que nadie ejecuta. El contrato vive aquí para que sea legible y comprobable,
y cada región deja constancia en su metadata de si se respetó lo que se pidió.

Para que la estrategia sea la última palabra: `visual_inpaint_retry: false`.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

#: Prefijo de `bubble_fill_method` que corresponde a cada estrategia pedida.
_METODO_ESPERADO = {
    "solid": "solid_color",
    "inpaint": "configured_inpaint",
}


def strategy_honored(*, requested: str, method: str) -> bool:
    """¿El relleno entregado es el que pedía la estrategia configurada?

    `auto` se considera siempre respetada: delega la elección en el contenido del fondo
    por definición, así que no hay nada que incumplir.
    """
    pedida = str(requested or "").strip().lower()
    entregado = str(method or "").strip().lower()
    esperado = _METODO_ESPERADO.get(pedida)
    if esperado is None:
        return True
    return entregado.startswith(esperado)


class BubbleFillPolicy:
    """Decide QUÉ candidato de relleno se intenta, no cómo se aplica ni si es bueno.

    Siete métodos que vivían en `CleanInpaintingPipelineMixin`, de sus 37. Cinco no
    tocaban estado y los otros dos sólo leían configuración, así que heredarlos servía
    únicamente para que el resto del mixin los llamara por `self`.

    Aplicar el relleno y verificarlo siguen en el pipeline: aquí sólo está la elección.
    """

    def __init__(
        self,
        *,
        bubble_fill_strategy: str = "inpaint",
        visual_inpaint_retry: bool = True,
        visual_inpaint_retry_models: str = "solid,opencv-tela,lama_mpe,aot",
        visual_inpaint_max_retries: int = 4,
    ) -> None:
        self.bubble_fill_strategy = str(bubble_fill_strategy or "inpaint")
        self.visual_inpaint_retry = bool(visual_inpaint_retry)
        self.visual_inpaint_retry_models = str(visual_inpaint_retry_models or "")
        self.visual_inpaint_max_retries = int(visual_inpaint_max_retries)

    def is_color_page(self, image: np.ndarray, threshold: float = 5.0) -> bool:
        """
        Detecta si una página es color o escala de grises.

        Si los canales RGB son prácticamente iguales en toda la imagen,
        se considera B/N.
        """
        if image.ndim != 3 or image.shape[2] < 3:
            return False

        b, g, r = cv2.split(image.astype(np.float32))

        rg = np.mean(np.abs(r - g))
        rb = np.mean(np.abs(r - b))
        gb = np.mean(np.abs(g - b))

        color_score = (rg + rb + gb) / 3.0

        return color_score > threshold

    def resolve_auto_inpaint_model(self, image: np.ndarray) -> str:
        if self.is_color_page(image):
            return "lama_mpe"

        return "B/N"

    def auto_inpaint_candidate(
        self,
        imagen: np.ndarray,
        background_variation: float,
        variation_threshold: float,
    ) -> str:
        """Resuelve ``inpaint_model: auto`` por región, no por página.

        LaMa solo donde hay que reconstruir estructura: página a color, o fondo con
        textura —trama, degradado, arte— medido por la variación del fondo de esa región.
        Sobre un fondo plano en blanco y negro no aporta nada y cuesta GPU, así que ahí se
        deja el relleno plano, que luego se convierte en ``opencv-tela`` como reserva.

        La resolución anterior era por página y solo miraba si era a color, de modo que
        una página monocroma con trama nunca llegaba a LaMa.
        """
        if self.is_color_page(imagen):
            return "lama_mpe"
        if float(background_variation) >= float(variation_threshold):
            return "lama_mpe"
        return "solid"

    @staticmethod
    def background_variation_score(region_img: np.ndarray, local_mask: np.ndarray, exclude_mask: Optional[np.ndarray] = None) -> float:
        """Mide variación del fondo seguro."""
        if region_img.size == 0 or local_mask is None or getattr(local_mask, "size", 0) == 0:
            return 0.0
        sample_mask = BubbleFillPolicy.background_sample_mask(local_mask, exclude_mask)
        if sample_mask.size == 0 or cv2.countNonZero(sample_mask) == 0:
            return 0.0
        pixels = region_img[sample_mask > 0]
        if pixels.size == 0:
            return 0.0
        pixels = pixels.reshape(-1, 3).astype(np.float32)
        luma = pixels[:, 0] * 0.114 + pixels[:, 1] * 0.587 + pixels[:, 2] * 0.299
        return float(np.std(luma))

    def normalize_inpaint_candidate(self, model_name: str) -> str:
        model_name = str(model_name or "").strip()
        aliases = {
            "": "opencv-tela",
            "opencv": "opencv-tela",
            "cv2": "opencv-tela",
            "opencv_ns": "opencv-tela",
            "opencv-tela": "opencv-tela",
            "tela": "opencv-tela",
            "solid_color": "solid",
            "solid-fill": "solid",
            "bn": "solid",
            "b/n": "solid",
            "B/N": "solid",
            "lama": "lama_mpe",
            "lama-mpe": "lama_mpe",
            "lama_mpe": "lama_mpe",
            "lama_large": "lama_large_512px",
            "lama_large_512px": "lama_large_512px",
            "aot": "aot",
            "auto": "auto",
        }
        return aliases.get(model_name, aliases.get(model_name.lower(), model_name))

    def visual_retry_candidates(self, initial_candidate: str) -> List[str]:
        candidates: List[str] = []

        def _add(raw_name: str) -> None:
            name = self.normalize_inpaint_candidate(raw_name)
            if name == "auto":
                # auto se resuelve contra la página/crop al ejecutar; se conserva solo una vez.
                name = "auto"
            if name and name not in candidates:
                candidates.append(name)

        _add(initial_candidate)
        if bool(getattr(self, "visual_inpaint_retry", True)):
            raw_models = str(getattr(self, "visual_inpaint_retry_models", "solid,opencv-tela,lama_mpe,aot") or "")
            for token in raw_models.replace(";", ",").split(","):
                token = token.strip()
                if token:
                    _add(token)

        max_retries = max(0, int(getattr(self, "visual_inpaint_max_retries", 4) or 0))
        max_attempts = 1 + max_retries if bool(getattr(self, "visual_inpaint_retry", True)) else 1
        return candidates[:max_attempts]

    def normalized_fill_strategy(self) -> str:
        fill_strategy = str(getattr(self, "bubble_fill_strategy", "inpaint") or "inpaint").strip().lower()
        if fill_strategy in {"configured_inpaint", "config_inpaint", "lama"}:
            return "inpaint"
        if fill_strategy in {"adaptive"}:
            return "auto"
        return fill_strategy if fill_strategy in {"inpaint", "auto", "solid"} else "inpaint"

    @staticmethod
    def background_sample_mask(local_mask: np.ndarray, exclude_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Devuelve la zona de fondo: región segura menos tinta a borrar."""
        if local_mask is None or getattr(local_mask, "size", 0) == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        base_mask = (local_mask > 0).astype(np.uint8)
        if cv2.countNonZero(base_mask) == 0:
            return base_mask

        if exclude_mask is None or not getattr(exclude_mask, "size", 0):
            return base_mask

        exclusion = (exclude_mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_exclusion = cv2.dilate(exclusion, kernel, iterations=1)
        sample_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(dilated_exclusion))

        if cv2.countNonZero(sample_mask) < max(16, int(cv2.countNonZero(base_mask) * 0.04)):
            sample_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(exclusion))
        if cv2.countNonZero(sample_mask) == 0:
            sample_mask = base_mask
        return sample_mask
