"""Cliente de visión sobre Groq.

Solo sabe de Groq y de imágenes: ni de regiones, ni de prompts de dominio, ni de qué se
hace con la respuesta. El servicio que lo usa depende de esta interfaz, no de la clase,
así que en los tests entra un doble sin tocar la red.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from parallel_manga_translator.infrastructure.execution_control import get_execution_control
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.vision.annotated_page import encode_data_url

logger = get_logger(__name__)


@runtime_checkable
class VisionClientPort(Protocol):
    """Contrato mínimo: imágenes + instrucción -> texto de respuesta."""

    def analyze(self, images: Sequence[np.ndarray], prompt: str) -> str:
        ...


class GroqVisionClient:
    """Llama a un modelo multimodal de Groq con la página anotada y sus recortes."""

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        json_response: bool = True,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.json_response = bool(json_response)
        self._client: Any = None

    def _build_client(self):
        if self._client is not None:
            return self._client
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("El cliente de visión necesita el paquete `groq`.") from exc
        if not self.api_key:
            raise RuntimeError("Falta GROQ_API_KEY para el refinamiento con VLM.")
        if not self.model:
            raise RuntimeError("Define vlm.model en config.yaml: no hay un modelo de visión por defecto seguro.")
        self._client = Groq(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def analyze(self, images: Sequence[np.ndarray], prompt: str) -> str:
        if not images:
            return ""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": encode_data_url(image)}})

        client = self._build_client()
        control = get_execution_control()
        reservation = None
        if control is not None:
            # Mismo checkpoint cooperativo que usan las llamadas de traducción: una pausa
            # o una cancelación deben detener el trabajo también aquí.
            reservation = control.reserve_external_call(kind="llm", provider="groq-vision")
        failed = False
        try:
            request: dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
            if self.json_response:
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request)
            return str(response.choices[0].message.content or "")
        except Exception:
            failed = True
            raise
        finally:
            if control is not None and reservation is not None:
                control.commit_external_call(reservation, failed=failed)

    def describe(self) -> Mapping[str, Any]:
        return {"provider": "groq", "model": self.model}


class NullVisionClient:
    """Cliente que no llama a nada. Es el default: el VLM cuesta dinero por página."""

    def analyze(self, images: Sequence[np.ndarray], prompt: str) -> str:  # noqa: D102
        del images, prompt
        return ""

    def describe(self) -> Mapping[str, Any]:
        return {"provider": "none", "model": ""}


def build_vision_client(
    *,
    enabled: bool,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 60.0,
) -> VisionClientPort:
    """Factory: el pipeline pide un cliente, no elige una clase."""
    if not enabled:
        return NullVisionClient()
    return GroqVisionClient(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
