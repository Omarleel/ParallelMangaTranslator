from __future__ import annotations

from typing import Callable, Dict

from parallel_manga_translator.translation.providers.base import TranslationProvider, TranslationProviderConfig
from parallel_manga_translator.translation.providers.groq_provider import GroqTranslationProvider
from parallel_manga_translator.translation.providers.traditional_provider import TraditionalTranslationProvider

ProviderBuilder = Callable[[TranslationProviderConfig], TranslationProvider]


class TranslatorFactory:
    """Registry abierto/cerrado para proveedores de traducción.

    Un proveedor nuevo solo debe implementar `TranslationProvider` y registrarse aquí.
    El pipeline y `TranslatorManager` permanecen cerrados a modificación.
    """

    _registry: Dict[str, ProviderBuilder] = {}
    _aliases: Dict[str, str] = {
        "tradicional": "traditional",
        "trad": "traditional",
        "google": "traditional",
        "deepl": "traditional",
        "llm": "groq",
    }

    @classmethod
    def register(cls, name: str, builder: ProviderBuilder) -> None:
        cls._registry[name.strip().lower()] = builder

    @classmethod
    def create(cls, config: TranslationProviderConfig) -> TranslationProvider:
        requested = cls._provider_name_from_config(config)
        try:
            return cls._registry[requested](config)
        except KeyError as exc:
            supported = ", ".join(sorted(cls.supported_providers()))
            raise ValueError(f"Proveedor de traducción no soportado: {requested!r}. Soportados: {supported}") from exc

    @classmethod
    def _provider_name_from_config(cls, config: TranslationProviderConfig) -> str:
        method = str(config.method or "traditional").strip().lower()
        if method in {"llm", "ia", "ai"}:
            requested = str(config.llm_provider or "groq").strip().lower()
        else:
            requested = str(config.traditional_provider or "traditional").strip().lower()
            if requested in {"auto", ""}:
                requested = "traditional"
        return cls._aliases.get(requested, requested)

    @classmethod
    def supported_providers(cls) -> set[str]:
        return set(cls._registry) | set(cls._aliases)


TranslatorFactory.register("traditional", TraditionalTranslationProvider)
TranslatorFactory.register("groq", GroqTranslationProvider)
