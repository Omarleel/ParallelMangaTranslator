from __future__ import annotations

import json

import unittest
from unittest import mock

from parallel_manga_translator.translation.groq_retry import (
    inspect_groq_error,
    parse_duration_seconds,
    retry_delay_seconds,
)


class _Response:
    def __init__(self, status_code=429, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _GroqError(Exception):
    def __init__(self, message, *, status_code=429, headers=None, code="rate_limit_exceeded"):
        super().__init__(message)
        self.status_code = status_code
        self.response = _Response(status_code, headers)
        self.body = {"error": {"message": message, "code": code}}


class GroqRetryTests(unittest.TestCase):
    def test_duration_parser_understands_groq_reset_format(self):
        self.assertAlmostEqual(parse_duration_seconds("2m59.56s"), 179.56, places=2)
        self.assertAlmostEqual(parse_duration_seconds("750ms"), 0.75, places=2)

    def test_tpm_429_prefers_retry_after_header(self):
        exc = _GroqError(
            "Rate limit reached on tokens per minute (TPM). Please try again in 9.9s.",
            headers={
                "retry-after": "4.25",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "4.1s",
            },
        )
        info = inspect_groq_error(exc)
        self.assertEqual(info.limit_scope, "tpm")
        self.assertTrue(info.retryable)
        self.assertFalse(info.daily_limit)
        self.assertAlmostEqual(info.retry_after, 4.25)
        self.assertEqual(info.remaining_tokens, "0")

    def test_tpm_429_can_use_message_when_header_missing(self):
        exc = _GroqError(
            "Rate limit reached on tokens per minute (TPM). Please try again in 5.289s.",
        )
        info = inspect_groq_error(exc)
        self.assertAlmostEqual(info.retry_after, 5.289, places=3)

    def test_daily_limit_is_not_marked_retryable(self):
        exc = _GroqError(
            "Rate limit reached on tokens per day (TPD). Please try again in 7h.",
        )
        info = inspect_groq_error(exc)
        self.assertEqual(info.limit_scope, "tpd")
        self.assertTrue(info.daily_limit)
        self.assertFalse(info.retryable)

    def test_json_validate_failure_is_retryable(self):
        exc = _GroqError(
            "Failed to validate JSON. Please adjust your prompt.",
            status_code=400,
            code="json_validate_failed",
        )
        info = inspect_groq_error(exc)
        self.assertTrue(info.retryable)
        self.assertEqual(info.error_code, "json_validate_failed")

    def test_retry_delay_respects_server_plus_small_buffer(self):
        exc = _GroqError(
            "Rate limit reached on tokens per minute (TPM).",
            headers={"retry-after": "3"},
        )
        info = inspect_groq_error(exc)
        with mock.patch("parallel_manga_translator.translation.groq_retry.random.uniform", return_value=0.0):
            self.assertEqual(retry_delay_seconds(info, attempt=1), 3.5)

    def test_500_uses_exponential_backoff_when_no_server_delay(self):
        exc = _GroqError("server failed", status_code=500, code="server_error")
        info = inspect_groq_error(exc)
        with mock.patch("parallel_manga_translator.translation.groq_retry.random.uniform", return_value=0.0):
            self.assertEqual(retry_delay_seconds(info, attempt=3, base_seconds=1.0, max_backoff_seconds=12), 4.0)


if __name__ == "__main__":
    unittest.main()


class _Cache:
    def get(self, _key):
        return None

    def set(self, _key, _value):
        return None


class _Glossary:
    def as_prompt_text(self):
        return ""

    def apply_to_translation(self, _source, translated):
        return translated


class _CharacterMemory:
    enabled = False

    def snapshot(self):
        return {"characters": [{"name": "SHOULD_NOT_BE_SENT"}]}

    def as_prompt_text(self):
        return "SHOULD_NOT_BE_SENT"


class _Usage:
    prompt_tokens = 100
    completion_tokens = 30
    total_tokens = 130


class _Message:
    content = '{"traducciones":[{"id":0,"traduccion":"Hola"}]}'


class _Choice:
    message = _Message()


class _Completion:
    choices = [_Choice()]
    usage = _Usage()


class _CompletionWithContent:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]
        self.usage = _Usage()


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Client:
    def __init__(self, outcomes):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(outcomes)


from parallel_manga_translator.translation.llm_translation_mixin import LlmTranslationMixin


class _RespaldoFalso:
    """El proveedor tradicional que Groq compone, reducido a contar llamadas."""

    def __init__(self, harness):
        self.harness = harness

    def traducir_textos(self, textos, **kwargs):
        self.harness.traditional_calls += 1
        return ["TRAD" for _ in textos]


class _LlmHarness(LlmTranslationMixin):
    def __init__(self, outcomes, *, fallback=False):
        self.client = _Client(outcomes)
        self.idioma_entrada = "Japonés"
        self.idioma_salida = "Español"
        self.modelo = "qwen/qwen3.8-27b"
        self.seed = 7
        self.max_retries = 3
        self.llm_retry_max_wait_seconds = 90.0
        self.llm_retry_base_seconds = 1.0
        self.llm_retry_max_backoff_seconds = 12.0
        self.llm_retry_jitter_seconds = 0.0
        self.llm_fallback_to_traditional_on_error = fallback
        self.llm_strict_json_schema = True
        self.provider_name = "groq"
        self.lore_manga = ""
        self.cache = _Cache()
        self.glossary = _Glossary()
        self.character_memory = _CharacterMemory()
        self.traditional_calls = 0
        # El respaldo ya no es un metodo heredado del motor tradicional: es un colaborador,
        # asi que el doble tambien lo es.
        self.traditional = _RespaldoFalso(self)

    def _same_language(self):
        return False

    @staticmethod
    def _persistent_key(text, method):
        return f"{method}:{text}"


class GroqRetryIntegrationTests(unittest.TestCase):
    def test_tpm_waits_and_retries_same_page_without_google(self):
        rate_error = _GroqError(
            "Rate limit reached on tokens per minute (TPM). Please try again in 2s.",
            headers={"retry-after": "2", "x-ratelimit-reset-tokens": "1.8s"},
        )
        harness = _LlmHarness([rate_error, _Completion()])
        with mock.patch(
            "parallel_manga_translator.translation.llm_translation_mixin.cooperative_sleep"
        ) as sleep:
            result = harness.traducir_textos_llm(["こんにちは"])
        self.assertEqual(result, ["Hola"])
        self.assertEqual(harness.traditional_calls, 0)
        sleep.assert_called_once_with(2.5)
        self.assertEqual(len(harness.client.chat.completions.requests), 2)

    def test_disabled_character_memory_is_not_in_prompt(self):
        harness = _LlmHarness([_Completion()])
        harness.traducir_textos_llm(["こんにちは"])
        request = harness.client.chat.completions.requests[0]
        joined = "\n".join(message["content"] for message in request["messages"])
        self.assertNotIn("SHOULD_NOT_BE_SENT", joined)

    def test_daily_limit_leaves_page_untouched_by_raising(self):
        daily_error = _GroqError(
            "Rate limit reached on tokens per day (TPD). Please try again in 7h.",
        )
        harness = _LlmHarness([daily_error])
        with self.assertRaisesRegex(RuntimeError, "límite diario"):
            harness.traducir_textos_llm(["こんにちは"])
        self.assertEqual(harness.traditional_calls, 0)


    def test_partial_response_repairs_only_missing_ids(self):
        first = _CompletionWithContent(
            '{"traducciones":[{"id":0,"traduccion":"A"},{"id":1,"traduccion":"B"}]}'
        )
        second = _CompletionWithContent(
            '{"traducciones":[{"id":2,"traduccion":"C"}]}'
        )
        harness = _LlmHarness([first, second])
        with mock.patch(
            "parallel_manga_translator.translation.llm_translation_mixin.cooperative_sleep"
        ) as sleep:
            result = harness.traducir_textos_llm(["uno", "dos", "tres"])

        self.assertEqual(result, ["A", "B", "C"])
        self.assertEqual(harness.traditional_calls, 0)
        self.assertEqual(len(harness.client.chat.completions.requests), 2)
        second_request = harness.client.chat.completions.requests[1]
        payload = json.loads(second_request["messages"][1]["content"])
        self.assertEqual(payload["ids_obligatorios"], [2])
        self.assertEqual([item["id"] for item in payload["textos_a_traducir"]], [2])
        sleep.assert_called_once()

    def test_partial_repair_preserves_noncontiguous_original_ids(self):
        first = _CompletionWithContent(
            '{"traducciones":[{"id":0,"traduccion":"A"},{"id":2,"traduccion":"C"}]}'
        )
        second = _CompletionWithContent(
            '{"traducciones":[{"id":1,"traduccion":"B"}]}'
        )
        harness = _LlmHarness([first, second])
        result = harness.traducir_textos_llm(["uno", "dos", "tres"])
        self.assertEqual(result, ["A", "B", "C"])

    def test_explicit_fallback_option_can_still_use_traditional(self):
        daily_error = _GroqError(
            "Rate limit reached on tokens per day (TPD). Please try again in 7h.",
        )
        harness = _LlmHarness([daily_error], fallback=True)
        self.assertEqual(harness.traducir_textos_llm(["こんにちは"]), ["TRAD"])
        self.assertEqual(harness.traditional_calls, 1)
        self.assertIn("traductor tradicional", harness.llm_fallback_reason)
