from __future__ import annotations

import email.utils
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional


_DURATION_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m|h)", re.IGNORECASE)
_TRY_AGAIN_RE = re.compile(
    r"try\s+again\s+in\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|milliseconds?|s|seconds?|m|minutes?|h|hours?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GroqRetryInfo:
    status_code: Optional[int]
    error_code: str
    message: str
    headers: Mapping[str, str]
    limit_scope: str
    retry_after: Optional[float]
    daily_limit: bool
    retryable: bool

    @property
    def remaining_tokens(self) -> str:
        return self.headers.get("x-ratelimit-remaining-tokens", "")

    @property
    def token_reset(self) -> str:
        return self.headers.get("x-ratelimit-reset-tokens", "")

    @property
    def remaining_requests(self) -> str:
        return self.headers.get("x-ratelimit-remaining-requests", "")

    @property
    def request_reset(self) -> str:
        return self.headers.get("x-ratelimit-reset-requests", "")


def _normalise_headers(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    try:
        items = raw.items()
    except AttributeError:
        return {}
    return {str(k).lower(): str(v) for k, v in items}


def exception_headers(exc: BaseException) -> dict[str, str]:
    response = getattr(exc, "response", None)
    headers = _normalise_headers(getattr(response, "headers", None))
    if not headers:
        headers = _normalise_headers(getattr(exc, "headers", None))
    return headers


def exception_status(exc: BaseException) -> Optional[int]:
    for value in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(exc, "status", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?:error\s+code|status(?:\s+code)?)\s*[:=]?\s*(\d{3})", str(exc), re.IGNORECASE)
    return int(match.group(1)) if match else None


def exception_body(exc: BaseException) -> Mapping[str, Any]:
    body = getattr(exc, "body", None)
    return body if isinstance(body, Mapping) else {}


def exception_error_code(exc: BaseException) -> str:
    body = exception_body(exc)
    error = body.get("error") if isinstance(body, Mapping) else None
    if isinstance(error, Mapping):
        code = error.get("code")
        if code:
            return str(code)
    text = str(exc)
    match = re.search(r"['\"]code['\"]\s*:\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
    return match.group(1) if match else ""


def exception_message(exc: BaseException) -> str:
    body = exception_body(exc)
    error = body.get("error") if isinstance(body, Mapping) else None
    if isinstance(error, Mapping) and error.get("message"):
        return str(error["message"])
    if isinstance(body, Mapping) and body.get("message"):
        return str(body["message"])
    return str(exc)


def parse_duration_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
        return numeric if math.isfinite(numeric) and numeric >= 0 else None
    except ValueError:
        pass

    total = 0.0
    matched = False
    for match in _DURATION_RE.finditer(text):
        matched = True
        amount = float(match.group("value"))
        unit = match.group("unit").lower()
        if unit == "ms":
            total += amount / 1000.0
        elif unit == "s":
            total += amount
        elif unit == "m":
            total += amount * 60.0
        elif unit == "h":
            total += amount * 3600.0
    return total if matched else None


def _retry_after_from_header(value: Any) -> Optional[float]:
    parsed = parse_duration_seconds(value)
    if parsed is not None:
        return parsed
    if value is None:
        return None
    try:
        parsed_date = email.utils.parsedate_to_datetime(str(value))
        delay = parsed_date.timestamp() - time.time()
        return max(0.0, delay)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _scope_from_message(message: str) -> str:
    text = message.lower()
    checks = (
        ("itpm", ("(itpm)", "input tokens per minute")),
        ("otpm", ("(otpm)", "output tokens per minute")),
        ("tpm", ("(tpm)", "tokens per minute")),
        ("tpd", ("(tpd)", "tokens per day")),
        ("rpm", ("(rpm)", "requests per minute")),
        ("rpd", ("(rpd)", "requests per day")),
    )
    for scope, needles in checks:
        if any(needle in text for needle in needles):
            return scope
    return "unknown"


def _retry_after_from_message(message: str) -> Optional[float]:
    match = _TRY_AGAIN_RE.search(message)
    if not match:
        return None
    amount = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit.startswith("ms") or unit.startswith("millisecond"):
        return amount / 1000.0
    if unit.startswith("m") and not unit.startswith("ms"):
        return amount * 60.0
    if unit.startswith("h"):
        return amount * 3600.0
    return amount


def inspect_groq_error(exc: BaseException) -> GroqRetryInfo:
    status = exception_status(exc)
    headers = exception_headers(exc)
    message = exception_message(exc)
    code = exception_error_code(exc)
    scope = _scope_from_message(message)

    retry_after = _retry_after_from_header(headers.get("retry-after"))
    if retry_after is None:
        retry_after = _retry_after_from_message(message)
    # x-ratelimit-reset-tokens siempre representa TPM según Groq. Es un buen
    # fallback si un proxy o versión antigua del SDK pierde retry-after.
    if retry_after is None and status == 429 and scope in {"tpm", "itpm", "otpm", "unknown"}:
        retry_after = parse_duration_seconds(headers.get("x-ratelimit-reset-tokens"))

    lower = message.lower()
    daily = scope in {"tpd", "rpd"} or "per day" in lower
    class_name = type(exc).__name__.lower()
    connection_error = any(piece in class_name for piece in ("connection", "timeout"))
    transient_status = status in {408, 409, 429} or (status is not None and status >= 500)
    json_generation_failure = status in {400, 422} and (
        code == "json_validate_failed" or "failed to validate json" in lower
    )
    retryable = bool((transient_status or connection_error or json_generation_failure) and not daily)

    return GroqRetryInfo(
        status_code=status,
        error_code=code,
        message=message,
        headers=headers,
        limit_scope=scope,
        retry_after=retry_after,
        daily_limit=daily,
        retryable=retryable,
    )


def retry_delay_seconds(
    info: GroqRetryInfo,
    *,
    attempt: int,
    base_seconds: float = 1.0,
    max_backoff_seconds: float = 12.0,
    jitter_seconds: float = 0.35,
    rate_limit_buffer_seconds: float = 0.5,
) -> float:
    """Calcula una espera cooperativa para el siguiente intento.

    Para 429 se respeta la espera indicada por Groq. Para errores transitorios sin
    indicación del servidor se usa backoff exponencial corto con jitter.
    """

    if info.retry_after is not None:
        delay = max(0.0, info.retry_after)
        if info.status_code == 429:
            delay += max(0.0, rate_limit_buffer_seconds)
    else:
        exponent = max(0, int(attempt) - 1)
        delay = min(max(0.0, max_backoff_seconds), max(0.0, base_seconds) * (2 ** exponent))
    if jitter_seconds > 0:
        delay += random.uniform(0.0, float(jitter_seconds))
    return max(0.0, delay)
