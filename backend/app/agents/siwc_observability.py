"""Low-cardinality, secret-safe SIWC lifecycle events."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

_LOGGER = logging.getLogger("app.agents.siwc")
_COMPONENTS = frozenset({"pairing", "credentials", "responses"})
_OPERATIONS = frozenset({"start", "attempt", "complete", "cancel", "refresh", "revoke", "inference"})
_OUTCOMES = frozenset({"success", "failure", "cancelled", "partial", "skipped"})
_FAILURE_CODES = frozenset({
    "cancelled",
    "credential_busy",
    "invalid_credential",
    "invalid_request",
    "not_available",
    "provider_unavailable",
    "reauth_required",
    "retryable",
    "internal",
})
_MAX_COUNT = 2_147_483_647

P = ParamSpec("P")
R = TypeVar("R")


def _count(value: int | None) -> int | None:
    if type(value) is not int:
        return None
    return min(max(value, 0), _MAX_COUNT)


def emit_siwc_event(
    *,
    component: str,
    operation: str,
    outcome: str,
    failure_code: str | None = None,
    duration_ms: int | None = None,
    retry_count: int | None = None,
    selected_count: int | None = None,
    processed_count: int | None = None,
) -> None:
    """Emit only fixed enums and bounded counters; logging is best-effort."""
    if component not in _COMPONENTS or operation not in _OPERATIONS or outcome not in _OUTCOMES:
        return
    event: dict[str, str | int] = {
        "event": "siwc.lifecycle",
        "component": component,
        "operation": operation,
        "outcome": outcome,
    }
    safe_failure = failure_code if failure_code in _FAILURE_CODES else None
    if safe_failure is not None:
        event["failure_code"] = safe_failure
    for key, value in (
        ("duration_ms", duration_ms),
        ("retry_count", retry_count),
        ("selected_count", selected_count),
        ("processed_count", processed_count),
    ):
        safe_value = _count(value)
        if safe_value is not None:
            event[key] = safe_value
    try:
        _LOGGER.info(json.dumps(event, separators=(",", ":"), sort_keys=True))
    except BaseException:
        # Observability must never alter credential lifecycle semantics.
        pass


def observe_siwc(*, component: str, operation: str):
    """Observe an async lifecycle boundary without inspecting arguments/errors."""
    def decorate(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(function)
        async def observed(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.monotonic_ns()
            try:
                result = await function(*args, **kwargs)
            except asyncio.CancelledError:
                emit_siwc_event(
                    component=component,
                    operation=operation,
                    outcome="cancelled",
                    failure_code="cancelled",
                    duration_ms=(time.monotonic_ns() - started) // 1_000_000,
                )
                raise
            except BaseException:
                emit_siwc_event(
                    component=component,
                    operation=operation,
                    outcome="failure",
                    failure_code="internal",
                    duration_ms=(time.monotonic_ns() - started) // 1_000_000,
                )
                raise
            emit_siwc_event(
                component=component,
                operation=operation,
                outcome="success",
                duration_ms=(time.monotonic_ns() - started) // 1_000_000,
            )
            return result

        return observed
    return decorate