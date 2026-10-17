from __future__ import annotations

import asyncio
import json
import logging

import pytest

from app.agents.siwc_observability import emit_siwc_event, observe_siwc


SENTINELS = (
    "secret-token-value",
    "Bearer abc.def.ghi",
    "person@example.test",
    "subject-123",
    "https://provider.invalid/private",
    "ciphertext-material",
    "gpt-private-model",
    "raw exception detail",
)


def _events(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [json.loads(record.message) for record in caplog.records]


@pytest.mark.asyncio
async def test_observed_success_failure_and_cancel_are_allowlisted_and_secret_free(caplog):
    @observe_siwc(component="pairing", operation="attempt")
    async def succeeds(payload: str) -> None:
        assert payload == SENTINELS[0]

    @observe_siwc(component="pairing", operation="complete")
    async def fails() -> None:
        raise RuntimeError(" ".join(SENTINELS))

    @observe_siwc(component="pairing", operation="cancel")
    async def cancels() -> None:
        raise asyncio.CancelledError(SENTINELS[1])

    with caplog.at_level(logging.INFO, logger="app.agents.siwc"):
        await succeeds(SENTINELS[0])
        with pytest.raises(RuntimeError, match="raw exception detail"):
            await fails()
        with pytest.raises(asyncio.CancelledError):
            await cancels()

    events = _events(caplog)
    assert [event["outcome"] for event in events] == ["success", "failure", "cancelled"]
    assert all(
        set(event) <= {
            "event", "component", "operation", "outcome", "failure_code",
            "duration_ms", "retry_count", "selected_count", "processed_count",
        }
        for event in events
    )
    serialized = json.dumps(events)
    for sentinel in SENTINELS:
        assert sentinel not in serialized


def test_event_emission_rejects_unapproved_values_and_logging_failure_is_nonfatal(
    caplog, monkeypatch
):
    logger = logging.getLogger("app.agents.siwc")

    with caplog.at_level(logging.INFO, logger="app.agents.siwc"):
        emit_siwc_event(
            component="responses",
            operation="inference",
            outcome="failure",
            failure_code="raw exception detail",
        )
        emit_siwc_event(
            component="secret-token-value",
            operation="https://provider.invalid/private",
            outcome="person@example.test",
        )
    assert _events(caplog) == [{
        "component": "responses",
        "event": "siwc.lifecycle",
        "operation": "inference",
        "outcome": "failure",
    }]

    def broken_log(*_args, **_kwargs):
        raise RuntimeError("logger unavailable")

    monkeypatch.setattr(logger, "info", broken_log)
    emit_siwc_event(component="responses", operation="inference", outcome="success")