"""Strict public schemas for remote SIWC pairing."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PairingState = Literal[
    "pending", "attempt_registered", "completing", "completed", "cancelled", "failed"
]
PairingFailureCode = Literal[
    "transient",
    "revoked",
    "scope_loss",
    "corrupt_token",
    "reauth_required",
    "too_early",
    "race_lost",
    "redirect_rejected",
    "malformed_response",
    "inactive",
    "invalid_client",
    "expired",
    "refresh_outcome_unknown",
    "revocation_pending",
    "resource_cleanup_pending",
    "credential_busy",
    "start_outcome_unknown",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairingStart(StrictModel):
    pairing_id: uuid.UUID
    secret: str
    expires_at: datetime


class PairingStatus(StrictModel):
    pairing_id: uuid.UUID
    state: PairingState
    expires_at: datetime
    failure_code: PairingFailureCode | None = None


class PairingBootstrap(PairingStatus):
    ext_agent_host_id: str = Field(pattern=r"^urn:uuid:[0-9a-f-]{36}$")
    returning_client_id: str | None = Field(default=None, min_length=1, max_length=255)


class AttemptRegistration(StrictModel):
    attempt_id: uuid.UUID
    nonce: str = Field(min_length=32, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=128)


class CompletionRequest(StrictModel):
    attempt_id: uuid.UUID
    issued_client_id: str = Field(min_length=1, max_length=255)
    token_response: dict[str, Any]


class CompletionResult(StrictModel):
    state: str = "completed"
    subject: str
