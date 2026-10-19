"""Strict public schemas for remote SIWC pairing."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairingStart(StrictModel):
    pairing_id: uuid.UUID
    secret: str
    expires_at: datetime


class PairingStatus(StrictModel):
    pairing_id: uuid.UUID
    state: str
    expires_at: datetime


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
