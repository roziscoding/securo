from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConnectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    kind: str = Field(
        ...,
        description="ollama|openai|anthropic|openai_compatible|openai_chatgpt",
    )
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class ConnectionUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    base_url: Optional[str] = None
    # None = leave unchanged; empty string = clear; non-empty = replace
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    extra: Optional[dict[str, Any]] = None
    is_default: Optional[bool] = None


class ConnectionRead(BaseModel):
    """Note: api_key is NEVER returned. The presence of credentials is
    indicated by `has_api_key`."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    kind: str
    base_url: Optional[str]
    default_model: Optional[str]
    extra: dict[str, Any]
    is_default: bool
    has_api_key: bool
    auth_method: Literal["none", "api_key", "chatgpt_oauth"]
    credential_status: Optional[
        Literal[
            "not_connected",
            "active",
            "disabled",
            "revoked",
            "reauth_required",
        ]
    ] = None
    credential_error_code: Optional[
        Literal[
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
        ]
    ] = None
    account_email: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    access_token_expires_at: Optional[datetime] = None
    needs_reauthorization: bool = False
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_orm_row(cls, row) -> "ConnectionRead":
        credential = row.oauth_credential if row.kind == "openai_chatgpt" else None
        has_api_key = bool(row.api_key_encrypted)
        auth_method = (
            "chatgpt_oauth"
            if row.kind == "openai_chatgpt"
            else ("api_key" if has_api_key else "none")
        )
        credential_status = None
        if row.kind == "openai_chatgpt":
            credential_status = credential.status if credential else "not_connected"
        return cls(
            id=row.id,
            name=row.name,
            kind=row.kind,
            base_url=row.base_url,
            default_model=row.default_model,
            extra={} if row.kind == "openai_chatgpt" else (row.extra or {}),
            is_default=bool(row.is_default),
            has_api_key=has_api_key,
            auth_method=auth_method,
            credential_status=credential_status,
            credential_error_code=(credential.last_error_code if credential else None),
            account_email=credential.account_email if credential else None,
            scopes=list(credential.scopes or []) if credential else [],
            access_token_expires_at=(
                credential.access_token_expires_at if credential else None
            ),
            needs_reauthorization=(
                credential_status in {"revoked", "reauth_required"}
            ),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class OAuthModelRead(BaseModel):
    slug: str = Field(..., min_length=1, max_length=256)
    display_name: str = Field(..., min_length=1, max_length=256)


class ConnectionTestResult(BaseModel):
    ok: bool
    detail: str = Field(..., max_length=512)
    models: Optional[list[str | OAuthModelRead]] = Field(None, max_length=500)
    auth_state: Optional[
        Literal[
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
            "refresh_outcome_unknown",
            "revocation_pending",
        ]
    ] = None
