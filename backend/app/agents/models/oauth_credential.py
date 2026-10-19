"""Encrypted OpenAI SIWC credential attached to one LLM connection."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.agents.models.connection import LlmConnection


class OpenAIOAuthCredential(Base):
    __tablename__ = "agent_openai_oauth_credentials"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'disabled', 'revoked', "
            "'reauth_required')",
            name="ck_agent_oauth_credential_status",
        ),
        CheckConstraint(
            "credential_version > 0", name="ck_agent_oauth_credential_version"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_llm_connections.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    issuer: Mapped[str] = mapped_column(String(255))
    subject: Mapped[str] = mapped_column(String(255))
    issued_client_id: Mapped[str] = mapped_column(String(255))
    account_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    scopes: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), default=list
    )
    status: Mapped[str] = mapped_column(String(32), default="active", server_default="active")
    # NULL is the safe terminal representation after revocation or definitive
    # invalidation. Metadata remains so callers can explain recovery.
    token_bundle_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    access_token_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    earliest_refresh_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    refresh_token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    credential_version: Mapped[int] = mapped_column(
        Integer, default=1, server_default="1", nullable=False
    )
    last_refreshed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error_code: Mapped[str | None] = mapped_column(String(120), nullable=True)
    refresh_attempt_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    refresh_retry_after: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # A committed attempt marker means the rotating refresh token may have
    # been consumed. It is retained until explicit reauthorization/revocation.
    refresh_attempt_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    refresh_attempt_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    refresh_restore_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Short, committed leases serialize refresh/revoke across processes without
    # retaining a database transaction or pool connection during provider I/O.
    lifecycle_lease_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lifecycle_lease_kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    lifecycle_lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    connection: Mapped["LlmConnection"] = relationship(back_populates="oauth_credential")
