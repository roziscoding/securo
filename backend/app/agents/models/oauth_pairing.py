"""Short-lived, one-use SIWC remote pairing state."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class OpenAIOAuthPairing(Base):
    __tablename__ = "agent_openai_oauth_pairings"
    __table_args__ = (
        CheckConstraint(
            "state IN ('pending','attempt_registered','completing','completed','cancelled','failed')",
            name="ck_agent_oauth_pairing_state",
        ),
        Index("ix_agent_oauth_pairing_user", "user_id"),
        Index("ix_agent_oauth_pairing_expiry", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_llm_connections.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    secret_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    state: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attempt_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    nonce: Mapped[str | None] = mapped_column(String(256), nullable=True)
    redirect_uri: Mapped[str | None] = mapped_column(String(128), nullable=True)
    completion_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    completion_lease_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    completion_lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    result_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
