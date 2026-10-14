"""add encrypted OpenAI SIWC credentials

Revision ID: 097
Revises: 096
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "097"
down_revision = "096"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
_STATUSES = "'active', 'disabled', 'expired', 'revoked', 'error', 'reauth_required'"


def upgrade() -> None:
    op.create_table(
        "agent_openai_oauth_credentials",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("connection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("issuer", sa.String(length=255), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("issued_client_id", sa.String(length=255), nullable=False),
        sa.Column("account_email", sa.String(length=320), nullable=True),
        sa.Column("scopes", _JSON, nullable=False),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("token_bundle_encrypted", sa.Text(), nullable=False),
        sa.Column("access_token_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("earliest_refresh_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("refresh_token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("credential_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_refreshed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_code", sa.String(length=120), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(f"status IN ({_STATUSES})", name="ck_agent_oauth_credential_status"),
        sa.CheckConstraint("credential_version > 0", name="ck_agent_oauth_credential_version"),
        sa.ForeignKeyConstraint(
            ["connection_id"],
            ["agent_llm_connections.id"],
            name="agent_openai_oauth_credentials_connection_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("connection_id"),
    )


def downgrade() -> None:
    op.drop_table("agent_openai_oauth_credentials")
