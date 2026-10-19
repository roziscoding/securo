"""Allow credential scrubbing and short OAuth lifecycle leases.

Revision ID: 098
Revises: 097
"""
from alembic import op
import sqlalchemy as sa

revision = "098"
down_revision = "097"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "agent_openai_oauth_credentials",
        "token_bundle_encrypted",
        existing_type=sa.Text(),
        nullable=True,
    )
    op.add_column(
        "agent_openai_oauth_credentials",
        sa.Column("lifecycle_lease_token", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "agent_openai_oauth_credentials",
        sa.Column("lifecycle_lease_kind", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "agent_openai_oauth_credentials",
        sa.Column("lifecycle_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "agent_openai_oauth_credentials",
        sa.Column(
            "refresh_attempt_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "agent_openai_oauth_credentials",
        sa.Column("refresh_retry_after", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "agent_openai_oauth_credentials",
        sa.Column("refresh_attempt_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "agent_openai_oauth_credentials",
        sa.Column("refresh_attempt_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "agent_openai_oauth_credentials",
        sa.Column("refresh_restore_status", sa.String(length=32), nullable=True),
    )

    # 097 errors do not carry a durable attempt identity. Treat all legacy
    # terminal/unknown rows conservatively before narrowing the constraint.
    op.execute(
        "UPDATE agent_openai_oauth_credentials SET status = CASE "
        "WHEN status = 'revoked' THEN 'revoked' ELSE 'reauth_required' END, "
        "token_bundle_encrypted = NULL, "
        "last_error_code = CASE WHEN status = 'expired' THEN 'expired' "
        "ELSE 'reauth_required' END "
        "WHERE status IN ('expired', 'error', 'revoked', 'reauth_required')"
    )
    op.drop_constraint("ck_agent_oauth_credential_status", "agent_openai_oauth_credentials", type_="check")
    op.create_check_constraint(
        "ck_agent_oauth_credential_status", "agent_openai_oauth_credentials",
        "status IN ('active', 'disabled', 'revoked', 'reauth_required')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_agent_oauth_credential_status", "agent_openai_oauth_credentials", type_="check")
    op.create_check_constraint(
        "ck_agent_oauth_credential_status", "agent_openai_oauth_credentials",
        "status IN ('active', 'disabled', 'expired', 'revoked', 'error', 'reauth_required')",
    )
    op.drop_column("agent_openai_oauth_credentials", "refresh_restore_status")
    op.drop_column("agent_openai_oauth_credentials", "refresh_attempt_started_at")
    op.drop_column("agent_openai_oauth_credentials", "refresh_attempt_id")
    op.drop_column("agent_openai_oauth_credentials", "refresh_retry_after")
    op.drop_column("agent_openai_oauth_credentials", "refresh_attempt_count")
    op.drop_column("agent_openai_oauth_credentials", "lifecycle_lease_expires_at")
    op.drop_column("agent_openai_oauth_credentials", "lifecycle_lease_kind")
    op.drop_column("agent_openai_oauth_credentials", "lifecycle_lease_token")
    # Terminal rows have no bundle to restore. Removing them preserves the
    # pre-098 non-null invariant without fabricating credential material.
    op.execute(
        "DELETE FROM agent_openai_oauth_credentials "
        "WHERE token_bundle_encrypted IS NULL"
    )
    op.alter_column(
        "agent_openai_oauth_credentials",
        "token_bundle_encrypted",
        existing_type=sa.Text(),
        nullable=False,
    )
