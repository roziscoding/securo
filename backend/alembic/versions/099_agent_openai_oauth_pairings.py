"""Create short-lived OpenAI OAuth pairing sessions.

Revision ID: 099
Revises: 098
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "099"
down_revision = "098"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_openai_oauth_pairings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("connection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("secret_digest", sa.String(64), nullable=True),
        sa.Column("state", sa.String(24), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempt_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("nonce", sa.String(256), nullable=True),
        sa.Column("redirect_uri", sa.String(128), nullable=True),
        sa.Column("completion_digest", sa.String(64), nullable=True),
        sa.Column("completion_lease_token", sa.String(36), nullable=True),
        sa.Column("completion_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result_subject", sa.String(255), nullable=True),
        sa.Column("failure_code", sa.String(64), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "state IN ('pending','attempt_registered','completing','completed','cancelled','failed')",
            name="ck_agent_oauth_pairing_state",
        ),
        sa.ForeignKeyConstraint(["connection_id"], ["agent_llm_connections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("connection_id"),
    )
    op.create_index("ix_agent_oauth_pairing_user", "agent_openai_oauth_pairings", ["user_id"])
    op.create_index("ix_agent_oauth_pairing_expiry", "agent_openai_oauth_pairings", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_agent_oauth_pairing_expiry", table_name="agent_openai_oauth_pairings")
    op.drop_index("ix_agent_oauth_pairing_user", table_name="agent_openai_oauth_pairings")
    op.drop_table("agent_openai_oauth_pairings")
