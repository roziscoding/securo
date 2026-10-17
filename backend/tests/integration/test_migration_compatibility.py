from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text

from app.agents.services import connection_service
from app.models.user import User


def _revision(filename: str):
    path = Path(__file__).parents[2] / "alembic" / "versions" / filename
    spec = importlib.util.spec_from_file_location("integration_" + filename, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_populated_pre097_api_keys_survive_097_through_099_and_reupgrade(
    postgres_sessions,
):
    revisions = [
        _revision("097_agent_openai_oauth_credentials.py"),
        _revision("098_oauth_bundle_nullable.py"),
        _revision("099_agent_openai_oauth_pairings.py"),
    ]
    user_id = uuid.uuid4()
    async with postgres_sessions() as session:
        session.add(
            User(
                id=user_id,
                email=f"migration-layer6-{user_id}@example.test",
                hashed_password="unused",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        await session.commit()
        api_key = await connection_service.create_connection(
            session,
            user_id,
            name="Pre-097 API key",
            kind="openai",
            api_key="pre-097-secret",
            default_model="gpt-api-key",
        )
        oauth_parent = await connection_service.create_connection(
            session, user_id, name="OAuth active", kind="openai_chatgpt"
        )
        terminal_parent = await connection_service.create_connection(
            session, user_id, name="OAuth terminal", kind="openai_chatgpt"
        )
        api_key_id = api_key.id
        oauth_parent_id = oauth_parent.id
        terminal_parent_id = terminal_parent.id
        encrypted_before = api_key.api_key_encrypted
        assert encrypted_before and "pre-097-secret" not in encrypted_before

        connection = await session.connection()
        schema = connection.sync_connection.get_execution_options()["schema_translate_map"][None]
        await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))

        def migrate(sync):
            context = MigrationContext.configure(sync)
            operations = Operations(context)
            with Operations.context(context):
                operations.drop_table("agent_openai_oauth_pairings")
                operations.drop_table("agent_openai_oauth_credentials")
                revisions[0].upgrade()
                revisions[1].upgrade()
                revisions[2].upgrade()
                sync.execute(
                    text(
                        "INSERT INTO agent_openai_oauth_credentials "
                        "(id, connection_id, issuer, subject, issued_client_id, scopes, status, "
                        "token_bundle_encrypted, access_token_expires_at) "
                        "VALUES (:id, :connection, 'https://auth.openai.com', 'subject', "
                        "'client', '[]', 'active', 'encrypted-active', now())"
                    ),
                    {"id": uuid.uuid4(), "connection": oauth_parent_id},
                )
                sync.execute(
                    text(
                        "INSERT INTO agent_openai_oauth_credentials "
                        "(id, connection_id, issuer, subject, issued_client_id, scopes, status, "
                        "token_bundle_encrypted, access_token_expires_at, last_error_code) "
                        "VALUES (:id, :connection, 'https://auth.openai.com', 'terminal', "
                        "'client', '[]', 'revoked', NULL, now(), 'reauth_required')"
                    ),
                    {"id": uuid.uuid4(), "connection": terminal_parent_id},
                )
                sync.execute(
                    text(
                        "INSERT INTO agent_openai_oauth_pairings "
                        "(id, connection_id, user_id, secret_digest, state, expires_at) "
                        "VALUES (:id, :connection, :user, :digest, 'pending', now() + interval '5 minutes')"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "connection": terminal_parent_id,
                        "user": user_id,
                        "digest": "a" * 64,
                    },
                )
                assert sync.execute(
                    text(
                        "SELECT api_key_encrypted FROM agent_llm_connections WHERE id=:id"
                    ),
                    {"id": api_key_id},
                ).scalar_one() == encrypted_before

                # Downgrade 099 deliberately drops ephemeral pairings. Downgrade
                # 098 deliberately removes scrubbed terminal OAuth rows because
                # pre-098 requires a non-null encrypted bundle.
                revisions[2].downgrade()
                revisions[1].downgrade()
                remaining = sync.execute(
                    text(
                        "SELECT connection_id, status FROM agent_openai_oauth_credentials"
                    )
                ).all()
                assert remaining == [(oauth_parent_id, "active")]
                assert sync.execute(
                    text(
                        "SELECT api_key_encrypted FROM agent_llm_connections WHERE id=:id"
                    ),
                    {"id": api_key_id},
                ).scalar_one() == encrypted_before

                revisions[1].upgrade()
                revisions[2].upgrade()
                assert sync.execute(
                    text(
                        "SELECT token_bundle_encrypted FROM agent_openai_oauth_credentials "
                        "WHERE connection_id=:id"
                    ),
                    {"id": oauth_parent_id},
                ).scalar_one() == "encrypted-active"
                assert sync.execute(
                    text("SELECT count(*) FROM agent_openai_oauth_pairings")
                ).scalar_one() == 0
                assert sync.execute(
                    text(
                        "SELECT api_key_encrypted FROM agent_llm_connections WHERE id=:id"
                    ),
                    {"id": api_key_id},
                ).scalar_one() == encrypted_before

                # A full rollback is destructive only to SIWC resources: 099
                # drops transient pairings, 098 drops scrubbed terminal rows,
                # and 097 drops the OAuth credential table itself. The
                # pre-existing API-key connection remains byte-for-byte intact.
                revisions[2].downgrade()
                revisions[1].downgrade()
                revisions[0].downgrade()
                assert sync.execute(
                    text("SELECT to_regclass('agent_openai_oauth_credentials')")
                ).scalar_one() is None
                assert sync.execute(
                    text(
                        "SELECT api_key_encrypted FROM agent_llm_connections WHERE id=:id"
                    ),
                    {"id": api_key_id},
                ).scalar_one() == encrypted_before

                # Restore the fixture schema so teardown sees the current head.
                revisions[0].upgrade()
                revisions[1].upgrade()
                revisions[2].upgrade()

        await connection.run_sync(migrate)
        await session.commit()
