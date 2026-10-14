"""Focused regressions for the OAuth lifecycle contract."""
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from app.agents.services import oauth_refresh, openai_oauth


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """These unit regressions do not use a database."""
    yield


def test_operation_lease_covers_maximum_trusted_timeout_and_margin():
    assert oauth_refresh._OPERATION_LEASE >= timedelta(seconds=420)
    assert oauth_refresh._refresh_validity_margin() >= timedelta(seconds=660)


@pytest.mark.asyncio
@pytest.mark.parametrize("finalizer", ["scrub_credential_bundle", "transition_credential_status", "release_credential_lease"])
async def test_attempt_finalizers_require_live_database_lease(finalizer):
    import uuid
    from types import SimpleNamespace
    from sqlalchemy.dialects import postgresql
    from app.agents.services import oauth_credentials

    class Captured(Exception):
        pass

    class Session:
        def get_bind(self):
            return SimpleNamespace(dialect=postgresql.dialect())

        async def execute(self, statement):
            self.statement = statement
            raise Captured

    session = Session()
    kwargs = {"expected_version": 1}
    if finalizer == "release_credential_lease":
        kwargs["lease_token"] = "attempt-id"
    else:
        kwargs.update(expected_lease_token="attempt-id", status="reauth_required")
    with pytest.raises(Captured):
        await getattr(oauth_credentials, finalizer)(session, uuid.uuid4(), uuid.uuid4(), **kwargs)
    where = str(session.statement.whereclause.compile(dialect=postgresql.dialect()))
    assert "lifecycle_lease_expires_at > clock_timestamp()" in where
    assert "refresh_attempt_id" in where


def test_decrypt_bundle_rejects_duplicate_members(monkeypatch):
    from app.agents.services import oauth_credentials
    monkeypatch.setattr(oauth_credentials, "decrypt", lambda _: '{"access_token":"first","access_token":"second","refresh_token":"r","id_token":"i"}')
    with pytest.raises(ValueError):
        oauth_credentials.decrypt_bundle("ciphertext")


def test_exception_scrub_erases_cyclic_foreign_holders_retained_externally():
    class Holder:
        __slots__ = ("secret", "cycle", "__dict__")
        extra: object

        @property
        def hostile(self):
            raise AssertionError("must not invoke properties")

    holder = Holder()
    other = Holder()
    holder.secret = ("access-secret", b"refresh-secret")
    holder.cycle = other
    holder.extra = {"copy": "id-secret", "owner": holder}
    other.secret = "another-copy"
    other.cycle = holder
    class RetainedError(RuntimeError):
        holder: Holder

    error = RetainedError(holder)
    error.holder = holder
    openai_oauth._discard_exception_secrets(error)
    assert holder.secret is None
    assert holder.cycle is None
    assert holder.__dict__ == {}
    assert other.secret is None
    assert other.cycle is None
    assert error.args == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "delete"])
async def test_credential_create_and_delete_lock_owned_parent_first(operation):
    import uuid
    from types import SimpleNamespace
    from sqlalchemy.dialects import postgresql
    from app.agents.services import oauth_credentials, connection_service

    class Captured(Exception):
        pass

    class Session:
        def get_bind(self):
            return SimpleNamespace(dialect=postgresql.dialect())

        async def execute(self, statement):
            self.statement = statement
            raise Captured

    session = Session()
    with pytest.raises(Captured):
        if operation == "delete":
            await connection_service.delete_connection(cast(Any, session), uuid.uuid4(), uuid.uuid4())
        else:
            await oauth_credentials.create_credential(
                cast(Any, session), uuid.uuid4(), uuid.uuid4(), bundle=oauth_credentials.CredentialBundle("a", "r", "i"),
                issuer="issuer", subject="subject", issued_client_id="client", account_email=None,
                scopes=[], access_token_expires_at=datetime.now(timezone.utc),
            )
    sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in sql
    assert "user_id" in sql


@pytest.mark.asyncio
async def test_postgres_create_delete_race_twenty_iterations(postgres_sessions):
    import asyncio
    import uuid
    from sqlalchemy import select
    from app.models.user import User
    from app.agents.models.connection import LlmConnection
    from app.agents.models.oauth_credential import OpenAIOAuthCredential
    from app.agents.services import oauth_credentials, connection_service

    for iteration in range(20):
        user_id = uuid.uuid4()
        async with postgres_sessions() as session:
            session.add(User(id=user_id, email=f"race-{user_id}@example.com", hashed_password="unused", is_active=True, is_superuser=False, is_verified=True))
            await session.commit()
            parent = await connection_service.create_connection(session, user_id, name=f"race-{iteration}", kind="openai_chatgpt")
            connection_id = parent.id
        barrier = asyncio.Barrier(2)

        async def create():
            async with postgres_sessions() as session:
                await barrier.wait()
                try:
                    await oauth_credentials.create_credential(
                        session, connection_id, user_id, bundle=oauth_credentials.CredentialBundle("a", "r", "i"),
                        issuer="issuer", subject="subject", issued_client_id="client", account_email=None,
                        scopes=[], access_token_expires_at=datetime.now(timezone.utc),
                    )
                    return "created"
                except LookupError:
                    await session.rollback()
                    return "deleted-first"

        async def delete():
            async with postgres_sessions() as session:
                await barrier.wait()
                try:
                    return await connection_service.delete_connection(session, connection_id, user_id)
                except ValueError:
                    await session.rollback()
                    return "conflict"

        created, deleted = await asyncio.gather(create(), delete())
        async with postgres_sessions() as session:
            parent = await session.get(LlmConnection, connection_id)
            credential = (await session.execute(select(OpenAIOAuthCredential).where(OpenAIOAuthCredential.connection_id == connection_id))).scalar_one_or_none()
            if created == "created":
                assert deleted == "conflict"
                assert parent is not None and credential is not None
                assert credential.token_bundle_encrypted is not None
            else:
                assert deleted is True
                assert parent is None and credential is None


def test_agents_disabled_omits_beat_and_runtime_never_opens_database(monkeypatch):
    import runpy
    from pathlib import Path
    from types import SimpleNamespace
    from app.agents import config
    from app.agents.tasks import oauth_refresh as task

    monkeypatch.setattr(config, "get_agent_settings", lambda: SimpleNamespace(enabled=False))
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled task opened database/network")
    monkeypatch.setattr(task, "create_async_engine", forbidden)
    worker = runpy.run_path(str(Path(__file__).parents[1] / "app/worker.py"))
    assert "refresh-agent-oauth-near-expiry" not in worker["celery_app"].conf.beat_schedule
    assert task.refresh_near_expiry.run() == {"disabled": 1}


def test_migration_098_translates_legacy_states_before_narrowing(monkeypatch):
    import importlib.util
    from pathlib import Path
    from alembic import op
    path = Path(__file__).parents[1] / "alembic/versions/098_oauth_bundle_nullable.py"
    spec = importlib.util.spec_from_file_location("migration_098_contract", path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    calls = []
    for name in ("alter_column", "add_column", "drop_constraint", "create_check_constraint"):
        monkeypatch.setattr(op, name, lambda *args, _name=name, **kwargs: calls.append((_name, args)))
    monkeypatch.setattr(op, "execute", lambda statement: calls.append(("execute", str(statement))))
    migration.upgrade()
    translations = [i for i, (kind, args) in enumerate(calls) if kind == "execute" and "UPDATE" in args and "expired" in args and "error" in args]
    checks = [i for i, (kind, args) in enumerate(calls) if kind == "create_check_constraint"]
    assert translations and checks and translations[0] < checks[0]
    assert "token_bundle_encrypted = NULL" in calls[translations[0]][1]


@pytest.mark.asyncio
async def test_postgres_populated_097_upgrade_downgrade_upgrade(postgres_sessions):
    import importlib.util
    import uuid
    from pathlib import Path
    from sqlalchemy import text
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from app.models.user import User
    from app.agents.services import connection_service

    revisions = []
    for filename in ("097_agent_openai_oauth_credentials.py", "098_oauth_bundle_nullable.py"):
        spec = importlib.util.spec_from_file_location(filename, Path(__file__).parents[1] / "alembic/versions" / filename)
        assert spec is not None and spec.loader is not None
        revision = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(revision)
        revisions.append(revision)
    user_id = uuid.uuid4()
    async with postgres_sessions() as session:
        session.add(User(id=user_id, email=f"migration-{user_id}@example.com", hashed_password="unused", is_active=True, is_superuser=False, is_verified=True))
        await session.commit()
        parents = []
        for status in ("expired", "error", "revoked", "reauth_required", "active"):
            parents.append((status, (await connection_service.create_connection(session, user_id, name=status, kind="openai_chatgpt")).id))
        connection = await session.connection()
        schema = connection.sync_connection.get_execution_options()["schema_translate_map"][None]
        await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))

        def migrate(sync):
            with Operations.context(MigrationContext.configure(sync)):
                Operations(MigrationContext.configure(sync)).drop_table("agent_openai_oauth_credentials")
                revisions[0].upgrade()
                for status, parent_id in parents:
                    sync.execute(text("INSERT INTO agent_openai_oauth_credentials (id, connection_id, issuer, subject, issued_client_id, scopes, status, token_bundle_encrypted, access_token_expires_at) VALUES (:id, :parent, 'issuer', 'subject', 'client', '[]', :status, 'encrypted', now())"), {"id": uuid.uuid4(), "parent": parent_id, "status": status})
                revisions[1].upgrade()
                rows = sync.execute(text("SELECT status, token_bundle_encrypted FROM agent_openai_oauth_credentials ORDER BY status")).all()
                assert rows == [
                    ("active", "encrypted"),
                    ("reauth_required", None),
                    ("reauth_required", None),
                    ("reauth_required", None),
                    ("revoked", None),
                ]
                revisions[1].downgrade()
                assert sync.execute(text("SELECT count(*) FROM agent_openai_oauth_credentials")).scalar_one() == 1
                revisions[1].upgrade()
                assert sync.execute(text("SELECT status FROM agent_openai_oauth_credentials")).scalar_one() == "active"
        await connection.run_sync(migrate)
        await session.commit()


def test_implementation_study_documents_current_states_and_security_bounds():
    from pathlib import Path
    study = (Path(__file__).parents[2] / "docs/SIWC_IMPLEMENTATION_STUDY.md").read_text()
    assert "# active, disabled, revoked, reauth_required" in study
    assert "`status=reauth_required`, `last_error_code=refresh_outcome_unknown`" in study
    assert "720" in study and "660" in study
    assert "per_user_salt" in study and "encrypted_auth_metadata" in study


def test_connection_schema_accepts_persisted_expiry_error_code():
    from app.agents.schemas.connection import ConnectionRead

    field = ConnectionRead.model_fields["credential_error_code"]
    assert "expired" in str(field.annotation)


@pytest.mark.asyncio
async def test_delete_locks_fresh_credential_state_until_cascade():
    import uuid
    from types import SimpleNamespace
    from sqlalchemy.dialects import postgresql
    from app.agents.services import connection_service

    class Captured(Exception):
        pass

    class Session:
        calls = 0
        def get_bind(self):
            return SimpleNamespace(dialect=postgresql.dialect())
        async def execute(self, statement):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(kind="openai_chatgpt"))
            self.statement = statement
            raise Captured
    session = Session()
    with pytest.raises(Captured):
        await connection_service.delete_connection(cast(Any, session), uuid.uuid4(), uuid.uuid4())
    sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in sql
    assert session.statement.get_execution_options()["populate_existing"] is True
