"""Periodic near-expiry OAuth refresh task coverage."""
from __future__ import annotations

import asyncio
from collections import Counter
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from billiard.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.services import connection_service
from app.agents.services.oauth_credentials import (
    RESOURCE_CLEANUP_PENDING,
    CredentialBundle,
    create_credential,
)
from app.agents.services.openai_oauth import OAuthFailureCode, OAuthLifecycleError
from app.models.user import User
from app.worker import celery_app


async def _add_credential(session, test_user, *, expires_at, earliest_at, status="active"):
    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT task", kind="openai_chatgpt"
    )
    await create_credential(
        session,
        connection.id,
        test_user.id,
        bundle=CredentialBundle("access", "refresh", "id"),
        issuer="https://auth.openai.com",
        subject="subject",
        issued_client_id="oaiapp_test",
        account_email=None,
        scopes=["openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"],
        access_token_expires_at=expires_at,
        refresh_token_expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        earliest_refresh_at=earliest_at,
        status=status,
    )
    return connection


@pytest.mark.asyncio
async def test_periodic_task_refreshes_only_due_and_allowed_credentials(
    session, test_user, monkeypatch
):
    from app.agents.tasks.oauth_refresh import _do_refresh_due

    now = datetime.now(timezone.utc)
    due = await _add_credential(
        session, test_user, expires_at=now + timedelta(minutes=4), earliest_at=now
    )
    await _add_credential(
        session, test_user, expires_at=now + timedelta(hours=1), earliest_at=now
    )
    await _add_credential(
        session,
        test_user,
        expires_at=now + timedelta(minutes=4),
        earliest_at=now + timedelta(minutes=20),
    )
    await _add_credential(
        session,
        test_user,
        expires_at=now,
        earliest_at=now,
        status="disabled",
    )
    calls = []

    async def fake_refresh(db, connection_id, user_id, **kwargs):
        calls.append((connection_id, user_id))
        return "redacted"

    monkeypatch.setattr("app.agents.tasks.oauth_refresh.refresh_credential", fake_refresh)
    result = await _do_refresh_due(
        async_sessionmaker(session.bind, expire_on_commit=False), now=now
    )
    assert result == {
        "selected": 1,
        "refreshed": 1,
        "transient_failures": 0,
        "definitive_failures": 0,
        "unexpected_failures": 0,
    }
    assert calls == [(due.id, test_user.id)]


@pytest.mark.asyncio
async def test_periodic_task_never_selects_cleanup_pending_credential(
    session, test_user, monkeypatch
):
    from app.agents.tasks.oauth_refresh import _do_refresh_due

    now = datetime.now(timezone.utc)
    connection = await _add_credential(
        session, test_user, expires_at=now, earliest_at=now
    )
    credential = await session.scalar(
        select(OpenAIOAuthCredential).where(
            OpenAIOAuthCredential.connection_id == connection.id
        )
    )
    assert credential is not None
    credential.last_error_code = RESOURCE_CLEANUP_PENDING
    credential.lifecycle_lease_token = str(uuid.uuid4())
    credential.lifecycle_lease_kind = "resource"
    credential.lifecycle_lease_expires_at = now - timedelta(seconds=1)
    await session.commit()
    calls = 0

    async def should_not_refresh(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        "app.agents.tasks.oauth_refresh.refresh_credential", should_not_refresh
    )
    result = await _do_refresh_due(
        async_sessionmaker(session.bind, expire_on_commit=False), now=now
    )

    assert result["selected"] == 0
    assert calls == 0


@pytest.mark.asyncio
async def test_periodic_task_isolates_transient_failure(session, test_user, monkeypatch):
    from app.agents.tasks.oauth_refresh import _do_refresh_due

    now = datetime.now(timezone.utc)
    await _add_credential(
        session, test_user, expires_at=now + timedelta(minutes=4), earliest_at=now
    )

    async def fail(*args, **kwargs):
        raise OAuthLifecycleError(OAuthFailureCode.TRANSIENT, "temporary", retryable=True)

    monkeypatch.setattr("app.agents.tasks.oauth_refresh.refresh_credential", fail)
    result = await _do_refresh_due(
        async_sessionmaker(session.bind, expire_on_commit=False), now=now
    )
    assert result["transient_failures"] == 1
    assert result["definitive_failures"] == 0


@pytest.mark.asyncio
async def test_periodic_task_uses_retryable_flag_not_definitive_flag(
    session, test_user, monkeypatch
):
    from app.agents.tasks.oauth_refresh import _do_refresh_due

    now = datetime.now(timezone.utc)
    await _add_credential(session, test_user, expires_at=now, earliest_at=now)

    async def fail(*args, **kwargs):
        raise OAuthLifecycleError(
            OAuthFailureCode.MALFORMED_RESPONSE,
            "malformed",
            definitive=False,
            retryable=False,
        )

    monkeypatch.setattr("app.agents.tasks.oauth_refresh.refresh_credential", fail)
    result = await _do_refresh_due(
        async_sessionmaker(session.bind, expire_on_commit=False), now=now
    )
    assert result["transient_failures"] == 0
    assert result["definitive_failures"] == 1


def test_periodic_candidate_query_is_ordered_and_bounded():
    from app.agents.tasks import oauth_refresh

    statement = oauth_refresh._due_credentials_statement(
        datetime.now(timezone.utc), limit=100
    )
    compiled = str(statement.compile(compile_kwargs={"literal_binds": True})).lower()
    assert "order by" in compiled
    assert "limit 100" in compiled


def test_periodic_task_is_registered_with_bounded_retries_and_schedule():
    from app.agents.tasks import oauth_refresh  # noqa: F401

    task = celery_app.tasks["app.agents.tasks.oauth_refresh.refresh_near_expiry"]
    assert task.max_retries == 1
    assert task.soft_time_limit == 55
    assert task.time_limit == 65
    schedule = celery_app.conf.beat_schedule["refresh-agent-oauth-near-expiry"]
    assert schedule["task"] == task.name
    assert 0 < schedule["schedule"] <= 300
    assert (
        task.time_limit * (task.max_retries + 1)
        + task.default_retry_delay * task.max_retries
        <= 180
    )
    assert "app.agents.tasks.oauth_refresh" in celery_app.conf.include
    assert task.max_retries == 1


@pytest.mark.asyncio
async def test_periodic_task_isolates_unexpected_exception_per_credential(
    session, test_user, monkeypatch
):
    from app.agents.tasks.oauth_refresh import _do_refresh_due
    now = datetime.now(timezone.utc)
    await _add_credential(session, test_user, expires_at=now, earliest_at=now)
    async def fail(*args, **kwargs):
        raise RuntimeError("unexpected")
    monkeypatch.setattr("app.agents.tasks.oauth_refresh.refresh_credential", fail)
    result = await _do_refresh_due(async_sessionmaker(session.bind, expire_on_commit=False), now=now)
    assert result["unexpected_failures"] == 1


def test_periodic_task_invokes_and_exhausts_bounded_retry(monkeypatch):
    from app.agents.tasks import oauth_refresh
    async def transient_result():
        return {
            "selected": 1, "refreshed": 0, "transient_failures": 1,
            "definitive_failures": 0, "unexpected_failures": 0,
        }
    monkeypatch.setattr(oauth_refresh, "_run_periodic_refresh", transient_result)
    result = oauth_refresh.refresh_near_expiry.apply(throw=False)
    assert result.failed()
    assert result.result.__class__.__name__ == "MaxRetriesExceededError"
    assert result.result.task_args == ()


def test_periodic_task_engine_hides_parameters(monkeypatch):
    from app.agents.tasks import oauth_refresh
    seen = {}
    class Engine:
        async def dispose(self):
            pass
    def fake_engine(url, **kwargs):
        seen.update(kwargs)
        return Engine()
    async def fake_do_refresh(session_maker):
        return {"selected": 0, "refreshed": 0, "transient_failures": 0, "definitive_failures": 0, "unexpected_failures": 0}
    monkeypatch.setattr(oauth_refresh, "create_async_engine", fake_engine)
    monkeypatch.setattr(oauth_refresh, "async_sessionmaker", lambda *args, **kwargs: object())
    monkeypatch.setattr(oauth_refresh, "_do_refresh_due", fake_do_refresh)
    import asyncio
    asyncio.run(oauth_refresh._run_periodic_refresh())
    assert seen["hide_parameters"] is True


@pytest.mark.asyncio
async def test_periodic_soft_time_limit_is_never_swallowed(session, test_user, monkeypatch):
    from app.agents.tasks.oauth_refresh import _do_refresh_due

    now = datetime.now(timezone.utc)
    await _add_credential(session, test_user, expires_at=now, earliest_at=now)

    async def stop_now(*args, **kwargs):
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr("app.agents.tasks.oauth_refresh.refresh_credential", stop_now)
    with pytest.raises(SoftTimeLimitExceeded):
        await _do_refresh_due(async_sessionmaker(session.bind, expire_on_commit=False), now=now)


def test_periodic_task_boundary_propagates_same_soft_time_limit(monkeypatch):
    from app.agents.tasks import oauth_refresh

    limit = SoftTimeLimitExceeded()

    async def stop():
        raise limit

    monkeypatch.setattr(oauth_refresh, "_run_periodic_refresh", stop)
    with pytest.raises(SoftTimeLimitExceeded) as exc_info:
        oauth_refresh.refresh_near_expiry.run()

    assert exc_info.value is limit


def test_periodic_task_boundary_scrubs_soft_time_limit_in_place(monkeypatch):
    from app.agents.tasks import oauth_refresh

    sentinel = "periodic-soft-limit-sentinel"

    class SlottedSoftLimit(SoftTimeLimitExceeded):
        __slots__ = ("nested",)

    limit = SlottedSoftLimit(sentinel)
    limit.nested = {"secret": bytearray(sentinel.encode())}
    try:
        raise RuntimeError(sentinel)
    except RuntimeError as cause:
        limit.__cause__ = cause

    async def stop():
        raise limit

    monkeypatch.setattr(oauth_refresh, "_run_periodic_refresh", stop)
    with pytest.raises(SoftTimeLimitExceeded) as exc_info:
        oauth_refresh.refresh_near_expiry.run()

    assert exc_info.value is limit
    assert limit.args == ()
    assert limit.__cause__ is None
    assert limit.__context__ is None
    assert limit.nested is None


@pytest.mark.asyncio
async def test_periodic_jwks_soft_limit_scrubs_graph_without_committing_refresh_intent(
    session, test_user, monkeypatch
):
    from app.agents.services import openai_oauth
    from app.agents.tasks.oauth_refresh import _do_refresh_due

    sentinel = "periodic-provider-soft-limit-sentinel"
    now = datetime.now(timezone.utc)
    connection = await _add_credential(session, test_user, expires_at=now, earliest_at=now)
    connection_id = connection.id
    retained_request = httpx.Request(
        "GET",
        openai_oauth.JWKS_ENDPOINT,
        headers={"authorization": sentinel},
        content=sentinel,
    )
    retained_response = httpx.Response(
        503,
        request=retained_request,
        content=sentinel,
        extensions={"nested": {"secret": bytearray(sentinel.encode())}},
    )

    class SlottedSoftLimit(SoftTimeLimitExceeded):
        __slots__ = ("retained",)

    limit = SlottedSoftLimit(sentinel)
    limit.retained = retained_response

    def provider(request):
        raise limit

    transport = httpx.MockTransport(provider)
    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", lambda: transport)

    with pytest.raises(SoftTimeLimitExceeded) as exc_info:
        await _do_refresh_due(async_sessionmaker(session.bind, expire_on_commit=False), now=now)

    assert exc_info.value is limit
    assert limit.args == ()
    assert limit.retained is None
    assert retained_request.content == b""
    assert retained_response.content == b""
    session.expire_all()
    credential = (
        await session.execute(
            select(OpenAIOAuthCredential).where(
                OpenAIOAuthCredential.connection_id == connection_id
            )
        )
    ).scalar_one()
    assert credential.lifecycle_lease_token is None
    assert credential.lifecycle_lease_kind is None
    assert credential.lifecycle_lease_expires_at is None
    assert credential.refresh_attempt_id is None
    assert credential.refresh_attempt_started_at is None
    assert credential.status == "active"
    assert credential.last_error_code is None
    assert credential.credential_version == 1
    assert credential.token_bundle_encrypted is not None


@pytest.mark.asyncio
async def test_periodic_durable_backoff_prevents_first_100_starvation(
    session, test_user, monkeypatch
):
    from app.agents.tasks.oauth_refresh import _do_refresh_due

    now = datetime.now(timezone.utc)
    connections = [
        await _add_credential(
            session,
            test_user,
            expires_at=now - timedelta(minutes=200 - index),
            earliest_at=now,
        )
        for index in range(101)
    ]
    first_batch = {connection.id for connection in connections[:100]}
    first_connection_id = connections[0].id
    calls: list = []

    async def fail_first_hundred(db, connection_id, user_id, **kwargs):
        calls.append(connection_id)
        if connection_id in first_batch:
            raise OAuthLifecycleError(OAuthFailureCode.TRANSIENT, "temporary", retryable=True)
        return "later-row"

    monkeypatch.setattr(
        "app.agents.tasks.oauth_refresh.refresh_credential", fail_first_hundred
    )
    maker = async_sessionmaker(session.bind, expire_on_commit=False)
    first = await _do_refresh_due(maker, now=now)
    second = await _do_refresh_due(maker, now=now)

    assert first["selected"] == 100
    assert first["transient_failures"] == 100
    assert second["selected"] == 1
    assert second["refreshed"] == 1
    assert calls[-1] == connections[100].id
    session.expire_all()
    attempts = (
        await session.execute(
            select(OpenAIOAuthCredential).where(
                OpenAIOAuthCredential.connection_id == first_connection_id
            )
        )
    ).scalar_one()
    assert attempts.refresh_attempt_count == 1
    assert attempts.refresh_retry_after is not None


@pytest.mark.asyncio
async def test_stale_attempt_bookkeeping_does_not_overwrite_rotated_bundle_state(
    session, test_user
):
    from app.agents.tasks.oauth_refresh import _record_attempt

    now = datetime.now(timezone.utc)
    connection = await _add_credential(
        session, test_user, expires_at=now, earliest_at=now
    )
    connection_id = connection.id
    user_id = test_user.id
    credential = (
        await session.execute(
            select(OpenAIOAuthCredential).where(
                OpenAIOAuthCredential.connection_id == connection_id
            )
        )
    ).scalar_one()
    observed_version = credential.credential_version
    observed_expiry = credential.access_token_expires_at
    await session.execute(
        update(OpenAIOAuthCredential)
        .where(OpenAIOAuthCredential.connection_id == connection_id)
        .values(
            credential_version=observed_version + 1,
            refresh_attempt_count=0,
            refresh_retry_after=None,
            access_token_expires_at=now + timedelta(hours=1),
        )
    )
    await session.commit()

    changed = await _record_attempt(
        session,
        connection_id,
        user_id,
        expected_version=observed_version,
        observed_access_expiry=observed_expiry,
        failed=True,
        current_time=now,
    )
    assert changed is False
    session.expire_all()
    winner = (
        await session.execute(
            select(OpenAIOAuthCredential).where(
                OpenAIOAuthCredential.connection_id == connection_id
            )
        )
    ).scalar_one()
    assert winner.refresh_attempt_count == 0
    assert winner.refresh_retry_after is None


@pytest.mark.asyncio
async def test_postgres_stale_attempt_bookkeeping_cannot_mutate_rotation_winner(
    postgres_sessions,
):
    from app.agents.tasks.oauth_refresh import _record_attempt

    now = datetime.now(timezone.utc)
    user_id = uuid.uuid4()
    async with postgres_sessions() as setup:
        user = User(
            id=user_id,
            email=f"periodic-stale-{user_id}@example.com",
            hashed_password="unused",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )
        setup.add(user)
        await setup.commit()
        connection = await _add_credential(
            setup, user, expires_at=now, earliest_at=now
        )
        connection_id = connection.id

    async with postgres_sessions() as stale, postgres_sessions() as winner:
        observed = (
            await stale.execute(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == connection_id
                )
            )
        ).scalar_one()
        observed_version = observed.credential_version
        observed_expiry = observed.access_token_expires_at
        await winner.execute(
            update(OpenAIOAuthCredential)
            .where(OpenAIOAuthCredential.connection_id == connection_id)
            .values(
                credential_version=observed_version + 1,
                access_token_expires_at=now + timedelta(hours=1),
                refresh_attempt_count=0,
                refresh_retry_after=None,
            )
        )
        await winner.commit()
        assert not await _record_attempt(
            stale,
            connection_id,
            user_id,
            expected_version=observed_version,
            observed_access_expiry=observed_expiry,
            failed=True,
            current_time=now,
        )
        current = (
            await winner.execute(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == connection_id
                )
            )
        ).scalar_one()
        assert current.refresh_attempt_count == 0
        assert current.refresh_retry_after is None


@pytest.mark.asyncio
async def test_postgres_concurrent_periodic_workers_claim_fair_unique_batches(
    postgres_sessions, monkeypatch
):
    from app.agents.tasks.oauth_refresh import _do_refresh_due

    now = datetime.now(timezone.utc)
    user_id = uuid.uuid4()
    async with postgres_sessions() as setup:
        user = User(
            id=user_id,
            email=f"periodic-fairness-{user_id}@example.com",
            hashed_password="unused",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )
        setup.add(user)
        await setup.commit()
        connections = [
            await _add_credential(
                setup,
                user,
                expires_at=now - timedelta(minutes=400 - index),
                earliest_at=now,
            )
            for index in range(205)
        ]
    failure_ids = {connection.id for connection in connections[:5]}
    all_ids = {connection.id for connection in connections}
    calls: list[uuid.UUID] = []

    async def refresh_once(db, connection_id, user_id, **kwargs):
        calls.append(connection_id)
        await asyncio.sleep(0)
        if connection_id in failure_ids:
            raise OAuthLifecycleError(OAuthFailureCode.TRANSIENT, "temporary", retryable=True)
        await db.execute(
            update(OpenAIOAuthCredential)
            .where(OpenAIOAuthCredential.connection_id == connection_id)
            .values(
                access_token_expires_at=now + timedelta(hours=1),
                credential_version=OpenAIOAuthCredential.credential_version + 1,
                refresh_retry_after=None,
            )
        )
        await db.commit()
        return "fresh"

    monkeypatch.setattr("app.agents.tasks.oauth_refresh.refresh_credential", refresh_once)
    results = await asyncio.gather(
        *[_do_refresh_due(postgres_sessions, now=now) for _ in range(3)]
    )
    for _ in range(4):
        follow_up = await _do_refresh_due(postgres_sessions, now=now)
        results.append(follow_up)
        if follow_up["selected"] == 0:
            break

    counts = Counter(calls)
    assert len(counts) == 205
    assert len(counts) > 100
    assert set(counts) == all_ids
    assert set(counts.values()) == {1}
    assert sum(result["selected"] for result in results) == 205

    async with postgres_sessions() as verify:
        rows = list(
            (
                await verify.execute(
                    select(OpenAIOAuthCredential).where(
                        OpenAIOAuthCredential.connection_id.in_(all_ids)
                    )
                )
            ).scalars()
        )
    by_id = {row.connection_id: row for row in rows}
    for connection_id in failure_ids:
        assert by_id[connection_id].credential_version == 1
        assert by_id[connection_id].refresh_attempt_count == 1
        assert by_id[connection_id].refresh_retry_after is not None
    for connection_id in all_ids - failure_ids:
        assert by_id[connection_id].credential_version == 2
        assert by_id[connection_id].refresh_attempt_count == 0
        assert by_id[connection_id].refresh_retry_after is None
